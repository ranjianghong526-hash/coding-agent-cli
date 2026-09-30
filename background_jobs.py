"""会话级后台 job：异步启动命令、输出落盘、跟踪完成、清理进程树。"""
from __future__ import annotations

import asyncio
import os
import secrets
import signal
import string
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

JOBS_ROOT = Path.home() / ".my-claude-code" / "jobs"
_ID_ALPHABET = string.digits + string.ascii_lowercase
JobStatus = Literal["running", "completed", "failed", "killed"]


def _new_job_id() -> str:
    return "b" + "".join(secrets.choice(_ID_ALPHABET) for _ in range(8))


def _kill_process_tree(pid: int, *, force: bool = False) -> None:
    """只接收本注册表创建的 PID；Windows 关闭窗口，POSIX 杀独立进程组。"""
    if os.name == "nt":
        result = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW, timeout=5,
        )
        # 进程恰好已经结束时无需报错；其他失败保留错误，不能假装已终止。
        if result.returncode and result.returncode != 128:
            detail = result.stderr.decode(errors="replace").strip()
            raise OSError(f"终止进程树失败（taskkill exit code {result.returncode}）：{detail}")
    else:
        try:
            # spawn 使用 start_new_session=True，组 ID 就是 shell PID。
            os.killpg(pid, signal.SIGKILL if force else signal.SIGTERM)
        except ProcessLookupError:
            pass


@dataclass
class Job:
    id: str
    kind: str
    description: str
    log_path: Path
    status: JobStatus = "running"
    returncode: int | None = None
    notified: bool = False
    background: bool = True
    kill_func: Callable[[], None] | None = None

    def summary(self) -> str:
        status = {"completed": "执行成功", "failed": "执行失败", "killed": "已被终止"}.get(self.status, "运行中")
        code = f"，exit code {self.returncode}" if self.returncode is not None else ""
        return f"后台命令「{self.description}」{status}{code}"


class JobRegistry:
    """每个会话一份；asyncio watcher 与 UI/工具在同一事件循环内修改状态。"""
    def __init__(self, session_id: str, on_completed: Callable[[Job], None] | None = None):
        if not session_id or any(c not in string.ascii_letters + string.digits + "-_" for c in session_id):
            raise ValueError("无效的 job 会话编号")
        self._jobs_dir = JOBS_ROOT / session_id
        self._jobs: dict[str, Job] = {}
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._watchers: set[asyncio.Task] = set()
        self._on_completed = on_completed
        self._closing = False

    async def spawn_shell(self, command: str, background: bool = True) -> Job:
        if self._closing:
            raise RuntimeError("当前会话正在关闭，不能再启动命令")
        self._jobs_dir.mkdir(parents=True, exist_ok=True)
        # 随机 ID 加独占创建，发生碰撞也绝不覆盖旧日志。
        while True:
            job_id = _new_job_id()
            log_path = self._jobs_dir / f"{job_id}.log"
            try:
                log_file = log_path.open("xb")
                break
            except FileExistsError:
                continue
        options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
                   if os.name == "nt" else {"start_new_session": True})
        env = os.environ.copy()
        # Windows 下 Python 重定向输出默认可能是 GBK，文件工具按 UTF-8 读取日志。
        env.setdefault("PYTHONIOENCODING", "utf-8")
        try:
            proc = await asyncio.create_subprocess_shell(
                command, stdout=log_file, stderr=subprocess.STDOUT, env=env, **options,
            )
        except BaseException:
            log_file.close()
            raise
        job = Job(job_id, "shell", command, log_path, background=background,
                  kill_func=lambda: _kill_process_tree(proc.pid))
        self._jobs[job_id], self._processes[job_id] = job, proc
        watcher = asyncio.create_task(self._watch(job, proc, log_file))
        self._watchers.add(watcher)
        watcher.add_done_callback(self._watchers.discard)
        return job

    async def _watch(self, job: Job, proc, log_file) -> None:
        try:
            job.returncode = await proc.wait()
            if job.status != "killed":
                job.status = "completed" if job.returncode == 0 else "failed"
        finally:
            log_file.close()
        if job.background and not self._closing and self._on_completed:
            self._on_completed(job)

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return list(self._jobs.values())

    def running(self) -> list[Job]:
        return [job for job in self._jobs.values() if job.status == "running"]

    def background_foreground(self) -> list[Job]:
        jobs = [job for job in self.running() if not job.background]
        for job in jobs:
            job.background = True
        return jobs

    def kill(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None or job.status != "running":
            return False
        if job.kill_func:
            job.kill_func()
        job.status = "killed"
        return True

    def pop_unnotified(self) -> list[Job]:
        jobs = [job for job in self._jobs.values()
                if job.background and job.status != "running" and not job.notified]
        for job in jobs:
            job.notified = True
        return jobs

    def shutdown(self) -> int:
        self._closing = True
        jobs = self.running()
        for job in jobs:
            self.kill(job.id)
        return len(jobs)

    async def aclose(self) -> None:
        """清理时等待 watcher 关闭文件；POSIX 不响应 TERM 时补发 KILL。"""
        self.shutdown()
        watchers = list(self._watchers)
        if watchers:
            _, pending = await asyncio.wait(watchers, timeout=5)
            if pending:
                for proc in self._processes.values():
                    if proc.returncode is None:
                        _kill_process_tree(proc.pid, force=True)
                await asyncio.gather(*pending)
