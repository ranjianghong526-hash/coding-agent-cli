"""会话级后台 job：异步启动命令、输出落盘、跟踪完成、清理进程树。"""
from __future__ import annotations

import asyncio
import os
import secrets
import signal
import string
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

JOBS_ROOT = Path.home() / ".my-claude-code" / "jobs"
_ID_ALPHABET = string.digits + string.ascii_lowercase
JobStatus = Literal["running", "completed", "failed", "killed"]
EVENT_WINDOW = 10.0
MAX_EVENTS_PER_WINDOW = 20
MAX_PENDING_EVENTS = 200
MAX_DROPPED_EVENTS = 100
MAX_EVENT_CHARS = 500


def _process_options() -> dict:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return {"env": env, **({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
                            if os.name == "nt" else {"start_new_session": True})}


def _new_job_id(prefix: str = "b") -> str:
    return prefix + "".join(secrets.choice(_ID_ALPHABET) for _ in range(8))


def _kill_process_tree(pid: int, *, force: bool = False) -> None:
    """只接收本注册表创建的 PID；Windows 关闭窗口，POSIX 杀独立进程组。"""
    if os.name == "nt":
        import ctypes
        # 保存原进程句柄：taskkill 与命令自然退出可能同时发生，失败后检查
        # 原进程是否确已退出，而非仅按退出码忽略错误，也避免 PID 重用误判。
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x100000, False, pid)
        if not handle:
            if ctypes.get_last_error() == 87:  # PID 已不存在
                return
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            result = subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW, timeout=5,
            )
            if result.returncode and kernel.WaitForSingleObject(handle, 0) != 0:
                detail = result.stderr.decode(errors="replace").strip()
                raise OSError(f"终止进程树失败（taskkill exit code {result.returncode}）：{detail}")
        finally:
            kernel.CloseHandle(handle)
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
    result: str | None = None
    pending_events: list[str] = field(default_factory=list)
    dropped_events: int = 0
    dropped_total: int = 0
    window_start: float = field(default_factory=time.monotonic)
    window_count: int = 0

    def push_event(self, line: str) -> None:
        now = time.monotonic()
        if now - self.window_start >= EVENT_WINDOW:
            self.window_start, self.window_count = now, 0
        self.window_count += 1
        if self.window_count > MAX_EVENTS_PER_WINDOW or len(self.pending_events) >= MAX_PENDING_EVENTS:
            self.dropped_events += 1
            self.dropped_total += 1
            return
        self.pending_events.append(line[:MAX_EVENT_CHARS])

    def summary(self) -> str:
        status = {"completed": "执行成功", "failed": "执行失败", "killed": "已被终止"}.get(self.status, "运行中")
        code = f"，exit code {self.returncode}" if self.returncode is not None else ""
        label = {"agent": "sub agent", "monitor": "monitor"}.get(self.kind, "后台命令")
        return f"{label}「{self.description}」{status}{code}"


class JobRegistry:
    """每个会话一份；asyncio watcher 与 UI/工具在同一事件循环内修改状态。"""
    def __init__(self, session_id: str, on_completed: Callable[[Job], None] | None = None):
        # 子 Agent 使用 主会话ID/agentID；逐段校验，禁止绝对路径和 .. 路径逃逸。
        if not session_id or any(not segment or any(c not in string.ascii_letters + string.digits + "-_" for c in segment)
                                 for segment in session_id.split("/")):
            raise ValueError("无效的 job 会话编号")
        self.session_id = session_id
        self._jobs_dir = JOBS_ROOT / session_id
        self._jobs: dict[str, Job] = {}
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._watchers: set[asyncio.Task] = set()
        self._on_completed = on_completed
        self._closing = False

    async def spawn_agent(self, description: str, run) -> Job:
        """注册后台协程；与 shell 一样由 watcher 统一更新状态并发送通知。"""
        if self._closing:
            raise RuntimeError("当前会话正在关闭，不能再启动 agent")
        self._jobs_dir.mkdir(parents=True, exist_ok=True)
        while True:
            job_id = _new_job_id("a")
            log_path = self._jobs_dir / f"{job_id}.log"
            try:
                with log_path.open("x", encoding="utf-8"):
                    pass
                break
            except FileExistsError:
                continue
        job = Job(job_id, "agent", description, log_path)
        self._jobs[job_id] = job
        task = asyncio.create_task(run(job))
        job.kill_func = task.cancel
        watcher = asyncio.create_task(self._watch_agent(job, task))
        self._watchers.add(watcher)
        watcher.add_done_callback(self._watchers.discard)
        return job

    async def _watch_agent(self, job: Job, task: asyncio.Task) -> None:
        try:
            await task
            if job.status != "killed":
                job.status = "completed"
        except asyncio.CancelledError:
            job.status = "killed"
        except Exception as error:
            # 后台协程是异常边界：将失败交回主模型，不留下无人领取的 task 异常。
            job.status = "failed"
            job.result = f"sub agent 执行失败（{type(error).__name__}）：{error}"
            import traceback
            with job.log_path.open("a", encoding="utf-8") as log:
                traceback.print_exception(error, file=log)
        if not self._closing and self._on_completed:
            self._on_completed(job)

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
        try:
            proc = await asyncio.create_subprocess_shell(
                command, stdout=log_file, stderr=subprocess.STDOUT, **_process_options(),
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

    async def spawn_monitor(self, command: str, description: str, timeout: float | None) -> Job:
        if self._closing:
            raise RuntimeError("当前会话正在关闭，不能再启动 monitor")
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout 必须大于 0")
        self._jobs_dir.mkdir(parents=True, exist_ok=True)
        while True:
            job_id = _new_job_id("m")
            log_path = self._jobs_dir / f"{job_id}.log"
            try:
                with log_path.open("x", encoding="utf-8"):
                    pass
                break
            except FileExistsError:
                continue
        proc = await asyncio.create_subprocess_shell(command, stdout=asyncio.subprocess.PIPE,
                                                     stderr=subprocess.STDOUT, limit=1024 * 1024,
                                                     **_process_options())
        job = Job(job_id, "monitor", description, log_path, notified=True,
                  kill_func=lambda: _kill_process_tree(proc.pid))
        self._jobs[job_id], self._processes[job_id] = job, proc
        watcher = asyncio.create_task(self._pump_monitor(job, proc, timeout))
        self._watchers.add(watcher)
        watcher.add_done_callback(self._watchers.discard)
        return job

    def _stop_monitor(self, job: Job, note: str) -> None:
        if job.status != "running":
            return
        self.kill(job.id, notify=False)
        # 控制事件不受输出限流限制，确保模型得知为什么停止。
        job.pending_events.append(f"[{note}]")

    async def _monitor_timeout(self, job: Job, timeout: float) -> None:
        await asyncio.sleep(timeout)
        self._stop_monitor(job, f"monitor 已停止：到了 {timeout:g} 秒的超时上限，需要时重新挂载")

    async def _pump_monitor(self, job: Job, proc, timeout: float | None) -> None:
        timer = asyncio.create_task(self._monitor_timeout(job, timeout)) if timeout is not None else None
        try:
            with job.log_path.open("a", encoding="utf-8", buffering=1) as log:
                try:
                    async for raw in proc.stdout:
                        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                        log.write(line + "\n")
                        if not line.strip() or job.status != "running":
                            continue
                        job.push_event(line)
                        if job.dropped_total >= MAX_DROPPED_EVENTS:
                            self._stop_monitor(job, "monitor 已停止：输出太多，请用更精确的过滤条件重新挂载")
                except ValueError:
                    self._stop_monitor(job, "monitor 已停止：单行输出过长，请先在命令中截断或过滤")
                    # 管道过长行会暂停 transport；终止后仍需排空，避免 proc.wait 卡住。
                    while raw := await proc.stdout.read(65536):
                        log.write(raw.decode("utf-8", errors="replace"))
                job.returncode = await proc.wait()
                if job.status != "killed":
                    job.status = "completed" if job.returncode == 0 else "failed"
                    job.pending_events.append(f"[monitor 已停止：被监控的命令自己退出了，exit code {job.returncode}]")
        except (OSError, RuntimeError) as error:
            # 后台 IO 边界：失败也作为事件交回，不能遗留无人领取的异常或进程。
            if job.status == "running":
                self.kill(job.id, notify=False)
            job.status = "failed"
            job.pending_events.append(f"[monitor 读取失败（{type(error).__name__}）：{error}]")
        finally:
            if timer:
                timer.cancel()
                await asyncio.gather(timer, return_exceptions=True)
        if not self._closing and self._on_completed:
            self._on_completed(job)

    def pop_events(self) -> list[tuple[Job, list[str], int]]:
        out = []
        for job in self._jobs.values():
            if job.kind == "monitor" and (job.pending_events or job.dropped_events):
                out.append((job, job.pending_events, job.dropped_events))
                job.pending_events = []
                job.dropped_events = 0
        return out

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
        jobs = [job for job in self.running() if job.kind == "shell" and not job.background]
        for job in jobs:
            job.background = True
        return jobs

    def kill(self, job_id: str, *, notify: bool = True) -> bool:
        job = self.get(job_id)
        if job is None or job.status != "running":
            return False
        if job.kill_func:
            job.kill_func()
        job.status = "killed"
        if job.kind == "monitor" and notify and not self._closing:
            job.pending_events.append("[monitor 已停止：用户或 Agent 请求终止]")
        return True

    def pop_unnotified(self) -> list[Job]:
        jobs = [job for job in self._jobs.values()
                if job.kind != "monitor" and job.background and job.status != "running" and not job.notified]
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
