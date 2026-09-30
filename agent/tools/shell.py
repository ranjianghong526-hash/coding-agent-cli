"""
shell 工具：run_command，命中高危特征时通过自检强制走审批。
"""
import re
import asyncio
import time

from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry

from ..deps import AgentDeps

import permissions


FOREGROUND_TIMEOUT = 10


async def run_command(ctx: RunContext[AgentDeps], command: str, run_in_background: bool = False) -> str:
    """
    执行 shell 命令。长驻服务、长测试等用 run_in_background=True，立即返回 job ID 和日志路径。
    后台完成会收到 task-notification，不必原地轮询等待；命令末尾不需要加 &。
    前台命令最多等 10 秒，用户可按 Ctrl+B 将它转后台。
    """
    try:
        registry = ctx.deps.job_registry
        if registry is None:
            raise ModelRetry("当前执行上下文没有 job 注册表，无法运行命令")
        job = await registry.spawn_shell(command, background=run_in_background)
    except OSError as error:
        return f"[错误] 无法执行命令（{error}）"
    if run_in_background:
        return _background_notice(job, f"已放入后台运行，job id: {job.id}")
    start = time.monotonic()
    try:
        while job.status == "running" and not job.background:
            if time.monotonic() - start > FOREGROUND_TIMEOUT:
                registry.kill(job.id)
                return (f"[错误] 命令执行超时（{FOREGROUND_TIMEOUT}秒）；"
                        "长驻或耗时命令请改用 run_in_background=True 重新执行")
            await asyncio.sleep(0.05)
    except asyncio.CancelledError:
        # Esc 打断前台等待时一并终止命令；已转后台的命令由注册表继续管理。
        if not job.background:
            registry.kill(job.id)
        raise
    if job.background:
        return _background_notice(job, f"用户把这条命令转入了后台，它作为 job {job.id} 继续运行")
    output = job.log_path.read_text(encoding="utf-8", errors="replace")
    if job.returncode != 0:
        output += f"\n[错误] exit code {job.returncode}"
    return output or "(无输出)"


def _background_notice(job, lead: str) -> str:
    return (f"{lead}\n输出日志：{job.log_path}（随时可用 read_file 查看）\n"
            "完成后你会收到 <task-notification> 通知，不要原地等待。")


def job_kill(ctx: RunContext[AgentDeps], job_id: str) -> str:
    """终止当前会话自己启动的 job 及其进程树，不能操作任意系统 PID。"""
    registry = ctx.deps.job_registry
    job = registry.get(job_id) if registry else None
    if job is None:
        raise ModelRetry(f"job {job_id} 不存在")
    if not registry.kill(job_id):
        return f"job {job_id} 已经结束（{job.status}），无需终止"
    return f"job {job_id} 已终止"


# 高危命令的特征：删除文件、提权、直写磁盘
DANGEROUS_PATTERNS = [
    r"\brm\b",
    r"\bsudo\b",
    r"\bdd\b",
    r"\bmkfs\w*\b",
]


def run_command_self_check(args: dict):
    """
    run_command 的权限自检：扫一遍命令字符串，命中高危特征就要求审批。
    """
    command = args.get("command", "")
    if any(re.search(pattern, command) for pattern in DANGEROUS_PATTERNS):
        return "ask"
    # 没命中高危特征，交给通用规则决定
    return None


# 把自检挂到权限模块的注册表上
permissions.register_self_check("run_command", run_command_self_check)
