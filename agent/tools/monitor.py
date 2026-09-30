"""持续事件工具，权限与 shell 命令相同；只对主 Agent 开放。"""
from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry

import permissions
from ..deps import AgentDeps
from .shell import run_command_self_check


async def monitor(ctx: RunContext[AgentDeps], command: str, description: str,
                  timeout: int = 300, persistent: bool = False) -> str:
    """启动持续监控，将命令的非空输出行作为 <monitor-event> 推送。

    每次发生都要通知才用 monitor；只需一次完成通知使用 run_command 的后台模式。
    command 必须自行过滤事件并覆盖失败信号；每一级输出须及时 flush。
    POSIX grep 加 --line-buffered，上游 stderr 也要过滤时写 cmd 2>&1 | grep ...。
    Windows 使用 PowerShell 或 Python，不能假设存在 tail/grep；Python 使用 -u。
    未输出并不代表成功，过滤应考虑 Traceback、Error、FAILED 等失败信号。
    默认 300 秒，persistent=True 不设计时上限，但仍受限流、熔断和会话清理约束。
    启动后立即返回，不要原地等待，可用 job_kill 停止。
    """
    if timeout <= 0:
        raise ModelRetry("timeout 必须是正整数")
    registry = ctx.deps.job_registry
    if registry is None:
        raise ModelRetry("当前会话没有 job 注册表")
    job = await registry.spawn_monitor(command, description, None if persistent else timeout)
    return (f"monitor 已启动，job id: {job.id}\n输出日志：{job.log_path}（可用 read_file 查看已记录输出）\n"
            "非空输出行会通过 <monitor-event> 通知送达。不要原地等待；用 job_kill 可以随时停掉它。")


permissions.register_self_check("monitor", run_command_self_check)
