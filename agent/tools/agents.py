"""主 Agent 的委派入口；工具只负责注册后台 job，执行交给 subagents。"""
import dataclasses

from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry

from ..deps import AgentDeps
from .shell import _background_notice


async def run_agent(ctx: RunContext[AgentDeps], description: str, prompt: str,
                    agent_type: str = "general") -> str:
    """将独立任务交给子 Agent，立即返回 job ID；完成通知携带最终报告。

    子 Agent 不知道主对话，请在 prompt 中完整说明任务背景、目标和约束。
    不要原地轮询；收到 <task-notification> 的 <result> 后转述给用户。
    """
    # 延迟导入：subagents 的工具表依赖 file/shell，避免包初始化环。
    import subagents
    atype = subagents.get_agent_type(agent_type)
    if atype is None:
        raise ModelRetry(f"agent 类型 {agent_type} 不存在，可用类型：" + ", ".join(t.name for t in subagents.list_agent_types()))
    registry = ctx.deps.job_registry
    if registry is None:
        raise ModelRetry("当前会话没有 job 注册表")
    deps = dataclasses.replace(ctx.deps, user_authorization=list(ctx.messages))
    job = await registry.spawn_agent(description, lambda job: subagents.run_subagent(atype, prompt, job, deps))
    return _background_notice(job, f"sub agent 已在后台开始工作，job id: {job.id}，完成通知会在 <result> 字段附带最终报告")
