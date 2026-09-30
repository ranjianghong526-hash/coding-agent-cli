"""子 Agent 类型、独立执行器与后台审批队列；不向主对话写入中间消息。"""
import asyncio
import dataclasses
import platform
from collections import deque
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from pydantic_ai import Agent, Tool
from pydantic_ai.capabilities import Hooks
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.usage import UsageLimits
from pydantic_graph import End

import classifier
import permissions
from agent.deps import AgentDeps
from agent.file_state import ReadFileState
from agent.hooks import _retry_on_error
from agent.model import model
from agent.reminders import build_job_reminder_text
from agent.tools.file import edit_file, read_file, write_file
from agent.tools.shell import run_command
from background_jobs import Job, JobRegistry
from memory.store import _parse_frontmatter

MAX_SUBAGENT_REQUESTS = 40
sub_hooks = Hooks()


@dataclass
class PendingApproval:
    job: Job
    tool_name: str
    args: dict
    future: asyncio.Future


PENDING_APPROVALS: deque[PendingApproval] = deque()


async def _request_user_approval(job: Job, tool_name: str, args: dict) -> str:
    future = asyncio.get_running_loop().create_future()
    request = PendingApproval(job, tool_name, args, future)
    PENDING_APPROVALS.append(request)
    try:
        # 暂停的是这一条子 Agent 协程，不占着模型请求等待用户。
        return await future
    finally:
        if request in PENDING_APPROVALS:
            PENDING_APPROVALS.remove(request)


def pop_pending_approval() -> PendingApproval | None:
    while PENDING_APPROVALS:
        request = PENDING_APPROVALS.popleft()
        if not request.future.done() and request.job.status == "running":
            return request
    return None


@sub_hooks.on.tool_execute
async def _check_sub_permission(ctx, *, call, tool_def, args, handler):
    if permissions.compute_decision(call.tool_name, args) == "allow":
        return await handler(args)
    if permissions.state.mode == permissions.AUTO:
        # 委派 prompt 是主模型写的，不能伪装成人类授权。保留父会话用户原话
        # 和子 Agent 的工具调用用于审查，不把子 Agent 的初始 prompt 当授权。
        messages = list(ctx.deps.user_authorization or [])
        messages += [message for message in ctx.messages if message.kind == "response"]
        verdict = await classifier.classify(messages, call.tool_name, args)
        if not verdict.get("error"):
            if not verdict["should_block"]:
                return await handler(args)
            return f"安全检查拦截了这次 {call.tool_name}，没有执行。理由：{verdict['reason']}。不要绕过拦截，请在报告中说明。"
    choice = await _request_user_approval(ctx.deps.subagent_job, call.tool_name, args)
    if choice == "always":
        permissions.state.session_allowed.add(call.tool_name)
    if choice in ("once", "always"):
        return await handler(args)
    return f"用户拒绝了这次 {call.tool_name} 调用，没有执行。不要绕过拒绝，请在报告中说明未完成的步骤。"


@sub_hooks.on.before_model_request
async def _inject_job_reminders(ctx, request_context):
    text = build_job_reminder_text(ctx.deps.job_registry)
    if not text:
        return request_context
    notification = ModelRequest(parts=[UserPromptPart(text)], metadata={"origin": "dynamic-reminder"})
    return dataclasses.replace(request_context, messages=list(request_context.messages) + [notification])


@sub_hooks.on.model_request
async def _retry(ctx, *, request_context, handler):
    return await _retry_on_error(ctx, request_context=request_context, handler=handler)


@sub_hooks.on.tool_execute_error
async def _tool_error(ctx, *, call, tool_def, args, error):
    # 工具失败通过 tool result 写入子 Agent 日志，不刷主 Agent 的输出区。
    return f"工具 {call.tool_name} 执行失败（{type(error).__name__}）：{error}"


EXPLORE_INSTRUCTIONS = (
    "你是只读代码调查 agent，搜索、阅读代码，回答交给你的问题。严禁修改文件，"
    "run_command 只用于查看、搜索，不要执行有副作用的命令。\n"
    "结束后输出自包含报告：结论在前，附关键文件路径。调用方看不到你的中间过程。"
)
GENERAL_INSTRUCTIONS = (
    "你是主 agent 派出的 sub agent，独立完成任务，可以读写文件和执行命令。"
    "修改已有文件前必须 read_file；局部修改优先 edit_file，新建或整体重写使用 write_file。"
    "没有人回答你的需求澄清，拿不准时合理取舍并在报告里说明。"
    "耗时命令可用 run_in_background=True，只有下一次模型请求才会收到它的通知；"
    "不要在依赖该命令结果的任务尚未完成时提前结束报告。"
    "最终报告写清做了什么、改了哪些文件、验证与关键发现，调用方看不到中间过程。"
)
_TOOL_FUNCS = {"read_file": read_file, "edit_file": edit_file,
               "write_file": write_file, "run_command": run_command}


@dataclass
class AgentType:
    name: str
    description: str
    tool_names: list[str]
    instructions: str
    source: str
    agent: Agent = field(init=False, repr=False)


_TYPES: dict[str, AgentType] = {}


def _sub_env_context() -> str:
    parts = [f"工作目录：{Path.cwd()}", f"操作系统：{platform.system()}", f"日期：{date.today().isoformat()}"]
    path = Path.cwd() / "AGENTS.md"
    if path.is_file():
        parts.append("项目 AGENTS.md 约定：\n" + path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def _register(atype: AgentType) -> None:
    unknown = set(atype.tool_names) - _TOOL_FUNCS.keys()
    if unknown:
        raise ValueError("不支持的子 Agent 工具：" + ", ".join(sorted(unknown)))
    tools = [Tool(_TOOL_FUNCS[name], sequential=name in {"edit_file", "write_file"}) for name in atype.tool_names]
    atype.agent = Agent(model, instructions=[atype.instructions, _sub_env_context],
                        tools=tools, deps_type=AgentDeps, capabilities=[sub_hooks])
    _TYPES[atype.name] = atype


def load_agent_types(directory: Path | None = None) -> list[str]:
    """启动时重建类型表；自定义错误报告给用户，不阻止内置 Agent 使用。"""
    _TYPES.clear()
    _register(AgentType("explore", "只读搜索、调查代码与结构，不修改文件", ["read_file", "run_command"], EXPLORE_INSTRUCTIONS, "built-in"))
    _register(AgentType("general", "读写文件和执行命令，适合独立修复、补测试", list(_TOOL_FUNCS), GENERAL_INSTRUCTIONS, "built-in"))
    errors = []
    directory = directory if directory is not None else Path.cwd() / ".my-claude-code" / "agents"
    for path in sorted(directory.glob("*.md")):
        try:
            fields = _parse_frontmatter(path)
            name = fields.get("name", "").strip()
            description = fields.get("description", "").strip()
            if not name or not description:
                raise ValueError("name 和 description 必填")
            if name in _TYPES:
                raise ValueError(f"agent 类型 {name} 重名")
            text = path.read_text(encoding="utf-8")
            lines = text.splitlines()
            closing = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
            body = "\n".join(lines[closing + 1:]).strip()
            if not body:
                raise ValueError("必须提供 Agent 指令正文")
            tool_names = ([item.strip() for item in fields["tools"].split(",") if item.strip()]
                          if "tools" in fields else list(_TOOL_FUNCS))
            _register(AgentType(name, description, list(dict.fromkeys(tool_names)), body, str(path.resolve())))
        except (OSError, UnicodeError, ValueError, StopIteration) as error:
            errors.append(f"{path.name}：{error}")
    return errors


def get_agent_type(name: str) -> AgentType | None:
    return _TYPES.get(name)


def list_agent_types() -> list[AgentType]:
    return list(_TYPES.values())


def agent_types_prompt() -> str:
    return "run_agent 可用的 agent 类型：\n" + "\n".join(
        f"- {item.name}：{item.description}（可用工具：{', '.join(item.tool_names)}）" for item in list_agent_types())


def _log_parts(log, parts) -> None:
    for part in parts:
        content = getattr(part, "content", None)
        if part.part_kind == "tool-call":
            content = f"{part.tool_name}({part.args})"
        elif part.part_kind in {"tool-return", "retry-prompt"}:
            content = f"{getattr(part, 'tool_name', '')} -> {content}"
        log.write(f"[{part.part_kind}] {content}\n")
    log.flush()


async def run_subagent(atype: AgentType, prompt: str, job: Job, parent_deps: AgentDeps) -> None:
    sub_registry = JobRegistry(f"{parent_deps.job_registry.session_id}/{job.id}")
    deps = dataclasses.replace(parent_deps, read_file_state=ReadFileState(), tasks_store=None,
                               job_registry=sub_registry, subagent_job=job)
    try:
        with job.log_path.open("a", encoding="utf-8") as log:
            log.write(f"=== sub agent {job.id} ({atype.name}) ===\nprompt: {prompt}\n\n")
            log.flush()
            # 不传主会话 message_history：探索、试错的消息只留在这个 run 中。
            async with atype.agent.iter(prompt, deps=deps,
                                        usage_limits=UsageLimits(request_limit=MAX_SUBAGENT_REQUESTS)) as run:
                node = run.next_node
                while not isinstance(node, End):
                    node = await run.next(node)
                    if Agent.is_call_tools_node(node):
                        _log_parts(log, node.model_response.parts)
                    elif Agent.is_model_request_node(node):
                        _log_parts(log, [p for p in node.request.parts if p.part_kind in {"tool-return", "retry-prompt"}])
            job.result = run.result.output or "(sub agent 没有输出报告)"
            log.write(f"\n=== 最终报告 ===\n{job.result}\n")
    finally:
        # 子 Agent 的 shell job 跟着它收尾，取消 Agent 也必须等待进程树清理。
        await sub_registry.aclose()


# 内置类型在导入时可用；main 启动时再加载当前项目的自定义定义。
load_agent_types()
