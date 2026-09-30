"""
挂在 Agent 上的 hooks：
1. API 调用元数据记录（/api-detail 命令用）
2. system-reminder 注入（每次 model 请求前，把过期文件提醒和 task 提醒追加进 messages）
3. API 请求失败时的自动重试（wrap_model_request）
4. 工具执行异常的兜底处理（on_tool_execute_error）
"""
import asyncio
import dataclasses
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai.capabilities import Hooks
from pydantic_ai.exceptions import ModelHTTPError, ModelAPIError
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, UserPromptPart

import classifier
import permissions
from ui.render import console, print_step

from .deps import AgentDeps
from .reminders import build_reminder_text, build_task_reminder_text


MAX_RETRIES = 3

# task_reminder 触发阈值
# 距上次 task_create / task_update 调用 >= 这么多轮才考虑提醒
TASK_REMINDER_TURNS_SINCE_WRITE = 6
# 距上次 task_reminder 注入 >= 这么多轮，避免连续刷屏
TASK_REMINDER_TURNS_BETWEEN = 4

# 哪些工具算"task 管理动作"——调过它们就重置 since_write 计数
_TASK_MANAGEMENT_TOOLS = {"task_create", "task_update"}


@dataclass
class ApiCall:
    """
    一次 model API 调用的元数据。before_model_request 创建并填充上半部分，
    after_model_request 填充下半部分。
    """
    # request 侧
    model: str
    messages_count: int
    # 这次发送给模型的 messages 中最后一条消息的最后一个 part
    last_part: Any
    tools: list
    # response 侧（after hook 填充）
    finish_reason: str = ""
    parts_kinds: list = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


# 主循环在每轮 agent.iter() 之前清空它
api_call_log: list[ApiCall] = []

hooks = Hooks()


# ---------- API 调用记录 ----------

@hooks.on.before_model_request
async def _record_request(ctx, request_context):
    """
    每次发起 model 调用之前，创建一条 ApiCall 记录。
    """
    msgs = list(request_context.messages)
    last_part = msgs[-1].parts[-1] if msgs and msgs[-1].parts else None
    try:
        tool_names = [t.name for t in request_context.model_request_parameters.function_tools]
    except AttributeError:
        tool_names = []
    api_call_log.append(ApiCall(
        model=request_context.model.model_name,
        messages_count=len(msgs),
        last_part=last_part,
        tools=tool_names,
    ))
    return request_context


@hooks.on.after_model_request
async def _record_response(ctx, *, request_context, response):
    """
    每次 model 调用返回后，填充上面这条 ApiCall 的 response 字段。
    """
    if api_call_log:
        call = api_call_log[-1]
        call.finish_reason = str(response.finish_reason) if response.finish_reason else "unknown"
        call.parts_kinds = [p.part_kind for p in response.parts]
        call.input_tokens = response.usage.input_tokens
        call.output_tokens = response.usage.output_tokens
    return response


# ---------- system-reminder 注入 ----------

def _scan_task_turn_counters(messages) -> tuple[int, int]:
    """
    一次反向扫描历史，同时算出 (距上次 task 管理工具多少轮, 距上次 task_reminder 多少轮)。两个结果都拿到就早退，避免长 history 下扫两遍。
    """
    since_mgmt = 0
    since_reminder = 0
    found_mgmt = False
    found_reminder = False
    for msg in reversed(messages):
        if isinstance(msg, ModelResponse):
            if not found_mgmt:
                for part in msg.parts:
                    if isinstance(part, ToolCallPart) and part.tool_name in _TASK_MANAGEMENT_TOOLS:
                        found_mgmt = True
                        break
                if not found_mgmt:
                    since_mgmt += 1
            if not found_reminder:
                since_reminder += 1
        elif isinstance(msg, ModelRequest) and not found_reminder:
            for part in msg.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, str) and _REMINDER_SENTINELS["task"] in part.content:
                    found_reminder = True
                    break
        if found_mgmt and found_reminder:
            break
    return since_mgmt, since_reminder


def _build_file_reminder(ctx, messages) -> str | None:
    # 单次扫描的 readFileState 变更检测；只看 state，不看 messages
    return build_reminder_text(ctx.deps.read_file_state)


def _build_task_reminder(ctx, messages) -> str | None:
    # task reminder 的两道阈值都过了才发：沉默够久 + 上次提醒也够久了
    since_mgmt, since_reminder = _scan_task_turn_counters(messages)
    if since_mgmt < TASK_REMINDER_TURNS_SINCE_WRITE:
        return None
    if since_reminder < TASK_REMINDER_TURNS_BETWEEN:
        return None
    return build_task_reminder_text(ctx.deps.tasks_store)


# 注册要在 before_model_request 触发的 reminder builder：每条 (sentinel, builder)，sentinel 仅用于回扫识别（task reminder 复用）
_REMINDER_SENTINELS = {
    # task reminder 的识别串就是它正文里 builder 必定带的那句话，不再单独嵌一个 marker
    "task": "task 工具最近没有被使用",
}
_REMINDER_BUILDERS = (_build_file_reminder, _build_task_reminder)


@hooks.on.before_model_request
async def _inject_reminders(ctx, request_context):
    """
    依次跑每个注册过的 reminder builder：返回非 None 的就拼到 request_context.messages 末尾。当前两条：过期文件提醒（看 readFileState）+ task 提醒（看 turn 计数 + tasks_store）。要加第 3 种 reminder 时只需多写一个 builder，不必再复制一次 hook 框架。
    """
    messages = list(request_context.messages)
    appended = []
    for builder in _REMINDER_BUILDERS:
        text = builder(ctx, messages + appended)
        if text is None:
            continue
        appended.append(ModelRequest(parts=[UserPromptPart(content=text)]))
        print_step("[dim]◇ system[/]", f"[dim]{text[:200]}[/]")
    if not appended:
        return request_context
    return dataclasses.replace(request_context, messages=messages + appended)


# ---------- API 请求重试 ----------

@hooks.on.model_request
async def _retry_on_error(ctx, *, request_context, handler):
    """
    包裹 model 请求，遇到可重试错误时自动指数退避重试。

    重试在 wrap 内部完成，对话历史和 before/after hooks 不受影响。
    """
    for attempt in range(MAX_RETRIES + 1):
        try:
            return await handler(request_context)
        except ModelHTTPError as e:
            if e.status_code < 500:
                raise
            if attempt >= MAX_RETRIES:
                console.print(f"[bold red]✗ HTTP {e.status_code}，重试 {MAX_RETRIES} 次后仍失败[/]")
                raise
            wait = 2 ** attempt
            console.print(
                f"[bold yellow]⟳ HTTP {e.status_code}，{wait}s 后重试 "
                f"({attempt + 1}/{MAX_RETRIES})...[/]"
            )
            await asyncio.sleep(wait)
        except ModelAPIError as e:
            if attempt >= MAX_RETRIES:
                console.print(
                    f"[bold red]✗ 网络连接失败，重试 {MAX_RETRIES} 次后仍无法连接[/]"
                )
                raise
            wait = 2 ** attempt
            console.print(
                f"[bold yellow]⟳ 网络连接失败，{wait}s 后重试 "
                f"({attempt + 1}/{MAX_RETRIES})...[/]"
            )
            await asyncio.sleep(wait)


# ---------- 工具调用权限检查 ----------

@hooks.on.tool_execute
async def _check_permission(ctx, *, call, tool_def, args, handler):
    """
    工具执行前的权限关卡。allow 就调用 handler 真正执行；
    ask 就弹审批列表；deny 则把拒绝原因当作工具结果回填，让模型自行纠正。
    auto 模式下，ask 不直接弹窗，先交给 LLM classifier 判定。
    """
    decision = permissions.compute_decision(call.tool_name, args)
    if decision == "allow":
        # 放行，handler(args) 才是真正执行工具的那一步
        return await handler(args)

    # decision == "ask" 且当前是 auto 模式：让 classifier 替用户做决定
    if permissions.state.mode == permissions.AUTO:
        # 审查过程对齐成和 thinking、tool_call 一样的「图标 + 标签独占一行、内容换行」格式，用蓝色让用户一眼看到 classifier 在工作
        print_step("[blue]◆ auto_check[/]", f"[blue dim]正在请 LLM 审查 {call.tool_name}...[/]")
        verdict = await classifier.classify(ctx.messages, call.tool_name, args)

        if verdict.get("error"):
            # classifier 自己出错，回退到下面的人工审批弹窗，不能因为审查失败就放行
            print_step("[yellow]⚠ auto_check[/]", f"[yellow]{verdict['reason']}[/]")
        elif not verdict["should_block"]:
            # classifier 判定安全，放行执行，并把理由打出来让用户随时能审计；标签用 ✔ 表示放行
            print_step("[blue]✔ auto_check[/]", f"[blue dim]放行：{verdict['reason']}[/]")
            return await handler(args)
        else:
            # classifier 判定危险：和人工拒绝一样回填给模型，多带上一句拦截理由；标签用 ✘ 表示拦截
            print_step("[red]✘ auto_check[/]", f"[red]拦截：{verdict['reason']}[/]")
            return (
                f"安全检查拦截了这次 {call.tool_name} 调用，没有执行。拦截理由：{verdict['reason']}。"
                "不要尝试绕过拦截，请停下来向用户说明情况，由用户决定接下来怎么做。"
            )

    # default / acceptEdits 模式的 ask，或 auto 模式下 classifier 出错的回退：弹审批让用户决定
    choice = await permissions.prompt_approval(call.tool_name, args)
    if choice == "once":
        return await handler(args)
    if choice == "always":
        # 记进会话白名单，本会话内这个工具不再询问
        permissions.state.session_allowed.add(call.tool_name)
        return await handler(args)

    # 拒绝：不执行工具，把拒绝原因回填给模型，让它停下来等用户发话，而不是自作主张绕过去
    return f"用户拒绝了对 {call.tool_name} 的调用，这次调用没有执行。请停下手上的事，等用户告诉你接下来该怎么做。"


# ---------- 工具执行异常兜底 ----------

@hooks.on.tool_execute_error
async def _handle_tool_error(ctx, *, call, tool_def, args, error):
    """
    工具函数抛出未捕获异常时，不让进程崩溃，
    而是把错误信息作为 tool result 返回给模型，让它自行纠正。
    """
    console.print(f"[bold red]✗ 工具 {call.tool_name} 出错：{error}[/]")
    return f"工具执行出错：{type(error).__name__}: {error}"
