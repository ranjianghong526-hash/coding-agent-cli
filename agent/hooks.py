"""
挂在 Agent 上的 hooks，用来抓每次 model API 调用的元数据。

主循环在每轮 run_agent 之前清空 api_call_log，跑完后快照到 SessionState 里，
/api-detail 命令再把这一轮的所有调用展示给用户。

hook（钩子）是框架在指定时机自动调用的函数：请求前检查文件变化并记录日志，请求失败时补日志，
工具执行前审批，工具未知异常时转换为 ModelRetry，让模型有机会修正操作。
日志只记录模型调用的摘要，没有保存完整的 HTTP 请求体和响应体。
"""
import asyncio
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic_ai.capabilities import Hooks
from pydantic_ai import ModelRetry
from pydantic_ai.exceptions import SkipToolExecution

from permissions import PermissionState, check_permission
from context_injection import collect_external_changes, make_system_reminder


# dataclass 根据字段自动生成初始化方法等，适合保存结构明确的一条调用记录。
@dataclass
class ApiCall:
    """
    一次 model API 调用的元数据。before_model_request 创建并填充上半部分，
    after_model_request 填充下半部分。
    """
    # 请求侧：发给哪个模型、带了多少条历史消息、提供了哪些工具。
    model: str
    messages_count: int
    # 这次发送给模型的 messages 中最后一条消息的最后一个 part
    last_part: Any
    # 这里保存可用工具的名称，不表示这些工具在本次请求中都被调用了。
    tools: list
    # 响应侧默认为空，在 after hook 收到模型返回后再补充。
    finish_reason: str = ""
    # default_factory 每次创建独立列表，避免不同 ApiCall 共享可变数据。
    parts_kinds: list = field(default_factory=list)
    # tokens 是模型处理文本的计量单位，这里分别记录输入和生成输出的用量。
    input_tokens: int = 0
    output_tokens: int = 0


# 主循环在每轮 run_agent 之前清空它
# 全局列表用于当前串行 CLI；若改成同时执行多个任务，需要按任务隔离记录。
api_call_log: list[ApiCall] = []

# 这个对象通过 core.py 的 capabilities=[hooks] 挂到 Agent 上。
hooks = Hooks()


@hooks.on.before_tool_execute
async def _approve_tool(ctx, *, call, tool_def, args):
    """参数校验完成、工具尚未执行时审批，拒绝后不会发生文件或命令副作用。"""
    if not isinstance(ctx.deps, PermissionState):
        # 调用方必须明确传入权限状态；漏传时不能绕过审批直接执行工具。
        raise SkipToolExecution("[权限拒绝] 缺少权限上下文，工具未执行。")
    # ctx.messages 包括本轮真实用户输入和模型刚提出的工具调用，不能只传上轮历史。
    await check_permission(ctx.deps, call.tool_name, args, ctx.messages)
    return args


@hooks.on.before_model_request
async def _record_request(ctx, request_context):
    """
    每次发起 model 调用之前，创建一条 ApiCall 记录。
    """
    # SDK 在每次模型请求前调用，包括同一轮工具完成后的再次请求。
    if isinstance(ctx.deps, PermissionState):
        reminder = await asyncio.to_thread(collect_external_changes, ctx.deps.files)
        # 聊天历史可能很长；每次请求都从独立状态生成最新任务清单。
        task_reminder = ctx.deps.tasks.reminder()
        reminder = "\n\n".join(text for text in (reminder, task_reminder) if text)
        if reminder:
            # 创建新列表；SDK 会将处理后的消息保存为本轮真实历史。
            request_context = replace(
                request_context,
                messages=request_context.messages + [make_system_reminder(reminder)],
            )
    msgs = list(request_context.messages)
    # 一条消息可以含多个 part（内容片段），如文本、工具调用或工具结果。
    # 先判断列表是否为空，避免用 [-1] 访问不存在的最后一项。
    last_part = msgs[-1].parts[-1] if msgs and msgs[-1].parts else None
    try:
        # 本次请求传给模型的工具定义，提取名称作为便于阅读的摘要。
        tool_names = [t.name for t in request_context.model_request_parameters.function_tools]
    except AttributeError:
        # 若当前请求没有预期的工具属性，日志按空工具列表记录。
        tool_names = []
    # 先创建只有请求侧数据的记录，响应钩子会补齐同一个对象。
    api_call_log.append(ApiCall(
        model=request_context.model.model_name,
        messages_count=len(msgs),
        last_part=last_part,
        tools=tool_names,
    ))
    # 日志在注入之后记录；SDK 随后还会合并相邻请求并转换为提供方格式。
    return request_context


@hooks.on.after_model_request
async def _record_response(ctx, request_context, response):
    """
    每次 model 调用返回后，填充上面这条 ApiCall 的 response 字段。
    """
    if api_call_log:
        # 当前串行流程下，最后一条记录对应刚返回的模型请求。
        # 这不是通过请求 ID 匹配，因此不能直接用于并发任务的日志关联。
        call = api_call_log[-1]
        # finish_reason 描述模型为何结束；缺失时使用明确的 unknown 占位。
        call.finish_reason = str(response.finish_reason) if response.finish_reason else "unknown"
        # 只保存片段类型，例如 text / tool-call，不在这里渲染具体内容。
        call.parts_kinds = [p.part_kind for p in response.parts]
        call.input_tokens = response.usage.input_tokens
        call.output_tokens = response.usage.output_tokens
    # 原样返回响应，日志收集不改变后续的工具调度和结果处理。
    return response


@hooks.on.model_request_error
async def _record_request_error(ctx, request_context, error):
    """请求最终失败时补全日志，再原样抛出，交给 CLI 安全网处理。"""
    if api_call_log:
        # 不记录原始响应体，避免把接口错误中的敏感信息展示到 /api-detail。
        status = getattr(error, "status_code", None)
        api_call_log[-1].finish_reason = f"error: {status or type(error).__name__}"
    raise error


@hooks.on.tool_execute_error
async def _recover_tool_error(ctx, *, call, tool_def, args, error):
    """第二道工具防线：把未预料到的异常变成有次数限制的模型修正提示。"""
    if isinstance(error, ModelRetry):
        # 工具自己提供的修正信息保持不变，不覆盖成通用提示。
        raise error
    # 此处是框架明确提供的异常边界；不吞异常、不返回成功，也不输出原始异常内容。
    # ModelRetry 会生成 retry-prompt 交给模型，由模型选择改参数、换工具或解释失败。
    raise ModelRetry(
        f"工具 {call.tool_name} 执行失败（{type(error).__name__}）。"
        "请检查参数和调用方式，修正后再尝试；无法修复时向用户说明失败。"
    ) from error
