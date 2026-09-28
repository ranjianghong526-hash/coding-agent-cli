"""命令行入口：把终端输入、Agent 执行和结果展示串成完整的一轮交互。

推荐先读 main() 建立全貌，再沿着它调用的三个函数往下看。
这里的 while 循环负责多轮对话；通过 agent.iter() 在节点之间展示模型和工具结果。
"""

import asyncio

from pydantic_ai import Agent, FunctionToolResultEvent
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior

# prompt_toolkit 负责输入体验，模型判断与工具执行不由它处理。
from prompt_toolkit import PromptSession

# 导入 agent 包时会执行 agent/__init__.py，并进一步执行 core.py 的模型配置。
# 因此 API_KEY 校验发生在进入 main() 之前。
from agent import agent, MODEL_NAME, api_call_log
# UI 模块提供命令注册表、会话数据结构和展示函数，入口只负责调度它们。
from ui.commands import (
    COMMANDS,
    SessionState,
    console,
    print_part,
    print_divider,
    print_welcome_banner,
)

# PromptSession 比内置 input() 好用：支持左右移动光标编辑，还会记住本次运行的输入历史，上下方向键可以翻
prompt_session = PromptSession()


async def read_user_input():
    """
    打印上横线并读一行用户输入；回车后再补一条下横线，让输入在滚动历史里保持上下边界。返回 None 表示用户希望退出（Ctrl-C / Ctrl-D）。
    """
    print_divider()
    try:
        # 去掉首尾空白；空输入会由 main() 跳过，不会发送给模型。
        # main() 已运行在事件循环中，使用异步输入，避免同步 prompt() 嵌套事件循环。
        user_input = (await prompt_session.prompt_async("❯ ")).strip()
    except (EOFError, KeyboardInterrupt):
        # 将 Ctrl-D / Ctrl-C 统一转换成 None，主循环据此退出。
        print()
        return None
    print_divider()
    return user_input


def handle_command(user_input, state):
    """
    处理以 / 开头的命令。
    返回 'pass'：不是命令，主循环继续往下走交给 Agent；
    返回 'continue'：命令已处理，主循环跳到下一轮；
    返回 'break'：命令要求退出主循环。
    """
    if not user_input.startswith("/"):
        # 普通自然语言需求继续走 Agent 分支。
        return "pass"
    # 去掉开头的 /，只取第一个词作为命令名；当前命令不解析额外参数。
    words = user_input[1:].split()
    if not words:
        console.print("请输入命令名，例如 /help\n")
        return "continue"
    cmd_name = words[0]
    # 注册表把名字映射到 Command 对象，避免为每条命令写一个 if 分支。
    command = COMMANDS.get(cmd_name)
    if command is None:
        console.print(f"未知命令：/{cmd_name}，输入 /help 查看可用命令\n")
        return "continue"
    # handler 的 bool 返回值是统一约定：True 继续接收输入，False 退出。
    return "continue" if command.handler(state) else "break"


def apply_result(state, result):
    """
    跑完一轮 Agent 后保存历史、用量和调用日志；过程已在执行期间显示。
    """
    # 完整历史包含之前的对话和本轮新增消息，还包括工具请求与工具返回。
    # 下一轮传给模型时，它才能理解“继续修改刚才的文件”这类上下文。
    state.history = result.all_messages()
    # 一轮需求可能触发多次模型请求；这里累计的是整轮的 token 用量。
    # SDK 旧版本提供 usage()，新版本提供 usage 属性；依赖未锁版本，兼容两种接口。
    usage = result.usage
    usage = usage() if callable(usage) else usage
    state.input_tokens += usage.input_tokens
    state.output_tokens += usage.output_tokens
    # 复制列表，避免下一轮 api_call_log.clear() 连带清空上轮保存的列表。
    # 这是浅复制：ApiCall 对象仍共享，但当前串行流程在本轮结束后不再修改它们。
    state.last_api_calls = list(api_call_log)


async def run_agent(user_input: str, state: SessionState):
    """逐节点执行一轮任务，模型返回和工具完成时立刻复用现有 UI 展示。

    iter() 返回异步上下文管理器；节点中的模型请求和工具操作仍由框架执行。
    此处展示完整响应片段，没有消费逐 token 的模型流。
    """
    async with agent.iter(user_input, message_history=state.history) as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node):
                # 这个节点尚未执行，下一次迭代才发出模型请求。
                console.print("[dim]✻ 正在请求模型…[/]")
            elif Agent.is_call_tools_node(node):
                # 模型响应已经返回，但工具尚未执行；先展示正文、思考和工具参数。
                for part in node.model_response.parts:
                    print_part(part)
                # 消费工具事件：每个工具返回时立即显示，不等全部工具或整轮结束。
                # stream() 执行这个节点；后续迭代会使用已执行的结果，不重复跑工具。
                async with node.stream(agent_run.ctx) as events:
                    async for event in events:
                        if isinstance(event, FunctionToolResultEvent):
                            print_part(event.part)
        # 最终结果与原同步执行一样提供完整历史和本轮用量。
        return agent_run.result


def print_run_error(error: Exception) -> None:
    """把最终失败转换成用户能采取行动的提示，不打印完整响应体或密钥。"""
    if isinstance(error, ModelHTTPError):
        hints = {
            400: "模型请求参数不正确，请检查模型名和接口配置。",
            401: "API Key 无效，请检查 .env 中的 API_KEY。",
            402: "账号余额不足，请检查模型服务账号。",
            403: "接口访问被拒绝，请检查账号权限。",
            404: "模型或接口不存在，请检查 MODEL_NAME 和服务地址。",
            429: "请求被限流，自动重试仍失败，请稍后再试。",
        }
        message = hints.get(error.status_code, "模型服务请求失败，请稍后再试。")
        message = f"HTTP {error.status_code}：{message}"
    elif isinstance(error, UnexpectedModelBehavior):
        message = "模型响应异常或工具修正次数已耗尽，请调整需求后再试。"
    else:
        message = f"本轮执行失败（{type(error).__name__}），请检查网络或运行环境后再试。"
    console.print(message, style="red", markup=False)
    console.print("可以继续输入；本轮未保存到对话历史，已执行的文件或命令操作不会自动撤销。\n")


async def main():
    """维护一份会话状态，持续接收用户输入，直到命令或输入信号要求退出。"""
    # 状态只在内存中存活；程序退出后不会自动保存到磁盘。
    state = SessionState(model_name=MODEL_NAME)
    print_welcome_banner("Coding Agent")

    while True:
        # 读用户输入
        user_input = await read_user_input()
        if user_input is None:
            # None 表示退出；空字符串表示只按了回车，二者含义不同。
            break
        if not user_input:
            continue

        try:
            # 先清空临时日志，防止本地命令抛错时把上一轮用量再次累计。
            api_call_log.clear()
            # /new、/status 等命令在本地完成，不触发模型请求。
            action = handle_command(user_input, state)
            if action == "break":
                break
            if action == "continue":
                continue

            result = await run_agent(user_input, state)
            apply_result(state, result)
        except (KeyboardInterrupt, asyncio.CancelledError):
            # 用户主动中断不是工具故障，不自动重试，结束程序。
            console.print("\n任务已中断，退出程序。")
            break
        except Exception as error:
            # CLI 最外层安全网捕获未知错误；只中止本轮，不重新执行可能有副作用的工具。
            # 历史仍保留上一轮成功状态，避免将未配对的工具调用传给下一轮模型。
            state.last_api_calls = list(api_call_log)
            state.input_tokens += sum(call.input_tokens for call in api_call_log)
            state.output_tokens += sum(call.output_tokens for call in api_call_log)
            print_run_error(error)


# 直接执行 python main.py 时启动交互；被其他模块导入时不自动进入主循环。
if __name__ == "__main__":
    # 创建事件循环来执行 async main()，程序退出时关闭事件循环。
    asyncio.run(main())
