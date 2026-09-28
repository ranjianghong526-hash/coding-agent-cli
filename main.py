"""命令行入口：把终端输入、Agent 执行和结果展示串成完整的一轮交互。

推荐先读 main() 建立全貌，再沿着它调用的三个函数往下看。
这里的 while 循环负责多轮对话；模型与工具之间的反复调用由 Pydantic AI 管理。
"""

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
    print_agent_steps,
    print_divider,
    print_welcome_banner,
)

# PromptSession 比内置 input() 好用：支持左右移动光标编辑，还会记住本次运行的输入历史，上下方向键可以翻
prompt_session = PromptSession()


def read_user_input():
    """
    打印上横线并读一行用户输入；回车后再补一条下横线，让输入在滚动历史里保持上下边界。返回 None 表示用户希望退出（Ctrl-C / Ctrl-D）。
    """
    print_divider()
    try:
        # 去掉首尾空白；空输入会由 main() 跳过，不会发送给模型。
        user_input = prompt_session.prompt("❯ ").strip()
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
    # 注意：单独输入 / 没有命令名，现有代码会触发 IndexError。
    cmd_name = user_input[1:].split()[0]
    # 注册表把名字映射到 Command 对象，避免为每条命令写一个 if 分支。
    command = COMMANDS.get(cmd_name)
    if command is None:
        console.print(f"未知命令：/{cmd_name}，输入 /help 查看可用命令\n")
        return "continue"
    # handler 的 bool 返回值是统一约定：True 继续接收输入，False 退出。
    return "continue" if command.handler(state) else "break"


def apply_result(state, result):
    """
    跑完一轮 Agent 后，把结果同步到 SessionState 并显示新增的中间过程。
    """
    # 完整历史包含之前的对话和本轮新增消息，还包括工具请求与工具返回。
    # 下一轮传给模型时，它才能理解“继续修改刚才的文件”这类上下文。
    state.history = result.all_messages()
    # 一轮需求可能触发多次模型请求；这里累计的是整轮的 token 用量。
    usage = result.usage()
    state.input_tokens += usage.input_tokens
    state.output_tokens += usage.output_tokens
    # 复制列表，避免下一轮 api_call_log.clear() 连带清空上轮保存的列表。
    # 这是浅复制：ApiCall 对象仍共享，但当前串行流程在本轮结束后不再修改它们。
    state.last_api_calls = list(api_call_log)
    # result.new_messages() 直接拿到这一轮新增的 message，不需要手动算偏移
    print_agent_steps(result.new_messages())


def main():
    """维护一份会话状态，持续接收用户输入，直到命令或输入信号要求退出。"""
    # 状态只在内存中存活；程序退出后不会自动保存到磁盘。
    state = SessionState(model_name=MODEL_NAME)
    print_welcome_banner("Coding Agent")

    while True:
        # 读用户输入
        user_input = read_user_input()
        if user_input is None:
            # None 表示退出；空字符串表示只按了回车，二者含义不同。
            break
        if not user_input:
            continue

        # /new、/status 等命令在本地完成，不触发模型请求。
        action = handle_command(user_input, state)
        if action == "break":
            break
        if action == "continue":
            continue

        # 日志按“本轮需求”收集：清空旧日志，再让请求前后的 hooks 写入新日志。
        api_call_log.clear()
        # 同步执行会阻塞当前输入循环。框架负责模型请求、工具调用与结果回传。
        # 这里没有流式输出；本轮完成后才由 apply_result() 展示消息。
        result = agent.run_sync(user_input, message_history=state.history)
        apply_result(state, result)


# 直接执行 python main.py 时启动交互；被其他模块导入时不自动进入主循环。
if __name__ == "__main__":
    main()
