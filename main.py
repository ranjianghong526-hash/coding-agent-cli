import asyncio

from pydantic_ai import Agent
from pydantic_graph import End
from pydantic_ai.messages import ModelRequest, UserPromptPart

import compact
import mcp_servers
from legacy_migration import migrate_legacy_data
import session
from agent import agent, MODEL_NAME, api_call_log
from agent.deps import AgentDeps
from agent.reminders import build_job_reminder_text
from background_jobs import JobRegistry
from ui.render import print_job_finished
from file_history import FileHistory
from memory import background, recall, store
from tasks_store import TasksStore
from ui.commands import (
    COMMANDS,
    SessionState,
    console,
    print_part,
    print_welcome_banner,
)
from ui.input_ui import Repl
from mentions import build_mention_messages, extract_at_mentions


async def handle_command(user_input, state):
    """
    处理以 / 开头的命令。
    返回 'pass'：不是命令，交给 Agent；
    返回 'continue'：命令已处理，进入下一轮；
    返回 'break'：命令要求退出程序。
    """
    if not user_input.startswith("/"):
        return "pass"
    cmd_name, _, args = user_input[1:].partition(" ")
    command = COMMANDS.get(cmd_name)
    if command is None:
        console.print(f"未知命令：/{cmd_name}，输入 /help 查看可用命令\n")
        return "continue"
    # 声明接收参数的命令（如 /compact），把命令名后面的整段文本传给它
    if command.takes_args:
        result = command.handler(state, args.strip())
    else:
        result = command.handler(state)
    # 个别命令（如 /resume）要弹交互式列表，是异步的，需要 await
    if asyncio.iscoroutine(result):
        result = await result
    return "continue" if result else "break"


def apply_result(state, result):
    """
    跑完一轮 Agent 后，把结果同步到 SessionState。
    """
    state.history = result.all_messages()
    # 当前 pydantic-ai 2.x 的 usage 是属性。
    usage = result.usage
    state.input_tokens += usage.input_tokens
    state.output_tokens += usage.output_tokens
    state.last_api_calls = list(api_call_log)
    # 把本轮新增的消息追加到会话文件
    session.append_messages(state.session_id, result.new_messages())


def inject_at_mentions(user_input, state):
    """
    解析用户输入里的 @path，把每个被引用的文件伪装成一次 read_file 调用，在 Agent 跑起来之前塞进对话历史。
    """
    paths = extract_at_mentions(user_input)
    if not paths:
        return
    mention_messages = build_mention_messages(paths, state.read_file_state)
    if not mention_messages:
        return
    # 塞进历史：模型下一轮就能看到这些「读文件」记录，以为是自己读的
    state.history += mention_messages
    # 持久化，/resume 恢复会话时能连同引用进来的文件内容一起还原
    session.append_messages(state.session_id, mention_messages)
    # 终端里也回显一下注入了哪些文件，让用户看到 @ 引用确实生效了
    for msg in mention_messages:
        for part in msg.parts:
            print_part(part)


async def run_agent_loop(user_input, state):
    """
    展开 agent.run_sync()，逐节点驱动 Agent 循环，每步实时打印。
    """
    api_call_log.clear()

    # deps 把本会话的 readFileState、tasksStore 和文件检查点一起打包成 AgentDeps 注入工具层，工具内通过 ctx.deps.* 访问
    deps = AgentDeps(
        read_file_state=state.read_file_state,
        tasks_store=state.tasks_store,
        file_history=state.file_history,
        job_registry=state.job_registry,
    )
    async with agent.iter(
        user_input, message_history=state.history, deps=deps,
        toolsets=mcp_servers.active_toolsets(),
    ) as run:
        node = run.next_node

        while not isinstance(node, End):
            node = await run.next(node)

            if Agent.is_call_tools_node(node):
                for part in node.model_response.parts:
                    print_part(part)

            elif Agent.is_model_request_node(node):
                for part in node.request.parts:
                    if part.part_kind in ("tool-return", "retry-prompt"):
                        print_part(part)

    apply_result(state, run.result)
    # 把本轮结果交回给调用方，后台记忆提炼要用本轮新增的消息判断该不该跑
    return run.result


async def watch_jobs(state, repl, interval=1):
    """仅扫描本地状态，不请求模型；空闲且 job 完成时才提交通知。"""
    while True:
        await asyncio.sleep(interval)
        if repl.is_idle:
            text = build_job_reminder_text(state.job_registry)
            if text:
                # 检查、领取、提交之间没有 await，避免与用户回车抢占。
                repl.submit_system(text)


async def main():
    migrated = migrate_legacy_data()
    if migrated["sessions"] or migrated["memories"]:
        console.print(f"已导入旧数据：{migrated['sessions']} 个会话、{migrated['memories']} 条记忆。")
    for item in migrated["errors"]:
        console.print(f"旧数据导入失败：{item}，原文件已保留。", markup=False)
    session_id = session.new_session_id()
    # 启动时建好记忆目录，system prompt 里承诺过「目录已存在」，模型就不必浪费回合去确认
    store.ensure_memory_dir()
    # 本会话的 TasksStore 落盘在 ~/.my-claude-code/tasks/<session_id>/，会话级隔离
    tasks_store = TasksStore(session_id=session_id)
    state = SessionState(
        model_name=MODEL_NAME,
        session_id=session_id,
        tasks_store=tasks_store,
        file_history=FileHistory(session_id=session_id),
        job_registry=JobRegistry(session_id, print_job_finished),
    )
    print_welcome_banner("my-claude-code")

    watcher = None
    try:
        # 转圈提示连接进度，否则冷启动拉包时用户会以为卡死了
        with console.status("正在连接 MCP server..."):
            mcp_summary = await mcp_servers.startup()
        if mcp_summary:
            console.print(mcp_summary + "\n")

        # 常驻输入区：输入框整个会话期间不消失。task 面板通过 state.tasks_store 拉数据，所以 /new、/resume 换会话时不需要重新接线
        repl = Repl(state)

        async def on_submit(user_input, is_system=False):
            if is_system:
                await compact.auto_compact_if_needed(state)
                notification = ModelRequest(parts=[UserPromptPart(user_input)],
                                            metadata={"origin": "background-job-notification"})
                state.history.append(notification)
                session.append_messages(state.session_id, [notification])
                repl.start_working()
                # None 表示从已有通知历史继续，不再添加一条普通用户输入。
                await run_agent_loop(None, state)
                return
            # 每次回车提交一行输入，都走这里
            # 先处理 / 开头的命令
            action = await handle_command(user_input, state)
            if action == "break":
                # 命令要求退出，结束常驻输入区
                repl.exit()
                return
            if action == "continue":
                return

            # 发请求前检查上下文水位，越过阈值就先自动压缩再继续
            await compact.auto_compact_if_needed(state)

            # 建检查点：此刻 @ 引用和记忆召回还没注入，len(history) 就是干净的回退下标
            state.file_history.make_checkpoint(len(state.history), user_input)

            # 解析 @ 引用，把被引用的文件伪装成 read_file 调用塞进历史
            inject_at_mentions(user_input, state)

            # 核心 Agent 循环：开请求时显示 working...，结束 / 被打断后由 Repl 统一隐藏；中途按 ESC / Ctrl+C 会打断
            repl.start_working()
            # 召回相关记忆塞进历史，模型带着过去积累的经验处理本轮输入
            await recall.inject_memories(user_input, state)
            result = await run_agent_loop(user_input, state)
            # 本轮结束后由后台提炼记忆，并顺带检查是否到了该定期合并的时候
            background.schedule(state, result.new_messages())

        watcher = asyncio.create_task(watch_jobs(state, repl))
        await repl.run(on_submit)
    finally:
        # 后台任务和 MCP 连接在正常退出、启动失败、取消时都能收尾。
        try:
            if watcher:
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)
            try:
                await state.job_registry.aclose()
            finally:
                await background.drain()
        finally:
            await mcp_servers.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, EOFError):
        pass
