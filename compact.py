"""
上下文压缩：用一次 LLM 调用把整段对话总结成结构化摘要，再用摘要加上最近读过的文件，重建一份小得多的新历史。
手动 /compact 命令和自动压缩共用这套流程，区别只在由谁触发。
"""
import os
import re

from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, UserPromptPart
from rich.markdown import Markdown
from rich.padding import Padding

import session
from agent.core import model
from agent.file_state import ReadFileState
from mentions import build_mention_messages
from ui.render import console

# 模型的上下文窗口大小
CONTEXT_WINDOW = 131_072
# 给压缩调用预留的摘要输出空间
COMPACT_OUTPUT_RESERVE = 20_000
# 水位线之前再留的安全余量
AUTO_COMPACT_BUFFER = 10_000
# 自动压缩连续失败这么多次后，本会话不再尝试
MAX_COMPACT_FAILURES = 3
# 压缩后恢复最近读过的文件：最多几个、总字符预算多少
RESTORE_MAX_FILES = 5
RESTORE_MAX_CHARS = 30_000


COMPACT_PROMPT = (
    "重要：只输出纯文本，不要调用任何工具。\n\n"
    "- 不要使用 read_file、edit_file、write_file、run_command 或其他任何工具。\n"
    "- 上面的对话里已经有你需要的全部信息。\n"
    "- 工具调用会被拒绝，还会浪费你唯一的回合——你将无法完成任务。\n"
    "- 你的整个回复必须是纯文本：先一个 <analysis> 块，再一个 <summary> 块。\n\n"
    "你的任务是为到目前为止的对话写一份详细摘要，重点关注用户的明确请求和你已经做过的事情。摘要要保留足够的技术细节，让后续工作能基于它无缝继续。\n"
    "这条压缩指令本身不属于要总结的对话——任何段落都不要提到或引用它，不要把它算作用户消息，也不要把写摘要说成当前工作。\n\n"
    "写正式摘要之前，先在 <analysis> 标签里打草稿整理思路：按顺序过一遍对话，核对用户的请求、你的处理方式、关键决策、文件名和代码片段、报错和修复过程。\n\n"
    "然后在 <summary> 标签里输出正式摘要，包含以下几段：\n\n"
    "1. 主要请求和意图：详细记录用户的所有明确请求和意图\n"
    "2. 关键技术概念：列出涉及的重要技术概念、框架和方案\n"
    "3. 文件和代码：列出查看过、修改过、新建过的具体文件和代码段，附上关键代码片段\n"
    "4. 错误与修复：列出遇到过的报错和修复方法，特别注意用户要求你换种做法的反馈\n"
    "5. 全部用户消息：一字不改地列出所有用户消息（工具结果和这条压缩指令都不算），不要转述——它们是理解用户反馈和意图变化的关键\n"
    "6. 当前工作与下一步：精确描述写摘要前正在做的事情和紧接着的下一步，逐字引用最近几条对话原文，确保接续时不跑偏\n\n"
    "再次提醒：不要调用任何工具。只输出纯文本——先 <analysis> 块，后 <summary> 块。"
)


SUMMARY_WRAPPER = (
    "本会话由一段因上下文写满而被压缩的对话延续而来，以下是之前对话的摘要：\n\n"
    "{summary}\n\n"
    "如果摘要里缺少你需要的细节（具体代码片段、报错原文等），可以用 read_file 读取压缩前的完整对话记录：{transcript}\n"
    "请基于这份摘要继续工作。不要向用户复述摘要内容，直接从对话中断的地方接着干。"
)


# 压缩专用 agent：不注册任何工具，模型想调也调不了
summarizer = Agent(model)


def compact_threshold() -> int:
    """
    触发自动压缩的水位线：从窗口上限往回退掉摘要输出预留，再退一段安全余量。
    """
    return CONTEXT_WINDOW - COMPACT_OUTPUT_RESERVE - AUTO_COMPACT_BUFFER


def context_tokens(history) -> int:
    """
    用最近一条模型回复的 usage 估算当前上下文占用：input 覆盖当轮发出去的全部内容，再加上它生成的部分。
    """
    for msg in reversed(history):
        if msg.kind == "response" and msg.usage.input_tokens:
            return msg.usage.input_tokens + msg.usage.output_tokens
    return 0


def extract_summary(text: str) -> str:
    """
    先整块剥掉 <analysis> 草稿，再取 <summary> 块的内容。
    先剥再取是有讲究的：草稿里可能字面提到 <summary> 标签，会带偏匹配。
    """
    text = re.sub(r"<analysis>.*?</analysis>", "", text, flags=re.DOTALL)
    match = re.search(r"<summary>(.*?)</summary>", text, re.DOTALL)
    text = match.group(1) if match else text
    # 正文里残留的标签字样一并清掉
    return re.sub(r"</?(analysis|summary)>", "", text).strip()


def build_summary_message(summary: str, transcript_path) -> ModelRequest:
    """
    把摘要包装成一条用户消息，作为新历史的唯一起点。
    """
    content = SUMMARY_WRAPPER.format(summary=summary, transcript=transcript_path)
    return ModelRequest(parts=[UserPromptPart(content=content)])


def restore_file_messages(old_state: ReadFileState, new_state: ReadFileState) -> list:
    """
    重新注入最近读过的文件：文件原文没法从摘要还原，由程序自己保留，并且带预算上限。
    """
    picked, used = [], 0
    for path in reversed(old_state.paths()):
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        if len(picked) >= RESTORE_MAX_FILES or used + size > RESTORE_MAX_CHARS:
            break
        picked.append(path)
        used += size
    # 恢复顺序保持原来的读取顺序
    picked.reverse()
    # 和 @ 引用同一套机制：伪装成 read_file 调用塞进历史，顺带登记进新 readFileState
    return build_mention_messages(picked, new_state)


async def run_compact(state, custom_instructions: str = "") -> None:
    """
    完整的压缩流程：一次 LLM 调用总结全部历史，然后用摘要加恢复的文件整体替换成新历史。
    """
    if not state.history:
        console.print("(当前会话还没有对话内容，无需压缩)\n")
        return

    prompt = COMPACT_PROMPT
    extra = custom_instructions.strip()
    if extra:
        prompt += "\n\n补充要求：\n" + extra

    console.print("正在压缩上下文，可能需要一会儿...\n")
    result = await summarizer.run(prompt, message_history=state.history)
    summary = extract_summary(result.output)

    # 压缩调用本身的开销也计入会话累计
    usage = result.usage()
    state.input_tokens += usage.input_tokens
    state.output_tokens += usage.output_tokens

    # 重写会话文件之前先存档完整对话记录，摘要还原不了的细节模型可以回头去读
    transcript_path = session.archive_session(state.session_id)

    # 换新 readFileState：旧登记随旧历史一起作废，恢复的文件会重新登记进来
    old_state = state.read_file_state
    state.read_file_state = ReadFileState()
    restored = restore_file_messages(old_state, state.read_file_state)
    state.history = [build_summary_message(summary, transcript_path)] + restored
    # 新历史整体重写会话文件
    session.rewrite_messages(state.session_id, state.history)

    # 旧检查点的 history_index 指向已被压缩掉的消息，整体丢弃
    fh = state.file_history
    if fh is not None and fh.checkpoints:
        fh.drop_from(fh.checkpoints[0])
    # 已召回的记忆随旧历史消失，允许重新召回；进程内 API 调用记录一并清空
    state.surfaced_memories = set()
    state.last_api_calls.clear()

    # 压缩调用的 input_tokens 就是压缩前的上下文大小
    console.print(f"[magenta]✻ 压缩完成：压缩前上下文 {usage.input_tokens:,} tokens，已重建为一份摘要 + {len(restored) // 2} 个最近读过的文件[/]")
    console.print(f"[dim]完整对话记录已存档：{transcript_path}[/]")
    console.print(Padding(Markdown(summary), (0, 0, 0, 2)), style="dim")
    console.print()


async def auto_compact_if_needed(state) -> None:
    """
    每次发请求前调用：上下文越过水位线就自动压缩；失败只提示不阻断，连续失败多次后不再重试。
    """
    if state.compact_failures >= MAX_COMPACT_FAILURES:
        return
    used = context_tokens(state.history)
    threshold = compact_threshold()
    if used < threshold:
        return
    console.print(f"[yellow]上下文接近上限（{used:,} / {threshold:,} tokens），自动压缩中[/]\n")
    try:
        await run_compact(state)
        state.compact_failures = 0
    except Exception as e:
        state.compact_failures += 1
        console.print(f"[yellow]自动压缩失败（{type(e).__name__}: {e}），继续对话[/]\n")
        if state.compact_failures >= MAX_COMPACT_FAILURES:
            console.print("[yellow]自动压缩已连续失败多次，本会话不再尝试[/]\n")
