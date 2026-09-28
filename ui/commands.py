from dataclasses import dataclass, field
from typing import Callable, Optional

from rich.markdown import Heading, Markdown
from rich.markup import escape
from rich.padding import Padding
from rich.rule import Rule

from .render import console, print_step, print_welcome_banner


class LeftAlignedHeading(Heading):
    """
    rich 默认把 Markdown 标题渲染成居中对齐，宽终端里看着像错位，覆盖成左对齐。
    """
    def __rich_console__(self, console, options):
        text = self.text
        text.justify = "left"
        yield text


# 全局替换 Markdown 的标题渲染元素
Markdown.elements["heading_open"] = LeftAlignedHeading


@dataclass
class SessionState:
    """
    跨命令共享的会话状态，主循环把它传给每个命令处理函数。
    """
    history: list = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    model_name: str = ""
    # 最近一轮 user input 触发的所有 model API 调用记录
    last_api_calls: list = field(default_factory=list)


@dataclass
class Command:
    name: str
    description: str
    # handler 返回 False 表示主循环应当退出
    handler: Callable[["SessionState"], bool]


def print_divider() -> None:
    """
    每轮交互之前打印一条分割线，区分输入区域。Rule 会自适应终端宽度。
    """
    console.print(Rule(style="grey50"))


def _truncate(text, limit: int = 120) -> str:
    """
    截断并 escape，用于 tool 参数 / 返回值 / 用户输入这类可能过长的内容。
    """
    text = str(text).strip()
    text = text if len(text) <= limit else text[:limit] + "..."
    return escape(text)


def _full(text) -> str:
    """
    完整显示，只做 escape 不截断，用于 thinking 和 assistant text 这种用户关心的内容。
    """
    return escape(str(text).strip())


def _format_part_line(part) -> Optional[str]:
    """
    把一条消息里的单个 part 格式化为带 Rich markup 的字符串。
    版式：图标 + role 标签独占一行，内容换行到下一行，不用「|」分隔。
    """
    # 内容行统一缩进 2 格，和图标（占 2 格：图标 + 空格）后的 role 名对齐
    kind = part.part_kind
    if kind == "user-prompt":
        return f"[cyan]❯ user[/]\n  {_truncate(part.content)}"
    if kind == "thinking":
        # thinking 整块 dim，弱化视觉权重；不截断，完整保留思考过程
        return f"[dim]✻ thinking[/]\n  [dim]{_full(part.content)}[/]"
    if kind == "text":
        content = (part.content or "").strip()
        if not content:
            return None
        # assistant 是用户最关心的最终回答，完整显示
        return f"[green]● assistant[/]\n  {_full(content)}"
    if kind == "tool-call":
        # 命令、路径动辄上百字符，参数放宽到 500 字符再截断
        return f"[yellow]⏺ tool_call[/]\n  [yellow dim]{part.tool_name}({_truncate(part.args, 500)})[/]"
    if kind == "tool-return":
        return f"[magenta]✔ tool_return[/]\n  [magenta dim]{part.tool_name} -> {_truncate(part.content)}[/]"
    if kind == "retry-prompt":
        # 工具抛 ModelRetry 后，SDK 生成 retry-prompt 把错误反馈给模型
        return f"[yellow]✘ tool_retry[/]\n  [yellow dim]{part.tool_name} -> {_truncate(part.content)}[/]"
    return None


def print_assistant_markdown(content: str) -> None:
    """
    模型的回复天然是 Markdown 格式，整块渲染出来，而不是打印原始文本。
    """
    console.print("[green]● assistant[/]")
    # Markdown 是块级渲染对象，没法跟在行内前缀后面，所以另起一行渲染；左缩进 2 格和 role 名对齐
    console.print(Padding(Markdown(content), (0, 0, 0, 2)))


def print_part(part) -> None:
    """
    渲染单个消息 part：assistant 文本走 Markdown 块渲染，其余 part 是单行文本。
    """
    if part.part_kind == "text":
        content = (part.content or "").strip()
        if content:
            print_assistant_markdown(content)
            # 每个 role block 末尾留一个空行，块与块之间不那么挤
            console.print()
        return
    line = _format_part_line(part)
    if line:
        label, _, body = line.partition("\n")
        # body 形如 "  [markup]…"，去掉字面前导 2 空格，交给 print_step 用 Padding 缩进（折行续行也保持缩进）
        print_step(label, body[2:])


def print_agent_steps(new_messages) -> None:
    """
    主循环里调用：显示这一轮 Agent 新增的中间过程（thinking、文本、工具调用、工具返回）。
    """
    for msg in new_messages:
        for part in msg.parts:
            # 主循环里不重复显示用户刚刚输入的内容
            if part.part_kind == "user-prompt":
                continue
            print_part(part)


def cmd_exit(state: SessionState) -> bool:
    console.print("再见 👋")
    return False


def cmd_help(state: SessionState) -> bool:
    console.print("可用命令：")
    for cmd in COMMANDS.values():
        console.print(f"  /{cmd.name:<10} {cmd.description}")
    console.print()
    return True


def cmd_new(state: SessionState) -> bool:
    """
    开启新会话：清空历史、token 计数、API 调用记录。
    """
    state.history.clear()
    state.input_tokens = 0
    state.output_tokens = 0
    state.last_api_calls.clear()
    console.print("已开启新会话\n")
    return True


def cmd_status(state: SessionState) -> bool:
    console.print(f"模型：           {state.model_name}")
    console.print(f"历史消息条数：    {len(state.history)}")
    console.print(f"累计输入 tokens：{state.input_tokens}")
    console.print(f"累计输出 tokens：{state.output_tokens}\n")
    return True


def cmd_api_detail(state: SessionState) -> bool:
    """
    显示最近一轮 user input 触发的所有 model API 调用元数据。
    """
    if not state.last_api_calls:
        console.print("(还没有任何模型调用记录，先发一条消息再来看)\n")
        return True

    console.print(f"最近一轮共发起 {len(state.last_api_calls)} 次 model API 调用\n")

    for i, call in enumerate(state.last_api_calls, 1):
        console.print(f"[bold]Call #{i}[/]")
        console.print(f"  Request:")
        console.print(f"    model:        {call.model}")
        console.print(f"    messages:     {call.messages_count} 条")
        if call.last_part is not None:
            preview = _format_part_line(call.last_part)
            if preview:
                console.print(f"    last_message: {preview}")
        console.print(f"    tools:        {', '.join(call.tools)}")
        console.print(f"  Response:")
        console.print(f"    finish_reason: {call.finish_reason}")
        console.print(f"    parts:         {', '.join(call.parts_kinds)}")
        console.print(f"    usage:         input={call.input_tokens}, output={call.output_tokens}")
        console.print()
    return True


COMMANDS = {
    "new": Command("new", "开启新会话", cmd_new),
    "status": Command("status", "显示当前会话状态", cmd_status),
    "api-detail": Command("api-detail", "显示最近一轮 model API 调用详情", cmd_api_detail),
    "help": Command("help", "显示可用命令", cmd_help),
    "exit": Command("exit", "退出程序", cmd_exit),
}
