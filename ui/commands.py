import difflib
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

import questionary
from prompt_toolkit.application import Application, in_terminal
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from rich.markdown import Heading, Markdown
from rich.markup import escape
from rich.padding import Padding

import compact
import mcp_servers
import permissions
import session
from agent import ReadFileState
from file_history import FileHistory
from memory import background, store
from tasks_store import TasksStore
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
    # 当前会话 ID，决定对话历史写入哪个 jsonl 文件
    session_id: str = ""
    # 最近一轮 user input 触发的所有 model API 调用记录
    last_api_calls: list = field(default_factory=list)
    # 本会话的 readFileState，注入给文件工具；切换会话时换上新实例
    read_file_state: ReadFileState = field(default_factory=ReadFileState)
    # 本会话的 TasksStore，注入给 task 工具；落盘在 ~/.my-claude-code/tasks/<session_id>/，所以必须由 main 显式构造传入
    tasks_store: TasksStore | None = None
    # 本会话已召回注入过的记忆文件名，同一条记忆不重复注入
    surfaced_memories: set = field(default_factory=set)
    # 本会话的文件检查点，/rewind 靠它回退文件和对话；由 main 构造传入
    file_history: FileHistory | None = None
    # /rewind 回退对话后待回填输入框的原 prompt，Repl 在命令结束后消费
    pending_input: str = ""
    # 自动压缩连续失败的次数，达到上限后不再重试
    compact_failures: int = 0


@dataclass
class Command:
    name: str
    description: str
    # handler 返回 False 表示主循环应当退出
    handler: Callable[..., bool]
    # 是否接收命令名后面的参数串（如 /compact 的补充指令）
    takes_args: bool = False


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
        # 工具成功返回，标签用 ✔
        return f"[magenta]✔ tool_return[/]\n  [magenta dim]{part.tool_name} -> {_truncate(part.content)}[/]"
    if kind == "retry-prompt":
        # 工具抛 ModelRetry 后，SDK 生成 retry-prompt 把错误反馈给模型，标签用 ✘ 表示这次调用失败
        return f"[yellow]✘ tool_retry[/]\n  [yellow dim]{part.tool_name} -> {_truncate(part.content)}[/]"
    return None


def print_assistant_markdown(content: str) -> None:
    """
    模型的回复天然是 Markdown 格式，整块渲染出来，而不是打印原始文本。
    """
    console.print("[green]● assistant[/]")
    # Markdown 是块级渲染对象，没法跟在行内前缀后面，所以另起一行渲染；左缩进 2 格和 role 名对齐
    console.print(Padding(Markdown(content), (0, 0, 0, 2)))


# read / write 内容预览最多显示的行数，超出的折叠成「… +N 行」
_PREVIEW_MAX_LINES = 8


def _preview_numbered(content: str) -> Optional[str]:
    """
    把内容加绿色行号、截断成预览，用于 write_file 的写入内容展示。
    """
    lines = content.splitlines()
    if not lines:
        return None
    width = len(str(len(lines)))
    shown = lines[:_PREVIEW_MAX_LINES]
    out = [f"  [green]{i:>{width}}[/] {escape(ln)}" for i, ln in enumerate(shown, 1)]
    if len(lines) > _PREVIEW_MAX_LINES:
        out.append(f"  [dim]… +{len(lines) - _PREVIEW_MAX_LINES} 行[/]")
    return "\n".join(out)


def _preview_diff(old: str, new: str) -> str:
    """
    用 difflib 把 old -> new 渲染成红绿 diff：删除行红、新增行绿、不变行 dim。
    """
    out = []
    for line in difflib.ndiff(old.splitlines(), new.splitlines()):
        tag, body = line[:2], escape(line[2:])
        if tag == "- ":
            out.append(f"  [red]- {body}[/]")
        elif tag == "+ ":
            out.append(f"  [green]+ {body}[/]")
        elif tag == "  ":
            out.append(f"  [dim]  {body}[/]")
        # "? " 是 difflib 的字符级提示行，跳过不显示
    return "\n".join(out)


def _print_file_op(part) -> bool:
    """
    文件工具的富展示：edit 红绿 diff、write 带行号写入预览、read 带行号读取结果。返回 True 表示已处理。
    """
    # 这里按工具名特判，没像 permissions 那样建注册表：展示逻辑依赖 rich，若让工具层注册渲染器，agent 层就会反向依赖 UI 层，得不偿失
    kind = part.part_kind
    if kind == "tool-call" and part.tool_name in ("edit_file", "write_file"):
        args = part.args_as_dict()
        path = escape(str(args.get("path", "")))
        # edit / write 的标签行与「工具名(路径)」内容行格式一致，统一打印，只有下方的预览不同
        console.print("[yellow]⏺ tool_call[/]")
        console.print(Padding(f"[yellow dim]{part.tool_name}({path})[/]", (0, 0, 0, 2)))
        if part.tool_name == "edit_file":
            console.print(_preview_diff(args.get("old_string", ""), args.get("new_string", "")))
        else:
            block = _preview_numbered(args.get("content", ""))
            if block:
                console.print(block)
        return True
    if kind == "tool-return" and part.tool_name == "read_file":
        # read_file 的返回已经是带行号的内容，按行截断预览
        console.print("[magenta]✔ tool_return[/]")
        console.print(Padding("[magenta dim]read_file[/]", (0, 0, 0, 2)))
        lines = str(part.content).splitlines()
        for ln in lines[:_PREVIEW_MAX_LINES]:
            console.print(f"  [dim]{escape(ln)}[/]")
        if len(lines) > _PREVIEW_MAX_LINES:
            console.print(f"  [dim]… +{len(lines) - _PREVIEW_MAX_LINES} 行[/]")
        return True
    return False


def print_part(part) -> None:
    """
    渲染单个消息 part：assistant 文本走 Markdown，文件操作走富展示，其余 part 是单行文本。
    """
    if part.part_kind == "text":
        content = (part.content or "").strip()
        if content:
            print_assistant_markdown(content)
            # 每个 role block 末尾留一个空行，块与块之间不那么挤
            console.print()
        return
    # 文件工具（read / edit / write）走专门的富展示
    if _print_file_op(part):
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
    开启新会话：清空历史、token 计数、API 调用记录，换一个新的会话 ID。
    """
    state.history.clear()
    state.input_tokens = 0
    state.output_tokens = 0
    state.last_api_calls.clear()
    state.compact_failures = 0
    state.session_id = session.new_session_id()
    # 权限白名单是会话级的，「本会话不再询问」不该带进新会话
    permissions.state.session_allowed.clear()
    # readFileState 也是会话级的，新会话从空白开始，旧会话读过的文件不带进来
    state.read_file_state = ReadFileState()
    # TasksStore 按会话隔离落盘（~/.my-claude-code/tasks/<session_id>/），换会话同时换一份新的，task 面板会在下一次重绘时反映新 store
    state.tasks_store = TasksStore(session_id=state.session_id)
    # 召回去重集合是会话级的，新会话里旧记忆可以重新被召回
    state.surfaced_memories = set()
    # 文件检查点也按会话隔离，新会话从零开始记
    state.file_history = FileHistory(session_id=state.session_id)
    console.print("已开启新会话\n")
    return True


def _one_line(text, limit: int = 50) -> str:
    """
    压掉空白折成一行并截断；给 prompt_toolkit 用的纯文本，不做 Rich escape。
    """
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "..."


def _summary_line(mtime, prompt: str) -> str:
    """
    拼一条会话列表的展示文本：修改时间 + 首条用户输入摘要。
    """
    return f"{mtime:%m-%d %H:%M}  {_one_line(prompt)}"


def _recount_tokens(state: SessionState) -> None:
    """
    按对话历史重算会话的 token 用量，每条模型回复都带 usage。
    历史被整体替换的地方调用（/resume 恢复历史、/rewind 截断历史）。
    """
    state.input_tokens = sum(
        m.usage.input_tokens for m in state.history if m.kind == "response"
    )
    state.output_tokens = sum(
        m.usage.output_tokens for m in state.history if m.kind == "response"
    )


async def cmd_resume(state: SessionState) -> bool:
    """
    列出当前项目的历史会话，选中后恢复对话历史。
    它跑在 REPL 的事件循环里，所以是异步的：in_terminal 把终端让给 questionary，结束后再恢复输入框。
    """
    sessions = session.list_sessions()
    if not sessions:
        console.print("(当前项目还没有历史会话)\n")
        return True

    choices = [
        questionary.Choice(title=_summary_line(mtime, prompt), value=sid)
        for sid, mtime, prompt in sessions
    ]
    async with in_terminal():
        selected = await questionary.select(
            "选择要恢复的会话（上下键移动，回车确认）：", choices=choices
        ).ask_async()
    # 用户按 Ctrl+C 取消选择
    if selected is None:
        return True

    # 还原对话历史，并把会话 ID 切换成选中的旧会话，后续消息继续追加到同一个文件
    state.history = session.load_history(selected)
    state.session_id = selected
    # 权限白名单是会话级的，切换会话后清空
    permissions.state.session_allowed.clear()
    # readFileState 也是会话级的，切换会话后换上新实例。恢复的历史里虽有读取痕迹，但进程退出后文件可能已变、mtime 不再可信，让模型恢复后首次编辑重读一次更稳
    state.read_file_state = ReadFileState()
    # TasksStore 指向当前 session_id 的目录，恢复时直接接上旧 task 列表（每个 task 是独立 JSON，已经落盘）
    state.tasks_store = TasksStore(session_id=state.session_id)
    # 召回去重集合从空集重新开始：恢复的历史里已注入的记忆可能被再召回一次，重复一次无伤大雅
    state.surfaced_memories = set()
    # 文件检查点随会话恢复：构造函数会加载该会话落盘的 checkpoints.json，旧检查点直接可用
    state.file_history = FileHistory(session_id=state.session_id)

    # 把恢复的会话的 token 用量累加回来
    _recount_tokens(state)
    # 最近一轮的 API 调用记录只在进程内有效，没法恢复，清空
    state.last_api_calls.clear()
    state.compact_failures = 0

    # 把恢复的对话回放到屏幕上
    console.print(f"\n已恢复会话 {selected[:8]}，共 {len(state.history)} 条消息：\n")
    for msg in state.history:
        for part in msg.parts:
            # 回放和实时输出共用同一套 part 渲染逻辑
            print_part(part)
    console.print()
    return True


# rewind picker 的配色：当前项蓝色高亮，描述与 footer 弱化，增删行数绿/红
_PICKER_STYLE = Style.from_dict({
    "question": "bold",
    "label-current": "#3b82f6 bold",
    "label": "",
    "desc": "#6b7280",
    "plus": "#10b981",
    "minus": "#ef4444",
    "footer": "#6b7280",
})


class _ListPicker:
    """
    手绘单选 picker，每个选项可以带一行弱化的描述，视觉对齐权限审批的 picker。
    """

    def __init__(self, question: str, options: list, header: list | None = None):
        # options 是 (value, label, desc) 列表；desc 是 FormattedText 片段列表，None 表示没有描述行
        self.question = question
        self.options = options
        # header 是渲染在问句和选项之间的 FormattedText 片段
        self.header = header
        self.cursor = 0
        # 选中项的 value；Esc / Ctrl+C 取消时保持 None
        self.result = None
        self.app = self._build_app()

    def _render_question(self):
        return FormattedText([("class:question", self.question)])

    def _render_header(self):
        return FormattedText(self.header)

    def _render_options(self):
        lines: list[tuple[str, str]] = []
        for i, (_value, label, desc) in enumerate(self.options):
            is_cursor = (i == self.cursor)
            pointer = "❯" if is_cursor else " "
            cls_label = "class:label-current" if is_cursor else "class:label"
            lines.append((cls_label, f" {pointer}  {i + 1}. {label}"))
            lines.append(("", "\n"))
            if desc:
                # 描述行缩进到与 label 文字对齐
                lines.append(("", "       "))
                lines.extend(desc)
                lines.append(("", "\n"))
        return FormattedText(lines)

    def _render_footer(self):
        return FormattedText([("class:footer", "  ↑↓ 选择 · Enter 确认 · Esc 取消")])

    def _move(self, delta: int):
        self.cursor = (self.cursor + delta) % len(self.options)

    def _build_app(self) -> Application:
        kb = KeyBindings()

        @kb.add("up")
        @kb.add("k")
        def _(event):
            self._move(-1)

        @kb.add("down")
        @kb.add("j")
        def _(event):
            self._move(1)

        @kb.add("enter")
        def _(event):
            self.result = self.options[self.cursor][0]
            self.app.exit()

        @kb.add("escape")
        @kb.add("c-c")
        def _(event):
            self.app.exit()

        windows = [
            Window(FormattedTextControl(self._render_question), height=1, always_hide_cursor=True),
            Window(FormattedTextControl(self._render_options), dont_extend_height=True, always_hide_cursor=True),
            Window(FormattedTextControl(self._render_footer), height=1, always_hide_cursor=True),
        ]
        if self.header:
            windows.insert(1, Window(
                FormattedTextControl(self._render_header),
                dont_extend_height=True, always_hide_cursor=True,
            ))
        layout = Layout(HSplit(windows))
        return Application(
            layout=layout,
            key_bindings=kb,
            style=_PICKER_STYLE,
            full_screen=False,
            mouse_support=False,
            # 选完擦掉整个 picker，滚动区里不留下选项
            erase_when_done=True,
        )

    async def run(self):
        await self.app.run_async()
        return self.result


def _age_text(timestamp: float) -> str:
    """
    把时间戳转成「14 分钟前」这样的相对时间。
    """
    seconds = max(0, time.time() - timestamp)
    if seconds < 60:
        return "刚刚"
    if seconds < 3600:
        return f"{int(seconds // 60)} 分钟前"
    if seconds < 86400:
        return f"{int(seconds // 3600)} 小时前"
    return f"{int(seconds // 86400)} 天前"


def _changes_headline(changes) -> tuple:
    """
    把 FileChange 清单压成 (名称, 增行, 删行)：单个文件显示文件名，多个文件显示个数。
    """
    plus = sum(c.insertions for c in changes)
    minus = sum(c.deletions for c in changes)
    if len(changes) == 1:
        name = os.path.basename(changes[0].path)
    else:
        name = f"{len(changes)} 个文件"
    return name, plus, minus


def _changes_fragments(changes) -> list:
    """
    把 FileChange 清单渲染成 picker 的描述片段：「main.py +12 -3」或「无代码改动」。
    """
    if not changes:
        return [("class:desc", "无代码改动")]
    name, plus, minus = _changes_headline(changes)
    return [
        ("class:desc", f"{name} "),
        ("class:plus", f"+{plus}"),
        ("class:minus", f" -{minus}"),
    ]


async def cmd_rewind(state: SessionState) -> bool:
    """
    回退到过去的某个检查点：先选检查点，再选回退对话、回退代码，还是两者都回退。
    """
    fh = state.file_history
    if fh is None or not fh.checkpoints:
        console.print("(暂无可回退的检查点，每条消息发出时会自动创建检查点)\n")
        return True

    # 检查点从新到旧排列，每条下面标注那一轮对话产生的改动量
    cps = list(reversed(fh.checkpoints))
    options = [
        (i, _one_line(s.prompt), _changes_fragments(fh.turn_stats(s)))
        for i, s in enumerate(cps)
    ]
    async with in_terminal():
        picked = await _ListPicker("回退到哪条消息之前：", options).run()
    if picked is None:
        return True
    cp = cps[picked]
    # 确认信息展示累计代价：从当前状态回到检查点要撤销的全部改动
    plan = fh.diff_stats(cp)

    # 确认信息作为第二级菜单的 header 由 picker 一起渲染
    header = [
        ("", "\n"),
        ("class:desc", f"  │ {_one_line(cp.prompt, 100)}\n"),
        ("class:desc", f"  │ （{_age_text(cp.timestamp)}）\n"),
        ("", "\n"),
    ]
    if plan:
        name, plus, minus = _changes_headline(plan)
        header += [
            ("", "代码将恢复 "),
            ("class:plus", f"+{plus}"),
            ("class:minus", f" -{minus}"),
            ("", f"（{name}）\n"),
        ]
    else:
        header.append(("class:desc", "代码没有改动\n"))
    header.append(("class:desc", "⚠ 手动或经 run_command 改动的文件不受回退影响\n\n"))

    channel_options = [
        ("both", "对话和代码都回退", None),
        ("conversation", "只回退对话", None),
        ("code", "只回退代码", None),
        (None, "取消", None),
    ]
    async with in_terminal():
        channel = await _ListPicker("将回退到你发出这条消息之前：", channel_options, header=header).run()
    if channel is None:
        return True

    # 先回退文件再回退对话，顺序不能反：回退对话会把这个检查点连同它之后的一起丢掉
    if channel in ("both", "code"):
        fh.rewind_files(cp)
    if channel in ("both", "conversation"):
        # 截断内存里的历史，并整体重写会话文件
        state.history = state.history[: cp.history_index]
        session.rewrite_messages(state.session_id, state.history)
        # 被截掉的对话对应的检查点不再有意义，丢弃
        fh.drop_from(cp)
        # 把被回退的那条输入回填到输入框，用户改一改就能重发
        state.pending_input = cp.prompt
        # token 计数按剩下的历史重算，进程内的 API 调用记录清空，记忆允许再召回
        _recount_tokens(state)
        state.last_api_calls.clear()
        state.surfaced_memories = set()
        state.compact_failures = 0
    # 无论回退哪个通道，readFileState 登记的内容和 mtime 都已过期，必须清空，否则先读后写闸门会放行基于旧内容的编辑
    state.read_file_state = ReadFileState()

    console.print(f"已回退到 {datetime.fromtimestamp(cp.timestamp):%H:%M} 的检查点\n")
    return True


async def cmd_compact(state: SessionState, args: str = "") -> bool:
    """
    手动压缩上下文，可以带补充指令，如 /compact 重点保留文件改动。
    """
    try:
        await compact.run_compact(state, custom_instructions=args)
    except Exception as e:
        console.print(f"[red]压缩失败：{type(e).__name__}: {e}[/]\n")
    return True


def cmd_status(state: SessionState) -> bool:
    console.print(f"模型：           {state.model_name}")
    console.print(f"权限模式：       {permissions.state.mode}")
    console.print(f"历史消息条数：    {len(state.history)}")
    used = compact.context_tokens(state.history)
    threshold = compact.compact_threshold()
    if used:
        console.print(f"当前上下文占用（估算）：{used:,} / {threshold:,} tokens（{used * 100 // threshold}%）")
    else:
        console.print(f"当前上下文占用（估算）：暂无数据（自动压缩阈值 {threshold:,} tokens）")
    console.print(f"累计输入 tokens：{state.input_tokens}")
    console.print(f"累计输出 tokens：{state.output_tokens}\n")
    return True


def cmd_mcp(state: SessionState) -> bool:
    """
    显示所有已配置 MCP server 的连接状态和工具清单。
    """
    if not mcp_servers.RECORDS:
        console.print(f"未配置任何 MCP server。可在项目根目录的 .mcp.json 或 {mcp_servers.USER_CONFIG} 中添加。\n")
        return True
    for record in mcp_servers.RECORDS:
        console.print(f"[bold]{record.server.id}[/]  [dim]{escape(record.transport)}[/]")
        if record.status == "connected":
            console.print(f"  [green]已连接[/]，{len(record.tool_names)} 个工具")
            for name in record.tool_names:
                console.print(f"    - {name}")
        else:
            console.print(f"  [red]连接失败[/]：{escape(record.error)}")
        console.print()
    return True


def cmd_memory(state: SessionState) -> bool:
    """
    显示记忆目录、所有记忆文件和 MEMORY.md 索引。
    """
    console.print(f"记忆目录：{store.memory_dir()}\n")
    headers = store.scan_memory_files()
    if not headers:
        console.print("(还没有任何记忆，记忆会随对话逐渐积累)\n")
        return True
    for h in headers:
        age = store.age_text(store.age_days(h.mtime))
        console.print(escape(f"  [{h.type or '?'}] {h.filename}（{age}）"))
        console.print(f"      [dim]{escape(h.description)}[/]")
    console.print()
    index = store.read_index()
    if index:
        console.print("MEMORY.md 索引：")
        console.print(Padding(Markdown(index), (0, 0, 0, 2)))
        console.print()
    return True


async def cmd_dream(state: SessionState) -> bool:
    """
    手动触发记忆合并整理，跳过自动闸门，方便立刻看到效果。
    """
    console.print("开始整理记忆，可能需要一会儿...\n")
    try:
        await background.dream(list(state.history), force=True, session_id=state.session_id)
    except Exception as e:
        console.print(f"[red]整理失败：{type(e).__name__}: {e}[/]\n")
    console.print()
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
    "resume": Command("resume", "恢复历史会话", cmd_resume),
    "rewind": Command("rewind", "回退到之前的检查点", cmd_rewind),
    "compact": Command("compact", "压缩上下文（可带补充指令）", cmd_compact, takes_args=True),
    "status": Command("status", "显示当前会话状态", cmd_status),
    "mcp": Command("mcp", "查看 MCP server 状态和工具", cmd_mcp),
    "memory": Command("memory", "查看长期记忆", cmd_memory),
    "dream": Command("dream", "立即整理合并长期记忆", cmd_dream),
    "api-detail": Command("api-detail", "显示最近一轮 model API 调用详情", cmd_api_detail),
    "help": Command("help", "显示可用命令", cmd_help),
    "exit": Command("exit", "退出程序", cmd_exit),
}
