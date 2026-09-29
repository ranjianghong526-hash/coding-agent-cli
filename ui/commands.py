"""会话状态、斜杠命令和消息展示的协调层。

阅读时分三组：SessionState/Command 定义数据；print_* 展示消息；cmd_* 处理命令。
COMMANDS 注册表供 main.py 查找命令，本模块不直接向大模型发送请求。
"""

from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional
from uuid import uuid4

from prompt_toolkit import PromptSession
from session_store import list_sessions, load_session, save_session
from permissions import PermissionState

from rich.markdown import Heading, Markdown
from rich.markup import escape
from rich.padding import Padding
from rich.rule import Rule

# 复用底层渲染对象；render.py 不依赖 commands.py，保持依赖方向清晰。
# print_welcome_banner 在这里导入后，也可以由 main.py 从本模块取到。
from .render import console, print_step, print_welcome_banner


class LeftAlignedHeading(Heading):
    """
    rich 默认把 Markdown 标题渲染成居中对齐，宽终端里看着像错位，覆盖成左对齐。
    """
    def __rich_console__(self, console, options):
        """实现 Rich 的渲染协议，用 yield 提供要输出的左对齐标题对象。"""
        text = self.text
        text.justify = "left"
        yield text


# 全局替换 Markdown 的标题渲染元素
# 模块被导入后，这个替换会影响同一进程中后续的 Rich Markdown 标题。
Markdown.elements["heading_open"] = LeftAlignedHeading


@dataclass
class SessionState:
    """
    跨命令共享的会话状态，主循环把它传给每个命令处理函数。
    """
    # 每份会话拥有独立列表，保存用户、模型及工具之间的完整消息历史。
    history: list = field(default_factory=list)
    # 累计计数由 main.apply_result() 在每轮结束后更新，/new 将它们归零。
    input_tokens: int = 0
    output_tokens: int = 0
    # 只用于展示当前配置的模型名，不在这里切换模型。
    model_name: str = ""
    # 最近一轮 user input 触发的所有 model API 调用记录
    last_api_calls: list = field(default_factory=list)
    # 每个新会话使用独立文件；saved_messages 标记已成功落盘的消息数量。
    session_id: str = field(default_factory=lambda: uuid4().hex)
    saved_messages: int = 0
    # 权限属于当前程序运行，/new 和 /resume 不重置，也不从 JSONL 恢复授权。
    permissions: PermissionState = field(default_factory=PermissionState)

    def __post_init__(self):
        # deps 中的任务存储与聊天会话使用相同编号，避免串到其他会话。
        self.permissions.tasks.bind(self.session_id)


@dataclass
class Command:
    """把命令名、帮助文本和处理函数放在一起，供注册表统一管理。"""
    # name 不带 /；description 是 /help 中显示的说明。
    name: str
    description: str
    # handler 返回 False 表示主循环应当退出
    # Callable 描述函数类型；引号中的 SessionState 是类型名称的字符串写法。
    # /resume 需要异步等待选择，其他命令仍可直接返回 bool。
    handler: Callable[["SessionState"], bool | Awaitable[bool]]


def print_divider() -> None:
    """
    每轮交互之前打印一条分割线，区分输入区域。Rule 会自适应终端宽度。
    """
    console.print(Rule(style="grey50"))


def _truncate(text, limit: int = 120) -> str:
    """
    截断并 escape，用于 tool 参数 / 返回值 / 用户输入这类可能过长的内容。
    """
    # 内容可能不是字符串，先转换后去掉首尾空白，再按字符数截断。
    # 截断仅影响展示，不会改变保存的原始消息和实际工具执行结果。
    text = str(text).strip()
    text = text if len(text) <= limit else text[:limit] + "..."
    # Rich 用 [green] 等标签控制样式；转义用户内容，避免其中的方括号变成样式。
    return escape(text)


def _full(text) -> str:
    """
    完整显示，只做 escape 不截断，用于 thinking 和 assistant text 这种用户关心的内容。
    """
    # 这里处理的是 Rich markup；最终回复的 Markdown 渲染走另一个函数。
    return escape(str(text).strip())


def _format_part_line(part) -> Optional[str]:
    """
    把一条消息里的单个 part 格式化为带 Rich markup 的字符串。
    版式：图标 + role 标签独占一行，内容换行到下一行，不用「|」分隔。
    """
    # 内容行统一缩进 2 格，和图标（占 2 格：图标 + 空格）后的 role 名对齐
    # 每种 part 需要不同的标签和取值字段，不同于只按 user / assistant 分类。
    kind = part.part_kind
    if kind == "user-prompt":
        return f"[cyan]❯ user[/]\n  {_truncate(part.content)}"
    if kind == "thinking":
        # 只有模型响应实际提供 thinking 片段时才显示，并非自行推测模型思考。
        # thinking 整块 dim，弱化视觉权重；不截断，完整保留思考过程
        return f"[dim]✻ thinking[/]\n  [dim]{_full(part.content)}[/]"
    if kind == "text":
        content = (part.content or "").strip()
        if not content:
            return None
        # assistant 是用户最关心的最终回答，完整显示
        return f"[green]● assistant[/]\n  {_full(content)}"
    if kind == "tool-call":
        # 显示模型请求执行的工具名和参数，这个格式化函数本身不执行工具。
        # 命令、路径动辄上百字符，参数放宽到 500 字符再截断
        return f"[yellow]⏺ tool_call[/]\n  [yellow dim]{part.tool_name}({_truncate(part.args, 500)})[/]"
    if kind == "tool-return":
        # 工具返回结果是下一次模型判断的依据，这里只显示摘要。
        return f"[magenta]✔ tool_return[/]\n  [magenta dim]{part.tool_name} -> {_truncate(part.content)}[/]"
    if kind == "retry-prompt":
        # 工具抛 ModelRetry 后，SDK 生成 retry-prompt 把错误反馈给模型
        return f"[yellow]✘ tool_retry[/]\n  [yellow dim]{part.tool_name} -> {_truncate(part.content)}[/]"
    # 尚未支持的片段类型跳过显示；消息仍保留在会话历史中。
    return None


def print_assistant_markdown(content: str) -> None:
    """
    模型的回复天然是 Markdown 格式，整块渲染出来，而不是打印原始文本。
    """
    # 先打印角色，再将回复中的标题、列表、代码块交给 Markdown 对象处理。
    console.print("[green]● assistant[/]")
    # Markdown 是块级渲染对象，没法跟在行内前缀后面，所以另起一行渲染；左缩进 2 格和 role 名对齐
    console.print(Padding(Markdown(content), (0, 0, 0, 2)))


def print_part(part) -> None:
    """
    渲染单个消息 part：assistant 文本走 Markdown 块渲染，其余 part 是单行文本。
    """
    # 正文单独走 Markdown；工具参数等内容使用转义后的普通文本。
    if part.part_kind == "text":
        content = (part.content or "").strip()
        if content:
            print_assistant_markdown(content)
            # 每个 role block 末尾留一个空行，块与块之间不那么挤
            console.print()
        return
    # None 表示不支持或无需展示的内容；有文本时再拆成标签和正文。
    line = _format_part_line(part)
    if line:
        label, _, body = line.partition("\n")
        # body 形如 "  [markup]…"，去掉字面前导 2 空格，交给 print_step 用 Padding 缩进（折行续行也保持缩进）
        print_step(label, body[2:])


def print_agent_steps(new_messages) -> None:
    """
    主循环里调用：显示这一轮 Agent 新增的中间过程（thinking、文本、工具调用、工具返回）。
    """
    # 两层遍历分别处理消息和消息中的内容片段，顺序沿用框架返回的历史顺序。
    # 这是已有消息的批量展示辅助函数；实时执行由 main.run_agent() 调用 print_part()。
    for msg in new_messages:
        for part in msg.parts:
            # 主循环里不重复显示用户刚刚输入的内容
            if part.part_kind == "user-prompt":
                continue
            print_part(part)


def cmd_exit(state: SessionState) -> bool:
    """输出告别文字，并通过 False 通知 main.py 结束输入循环。"""
    console.print("再见 👋")
    return False


def cmd_help(state: SessionState) -> bool:
    """从注册表生成帮助列表，新增注册项会自动出现在这里。"""
    console.print("可用命令：")
    for cmd in COMMANDS.values():
        # :<10 表示宽度至少 10 个字符、左对齐，让说明的起始位置更整齐。
        console.print(f"  /{cmd.name:<10} {cmd.description}")
    console.print()
    return True


def cmd_new(state: SessionState) -> bool:
    """
    开启新会话：清空历史、token 计数、API 调用记录。
    """
    # 切换前补存尚未写入的历史；失败则不清空状态，避免丢失可继续保存的内容。
    save_session(state)
    new_id = uuid4().hex
    state.permissions.tasks.bind(new_id)
    # 只切换当前会话，旧 JSONL 文件和工具写入的文件都保留。
    state.history.clear()
    state.input_tokens = 0
    state.output_tokens = 0
    state.last_api_calls.clear()
    state.session_id = new_id
    state.saved_messages = 0
    state.permissions.files.clear()
    console.print("已开启新会话\n")
    return True


def cmd_status(state: SessionState) -> bool:
    """展示当前会话的本地统计，读取这些数据不需要调用模型接口。"""
    # 历史条数不是用户提问次数，一轮需求可能产生多条模型和工具消息。
    console.print(f"会话编号：       {state.session_id}")
    console.print(f"模型：           {state.model_name}")
    console.print(f"权限模式：       {state.permissions.mode}")
    console.print(f"历史消息条数：    {len(state.history)}")
    console.print(f"累计输入 tokens：{state.input_tokens}")
    console.print(f"累计输出 tokens：{state.output_tokens}\n")
    return True


async def cmd_resume(state: SessionState) -> bool:
    """按编号选择项目中的历史会话，校验完成后一次性替换当前内存状态。"""
    sessions, unreadable = list_sessions()
    for name in unreadable:
        console.print(f"跳过无法读取的会话：{name}", style="yellow", markup=False)
    if not sessions:
        console.print("当前项目还没有已保存的会话。\n")
        return True
    console.print("历史会话（最近更新的在前）：")
    for index, session in enumerate(sessions, 1):
        updated = session.updated_at.astimezone().strftime("%Y-%m-%d %H:%M")
        console.print(f"  {index}. {updated}  {session.title}  [{session.session_id[:8]}]", markup=False)
    try:
        # 复用现有输入依赖，不引入新的选择菜单库；空输入或 q 表示取消。
        choice = (await PromptSession().prompt_async("选择会话编号（回车或 q 取消）：")).strip()
    except (EOFError, KeyboardInterrupt):
        console.print("已取消恢复。\n")
        return True
    if not choice or choice.lower() == "q":
        return True
    if not choice.isdecimal() or not 1 <= int(choice) <= len(sessions):
        console.print("编号无效，当前会话未改变。\n")
        return True
    selected = sessions[int(choice) - 1]
    # 恢复前先补存当前会话；保存失败时仍保留当前状态，不贸然切换。
    save_session(state)
    # 列表展示之后当前会话可能刚补存过，重新加载才能拿到最新完整历史。
    selected = load_session(selected.session_id)
    # 先校验并加载任务，再修改会话字段；坏任务文件不能导致半次切换。
    state.permissions.tasks.bind(selected.session_id)
    state.history = selected.history
    state.session_id = selected.session_id
    state.saved_messages = len(selected.history)
    state.input_tokens = selected.input_tokens
    state.output_tokens = selected.output_tokens
    state.last_api_calls.clear()
    # 模型继续使用当前 core.py 配置；不因历史文件而偷偷切换模型。
    state.permissions.files.clear()
    console.print(f"已恢复会话 {selected.session_id}，共 {len(selected.history)} 条消息。\n")
    print_tasks(state)
    return True


def print_tasks(state: SessionState) -> None:
    """完整显示当前任务进度；使用 Text，模型生成的标题不解释成 Rich 标签。"""
    from rich.table import Table
    from rich.text import Text

    tasks = state.permissions.tasks.list()
    if not tasks:
        return
    names = {"pending": "待办", "in_progress": "进行中", "completed": "已完成"}
    table = Table(title="当前会话任务", expand=True)
    for heading in ("编号", "状态", "任务"):
        table.add_column(heading)
    for task in tasks:
        table.add_row(str(task["id"]), names[task["status"]], Text(task["subject"]))
    console.print(table)


def cmd_tasks(state: SessionState) -> bool:
    """本地查看清单，不发起模型请求、不重新执行任务。"""
    if state.permissions.tasks.list():
        print_tasks(state)
    else:
        console.print("当前会话没有任务。\n")
    return True


def cmd_api_detail(state: SessionState) -> bool:
    """
    显示最近一轮 user input 触发的所有 model API 调用元数据。
    """
    # 读取上轮的日志快照；当前命令不会自己再发起一次模型请求。
    if not state.last_api_calls:
        console.print("(还没有任何模型调用记录，先发一条消息再来看)\n")
        return True

    console.print(f"最近一轮共发起 {len(state.last_api_calls)} 次 model API 调用\n")

    # enumerate(..., 1) 为每次模型调用生成从 1 开始的展示编号。
    # 一个用户需求可以对应多个 Call，常见原因是工具结果需要交回模型继续判断。
    for i, call in enumerate(state.last_api_calls, 1):
        console.print(f"[bold]Call #{i}[/]")
        console.print(f"  Request:")
        console.print(f"    model:        {call.model}")
        console.print(f"    messages:     {call.messages_count} 条")
        if call.last_part is not None:
            # 复用片段格式化函数展示请求尾部内容，而不是输出全部历史。
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


# 命令名 -> Command 对象；handler 只接收共享 state，统一用 bool 控制是否继续。
# 添加命令时定义 cmd_* 函数并在此注册，main.py 的分发逻辑通常无需修改。
COMMANDS = {
    "tasks": Command("tasks", "查看当前会话的任务清单", cmd_tasks),
    "new": Command("new", "开启新会话", cmd_new),
    "resume": Command("resume", "选择并恢复当前项目的历史会话", cmd_resume),
    "status": Command("status", "显示当前会话状态", cmd_status),
    "api-detail": Command("api-detail", "显示最近一轮 model API 调用详情", cmd_api_detail),
    "help": Command("help", "显示可用命令", cmd_help),
    "exit": Command("exit", "退出程序", cmd_exit),
}
