import asyncio
import re
import time
from html import escape as html_escape

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import ConditionalContainer, Float, FloatContainer, HSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.patch_stdout import patch_stdout
from rich.markup import escape

import permissions
from mentions import list_candidate_files
from .render import console

# 常驻输入区：输入框整个会话期间挂在屏幕底部不消失，Agent 输出通过 patch_stdout 打印在它上方，请求期间按 ESC / Ctrl+C 能立刻打断。

# working 指示器的转圈动画帧
_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# task 面板每行的渲染模板，键是 task.status，值里的 {s} 是 escape 后的 subject
_TASK_ROW_TEMPLATES = {
    "completed": "  <ansigreen>✔</ansigreen> <s><ansibrightblack>{s}</ansibrightblack></s>",
    "in_progress": "  <ansiyellow>▶</ansiyellow> <b>{s}</b>",
    "pending": "  <ansibrightblack>☐ {s}</ansibrightblack>",
}


class AtFileCompleter(Completer):
    """
    @ 引用补全：用户正在输入 @token 时，把匹配的文件列成下拉候选。
    """
    def get_completions(self, document, complete_event):
        # 只看光标前的文本，匹配正在输入的那个 @token（@ 到光标之间的非空白串）
        match = re.search(r"@(\S*)$", document.text_before_cursor)
        if match is None:
            return
        token = match.group(1).lower()
        for path in list_candidate_files():
            # 简单的子串模糊匹配：token 是路径的子串就算命中，不区分大小写
            if token in path.lower():
                # start_position 取负的 token 长度，选中后用完整路径替换掉已经敲进去的那部分
                yield Completion(path, start_position=-len(match.group(1)))


class Repl:
    # 常驻输入区，由 main 注入 on_submit 回调来处理每一行输入

    def __init__(self, state):
        self.state = state
        # on_submit 由 main 注入，处理一行输入（命令或交给 Agent）
        self._on_submit = None
        # 当前处理输入的后台任务，ESC / Ctrl+C 据此打断；None 表示空闲
        self._task = None
        # 是否正在请求模型，决定上方 working... 指示器的显隐
        self.working = False
        self._work_start = 0.0
        self._frame = 0
        # 给输入缓冲区挂上 @ 文件补全器，complete_while_typing 让候选随敲随出，不用按 Tab
        self._buffer = Buffer(
            multiline=False,
            history=InMemoryHistory(),
            completer=AtFileCompleter(),
            complete_while_typing=True,
        )
        self.app = self._build_app()

    def _prompt_prefix(self, line_number, wrap_count):
        return HTML("<ansicyan>❯ </ansicyan>")

    def _task_lines(self):
        # 每次 spinner tick (~100ms) 都会调到这里，store 已经把 list() 做成内存命中，所以贴着改贴着画也不肉痛
        tasks = self.state.tasks_store.list()
        if not tasks:
            return HTML("")
        rows = [_TASK_ROW_TEMPLATES[t.status].format(s=html_escape(t.subject)) for t in tasks]
        return HTML("\n".join(rows))

    def _working_line(self):
        # 输入框上方那行：转圈帧 + 已耗时 + 打断提示，只在请求期间显示
        frame = _SPINNER[self._frame % len(_SPINNER)]
        elapsed = int(time.monotonic() - self._work_start)
        return HTML(f"<ansigreen>{frame}</ansigreen> <b>Working…</b><ansibrightblack>（已耗时 {elapsed}s · 按 esc 打断）</ansibrightblack>")

    def _mode_line(self):
        # 输入框下方那行：当前权限模式 + 切换提示
        mode = permissions.state.mode
        return HTML(f"  <ansimagenta><b>▶▶ {mode}</b></ansimagenta><ansibrightblack>（Shift+Tab 切换）</ansibrightblack>")

    def _divider(self):
        # 一条横向分割线
        return Window(height=1, char="─", style="fg:ansibrightblack")

    def _build_app(self):
        # 从上到下：working 指示器（请求中才显示）、task 面板（有 task 才显示）、分割线、输入行、分割线、模式行
        body = HSplit(
            [
                # working 指示器排在最上面，task 面板紧贴它下方，整组贴着输入框上方分割线
                ConditionalContainer(
                    Window(FormattedTextControl(self._working_line), height=1),
                    filter=Condition(lambda: self.working),
                ),
                # task 面板按内容自适应高度，列表空时 ConditionalContainer 直接塌成 0 行不占地方，不需要手动开关
                ConditionalContainer(
                    Window(
                        FormattedTextControl(self._task_lines),
                        height=Dimension(min=0),
                        dont_extend_height=True,
                    ),
                    filter=Condition(lambda: len(self.state.tasks_store.list()) > 0),
                ),
                self._divider(),
                # 输入行只占内容高度，多行自动换行撑开
                Window(
                    BufferControl(buffer=self._buffer),
                    get_line_prefix=self._prompt_prefix,
                    height=Dimension(min=1),
                    wrap_lines=True,
                    dont_extend_height=True,
                ),
                self._divider(),
                Window(FormattedTextControl(self._mode_line), height=1),
            ]
        )
        # 用 FloatContainer 托起输入区，再叠一层补全菜单浮层，菜单跟着光标位置浮现
        layout = Layout(
            FloatContainer(
                content=body,
                floats=[
                    Float(
                        xcursor=True,
                        ycursor=True,
                        content=CompletionsMenu(max_height=8, scroll_offset=1),
                    ),
                ],
            )
        )
        return Application(layout=layout, key_bindings=self._build_key_bindings())

    def _build_key_bindings(self):
        kb = KeyBindings()

        @kb.add("enter")
        def _(event):
            self._on_enter()

        @kb.add("c-c")
        def _(event):
            # 请求中打断当前 Agent loop；空闲时退出程序
            if self._task is not None:
                self._task.cancel()
            else:
                event.app.exit()

        @kb.add("escape")
        def _(event):
            # 请求中打断；空闲时清空输入行
            if self._task is not None:
                self._task.cancel()
            else:
                self._buffer.reset()

        @kb.add("c-d")
        def _(event):
            # 空闲且输入为空时退出
            if self._task is None and not self._buffer.text:
                event.app.exit()

        @kb.add("s-tab")
        def _(event):
            # 循环切换权限模式
            permissions.cycle_mode()

        return kb

    def _on_enter(self):
        # 补全菜单正开着且有高亮项时，回车先采纳补全，不提交输入
        if self._buffer.complete_state and self._buffer.complete_state.current_completion:
            self._buffer.apply_completion(self._buffer.complete_state.current_completion)
            return
        # 请求中不接受新提交（输入框仍在，只是回车不触发新一轮）
        if self._task is not None:
            return
        text = self._buffer.text.strip()
        if not text:
            self._buffer.reset()
            return
        # 存进输入历史，清空输入行
        self._buffer.history.append_string(text)
        self._buffer.reset()
        # 把这行回显到上方滚动区，留下记录（输入框常驻，不回显的话提交后这行就没了）
        self._echo_input(text)
        # 把处理丢进后台任务，回车处理立刻返回，输入框继续渲染、随时能打断
        self._task = self.app.create_background_task(self._process(text))

    def _echo_input(self, text):
        # 回显刚提交的一行：上下分割线夹住 ❯ 文本，和输入框观感一致
        rule = "─" * console.width
        console.print(f"[bright_black]{rule}[/]")
        console.print(f"[cyan]❯[/] {escape(text)}")
        console.print(f"[bright_black]{rule}[/]")
        console.print()

    async def _process(self, text):
        # 后台任务：交给 on_submit，统一兜住打断和异常
        try:
            await self._on_submit(text)
        except asyncio.CancelledError:
            console.print("\n[bold yellow]已中断[/]\n")
        except Exception as e:
            console.print(f"\n[bold red]✗ {type(e).__name__}: {e}[/]\n")
        finally:
            self._task = None
            self.working = False
            # /rewind 回退对话后把原输入回填到输入框，用户改一改就能重发
            if self.state.pending_input:
                self._buffer.text = self.state.pending_input
                self._buffer.cursor_position = len(self.state.pending_input)
                self.state.pending_input = ""
            self.app.invalidate()

    def start_working(self):
        # 进入请求中状态，上方显示 working...；请求结束 / 被打断后由 _process 的 finally 统一清掉
        self.working = True
        self._work_start = time.monotonic()
        self.app.invalidate()

    def exit(self):
        # 结束常驻输入区（/exit 命令用）
        self.app.exit()

    async def run(self, on_submit):
        # 运行 REPL 直到用户退出；on_submit 是处理一行输入的异步回调
        self._on_submit = on_submit
        # 转圈动画的心跳：请求期间定时重绘
        ticker = asyncio.ensure_future(self._tick())
        try:
            # patch_stdout 让 Agent 的 rich 输出打印在输入框上方而不是冲掉它
            with patch_stdout(raw=True):
                await self.app.run_async()
        finally:
            ticker.cancel()

    async def _tick(self):
        try:
            while True:
                await asyncio.sleep(0.1)
                if self.working:
                    self._frame += 1
                    self.app.invalidate()
        except asyncio.CancelledError:
            pass
