"""
ask_user_question 工具：让模型在执行过程中主动向用户提多选题，收集偏好 / 澄清歧义 / 让用户做决定。

UI 用 prompt_toolkit 自己画，版式如下：
顶部 chip 导航栏（← chip chip ✔ Submit →）、完整问句、带 ❯ 指针和 dim description 的纵向选项列表。

in_terminal() 把常驻 Repl 输入框暂时让给本 picker Application，结束后 Repl 自动恢复。
"""
from dataclasses import dataclass, field

from prompt_toolkit.application import Application, in_terminal
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.styles import Style
from pydantic_ai.exceptions import ModelRetry

from ui.render import console

# 配色方案：当前 chip 紫色背景、Submit 完成态绿色、指针/当前 label 蓝色、description 灰色
# 抽到模块层是为了每次开 picker 都复用同一份样式，不必反复 from_dict
_STYLE = Style.from_dict({
    "nav-arrow": "#9ca3af",
    "nav-arrow-dim": "#4b5563",
    "chip-current": "bg:#5b21b6 #ffffff bold",
    "chip-answered": "#9ca3af",
    "chip-pending": "#6b7280",
    "chip-submit-ready": "#10b981 bold",
    "question": "bold",
    "row-current": "#3b82f6 bold",
    "row": "",
    "label-current": "#3b82f6 bold",
    "label": "",
    "label-dim": "#9ca3af",
    "desc": "#6b7280",
    "footer": "#6b7280",
    "text-input-label": "#9ca3af",
})

# 文本输入态的固定提示行，同样模块层常量化避免每帧重建 FormattedText
_TEXT_INPUT_LABEL = FormattedText([("class:text-input-label", "  > 自定义文本：")])


@dataclass
class QuestionOption:
    """
    一道题里的单个选项。
    """
    # 选项展示文本（1-5 个词为佳）
    label: str
    # 选项含义 / 选了之后会发生什么
    description: str = ""


@dataclass
class Question:
    """
    一道多选题。
    """
    # 完整问句，应以问号结尾
    question: str
    # ≤12 字符的标签，渲染在顶部 chip 导航栏上
    header: str
    # 2-4 个选项；末尾会自动追加「其他」让用户输入自定义文本
    options: list[QuestionOption]
    # True 走多选（Space 勾选、Enter 确认本题），False 走单选（Enter 选中并自动跳下一题）
    multi_select: bool = False


@dataclass
class _QuestionState:
    """
    每道题的运行时状态，由 picker Application 持有。
    单选题约束 |picks| ≤ 1；多选题允许任意选项子集。
    custom_text 是「其他」选项填的文本——多选时与 picks 叠加，单选时与 picks 互斥。
    """
    # 选项列表里的高亮行下标（0..len(options)），最后一行是「其他」
    cursor: int = 0
    # 选中的 option.label 集合；单选最多 1 个，多选可任意多个
    picks: set[str] = field(default_factory=set)
    # 用户通过「其他」入口输入的自定义文本
    custom_text: str = ""


class _Picker:
    """
    手绘的 picker，用 prompt_toolkit 自定义渲染多选题版式。

    焦点模型：一个 q_idx 字段就够——0..len(questions)-1 表示「在那道题上」，
    q_idx == len(questions) 表示「焦点在 Submit 上」。不再单独维护 on_submit 标志位。
    in_text_input 是正交的子模式，只在编辑「其他」答案时为真。
    """

    def __init__(self, questions: list[Question]):
        self.questions = questions
        # 0..len-1 → 在某道题上；len → 在 Submit 上
        self.q_idx = 0
        self.states = [_QuestionState() for _ in questions]
        self.in_text_input = False
        self.text_buffer = Buffer(multiline=False)
        self.cancelled = False
        self.app: Application = self._build_app()

    # ------------------------------------------------------------------
    # 内部状态查询
    # ------------------------------------------------------------------

    def _on_submit(self) -> bool:
        return self.q_idx == len(self.questions)

    def _current_q(self) -> Question:
        return self.questions[self.q_idx]

    def _current_state(self) -> _QuestionState:
        return self.states[self.q_idx]

    def _num_rows(self) -> int:
        # 选项行数 + 末尾「其他」一行
        return len(self._current_q().options) + 1

    def _is_answered(self, idx: int) -> bool:
        s = self.states[idx]
        return bool(s.picks) or bool(s.custom_text)

    def _all_answered(self) -> bool:
        return all(self._is_answered(i) for i in range(len(self.questions)))

    def _answer_for(self, idx: int) -> str:
        """
        把一道题的最终答案格式化成字符串：选中的 option label（按选项顺序）加上「其他」自定义文本（若有）。
        """
        q = self.questions[idx]
        s = self.states[idx]
        items = [opt.label for opt in q.options if opt.label in s.picks]
        if s.custom_text:
            items.append(s.custom_text)
        return ", ".join(items) if items else "[空]"

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------

    def _render_nav(self):
        """
        顶部 chip 导航栏：← [chip1] chip2 chip3 ✔ Submit →
        """
        # 一次性算好每题的答完标志，避免下面 chip + can_next + submit-ready 三处再各算一次
        answered_flags = [self._is_answered(i) for i in range(len(self.questions))]
        on_submit = self._on_submit()
        all_answered = all(answered_flags)
        parts: list[tuple[str, str]] = []

        # 左箭头：不在第 0 题（或在 Submit 上）时可用，否则 dim
        can_prev = on_submit or self.q_idx > 0
        parts.append(("class:nav-arrow" if can_prev else "class:nav-arrow-dim", "← "))

        for i, q in enumerate(self.questions):
            answered = answered_flags[i]
            symbol = "✔" if answered else "□"
            # 焦点在这道题（on_submit 时 q_idx == len，自然不会命中）
            if i == self.q_idx:
                cls = "class:chip-current"
            elif answered:
                cls = "class:chip-answered"
            else:
                cls = "class:chip-pending"
            parts.append((cls, f" {symbol} {q.header} "))
            parts.append(("", " "))

        # Submit chip：全答完后绿色（提示可提交），焦点落上时高亮紫底
        if on_submit:
            submit_cls = "class:chip-current"
        elif all_answered:
            submit_cls = "class:chip-submit-ready"
        else:
            submit_cls = "class:chip-pending"
        parts.append((submit_cls, " ✔ Submit "))

        # 右箭头：还有下一题，或在最后一题且全答完可以跳到 Submit
        can_next = not on_submit and (self.q_idx < len(self.questions) - 1 or all_answered)
        parts.append(("class:nav-arrow" if can_next else "class:nav-arrow-dim", "  →"))
        return FormattedText(parts)

    def _render_question_text(self):
        if self._on_submit():
            return FormattedText([("class:question", "所有问题已回答，按 Enter 提交答案")])
        return FormattedText([("class:question", self._current_q().question)])

    def _render_options(self):
        """
        纵向选项列表。当前行有 ❯ 指针；description 在下一行 dim 渲染。
        单选：● 标记选中项（picks 只有 1 个元素）。多选：每项前显示 ☐/☑ 复选框。
        """
        if self._on_submit():
            return FormattedText([])

        q = self._current_q()
        s = self._current_state()
        lines: list[tuple[str, str]] = []

        for i, opt in enumerate(q.options):
            is_cursor = (i == s.cursor)
            is_picked = opt.label in s.picks
            pointer = "❯" if is_cursor else " "
            if q.multi_select:
                mark = "☑ " if is_picked else "☐ "
            else:
                mark = "● " if is_picked else "  "

            cls_row = "class:row-current" if is_cursor else "class:row"
            cls_label = "class:label-current" if is_cursor else "class:label"
            lines.append((cls_row, f" {pointer}  "))
            lines.append((cls_label, f"{i + 1}. {mark}{opt.label}"))
            lines.append(("", "\n"))

            if opt.description:
                lines.append(("class:desc", f"      {opt.description}"))
                lines.append(("", "\n"))

        # 末尾「其他」行：选过则把已输入文本回显出来
        other_idx = len(q.options)
        is_cursor = (s.cursor == other_idx)
        pointer = "❯" if is_cursor else " "
        cls_row = "class:row-current" if is_cursor else "class:row"
        cls_label = "class:label-current" if is_cursor else "class:label-dim"
        suffix = f"：{s.custom_text}" if s.custom_text else ""
        other_text = f"{other_idx + 1}. 其他（输入自定义文本）{suffix}"
        lines.append((cls_row, f" {pointer}  "))
        lines.append((cls_label, other_text))
        lines.append(("", "\n"))

        return FormattedText(lines)

    def _render_footer(self):
        """
        底部帮助行，内容随当前模式变化。
        """
        if self.in_text_input:
            return FormattedText([("class:footer", "  输入完成按 Enter 确认 · Esc 取消文本输入")])
        if self._on_submit():
            return FormattedText([("class:footer", "  Enter 提交 · ← 返回上一题 · Esc 取消")])
        q = self._current_q()
        hints = ["↑↓ 选项"]
        if q.multi_select:
            hints += ["Space 勾选", "Enter 确认本题"]
        else:
            hints.append("Enter 选中并跳下一题")
        hints += ["←→ 切换题目", "Esc 取消"]
        return FormattedText([("class:footer", "  " + " · ".join(hints))])

    # ------------------------------------------------------------------
    # 动作
    # ------------------------------------------------------------------

    def _advance(self):
        """
        把焦点往前推一格：下一题，或者已在最后一题时跳到 Submit。
        """
        if not self._on_submit():
            self.q_idx += 1

    def _move(self, delta: int):
        """
        在选项列表里把光标移动 ±1，到边界后循环。在 Submit 或文本输入态下是 no-op。
        """
        if self.in_text_input or self._on_submit():
            return
        s = self._current_state()
        s.cursor = (s.cursor + delta) % self._num_rows()

    def _prev_question(self):
        # q_idx == len → 退回最后一题；其余情况就是简单 q_idx -= 1（已经统一在一条语句里）
        if self.in_text_input or self.q_idx == 0:
            return
        self.q_idx -= 1

    def _next_question(self):
        if self.in_text_input or self._on_submit():
            return
        # 不是最后一题就直接前进；最后一题只在全答完时才允许跳到 Submit
        if self.q_idx < len(self.questions) - 1 or self._all_answered():
            self.q_idx += 1

    def _enter_text_input(self):
        """
        切到文本输入模式；用本题已输入的自定义文本预填 buffer。
        """
        self.in_text_input = True
        self.text_buffer.text = self._current_state().custom_text
        # 把光标放在末尾，方便接着改
        self.text_buffer.cursor_position = len(self.text_buffer.text)

    def _commit_text(self):
        """
        把输入的自定义文本回写到题目状态。
        单选：自定义文本会清掉之前可能选过的列表项，并自动跳下一题。
        多选：自定义文本与列表项叠加，焦点不动，让用户继续勾别的。
        """
        if not self.in_text_input:
            return
        text = self.text_buffer.text.strip()
        s = self._current_state()
        s.custom_text = text
        self.in_text_input = False
        if text and not self._current_q().multi_select:
            # 单选下自定义文本作为唯一答案
            s.picks.clear()
            self._advance()

    def _toggle_pick(self, label: str):
        """
        在当前题的 picks 集合里 toggle 一个 label。Enter（多选）和 Space 共用。
        """
        s = self._current_state()
        if label in s.picks:
            s.picks.remove(label)
        else:
            s.picks.add(label)

    def _on_enter(self):
        """
        选择模式下的 Enter：行为随光标位置和单选/多选变化。
        """
        if self._on_submit():
            # 焦点在 Submit 上：提交结束
            self.app.exit(result=True)
            return

        q = self._current_q()
        s = self._current_state()

        # 光标在「其他」行：切到文本输入模式
        if s.cursor == len(q.options):
            self._enter_text_input()
            return

        if q.multi_select:
            # 多选 Enter 是「确认本题」——toggle 由 Space 负责，避免 Space 勾完按 Enter 又被取消的反直觉行为
            if self._is_answered(self.q_idx):
                self._advance()
            return

        # 单选 Enter：选中当前覆盖之前的（含 custom_text），跳下一题
        opt = q.options[s.cursor]
        s.picks = {opt.label}
        s.custom_text = ""
        self._advance()

    def _on_space(self):
        """
        Space 键：多选时 toggle 当前复选框，单选时忽略。
        """
        if self._on_submit() or self.in_text_input:
            return
        q = self._current_q()
        if not q.multi_select:
            return
        s = self._current_state()
        if s.cursor == len(q.options):
            self._enter_text_input()
            return
        self._toggle_pick(q.options[s.cursor].label)

    def _on_escape(self):
        """
        Esc：文本输入模式下丢弃输入回到选择态；否则取消整个对话框。
        """
        if self.in_text_input:
            self.in_text_input = False
            self.text_buffer.text = ""
            return
        self.cancelled = True
        self.app.exit(result=False)

    # ------------------------------------------------------------------
    # Application 构造
    # ------------------------------------------------------------------

    def _build_app(self) -> Application:
        kb = KeyBindings()
        sel = Condition(lambda: not self.in_text_input)
        text = Condition(lambda: self.in_text_input)

        # 堆叠装饰器让一个回调绑定多个键（kb.add 多参数是「按键序列」而非「任选其一」）
        @kb.add("up", filter=sel)
        @kb.add("k", filter=sel)
        def _(event):
            self._move(-1)

        @kb.add("down", filter=sel)
        @kb.add("j", filter=sel)
        def _(event):
            self._move(1)

        @kb.add("enter", filter=sel)
        def _(event):
            self._on_enter()

        @kb.add("space", filter=sel)
        def _(event):
            self._on_space()

        @kb.add("left", filter=sel)
        def _(event):
            self._prev_question()

        @kb.add("right", filter=sel)
        def _(event):
            self._next_question()

        @kb.add("escape", filter=sel)
        @kb.add("c-c", filter=sel)
        def _(event):
            self._on_escape()

        # 文本输入模式下的键位
        @kb.add("enter", filter=text)
        def _(event):
            self._commit_text()

        @kb.add("escape", filter=text)
        def _(event):
            self._on_escape()

        # 各区域 Window
        nav_win = Window(FormattedTextControl(self._render_nav), height=1)
        question_win = Window(FormattedTextControl(self._render_question_text), wrap_lines=True, dont_extend_height=True)
        options_win = Window(FormattedTextControl(self._render_options), dont_extend_height=True)
        text_win = ConditionalContainer(
            HSplit([
                # 文本输入态的提示行，内容是模块层常量，每帧返回同一个对象
                Window(FormattedTextControl(lambda: _TEXT_INPUT_LABEL), height=1),
                Window(BufferControl(buffer=self.text_buffer), height=1),
            ]),
            filter=text,
        )
        footer_win = Window(FormattedTextControl(self._render_footer), height=1)

        layout = Layout(HSplit([
            nav_win,
            Window(height=1, char=" "),
            question_win,
            Window(height=1, char=" "),
            options_win,
            text_win,
            Window(height=1, char=" "),
            footer_win,
        ]))

        return Application(
            layout=layout,
            key_bindings=kb,
            style=_STYLE,
            full_screen=False,
            mouse_support=False,
        )

    async def run(self) -> dict[str, str] | None:
        """
        跑 picker，返回答案字典，被取消时返回 None。
        """
        submitted = await self.app.run_async()
        if self.cancelled or not submitted:
            return None
        return {
            q.question: self._answer_for(i)
            for i, q in enumerate(self.questions)
        }


async def ask_user_question(questions: list[Question]) -> str:
    """
    在执行过程中向用户提 1-4 道多选题，通过 TUI picker 收集答案。

    每道题给 2-4 个选项；末尾固定一个「其他」入口让用户输入自定义文本。
    使用场景：需求有歧义、有多种合理实现要选、或者你拿不准方向时。

    Args:
        questions: 1-4 个多选题，每题至少 2 个、至多 4 个选项
    """
    # 入参校验：硬性限制为 1-4 题 / 每题 2-4 选项
    if not (1 <= len(questions) <= 4):
        raise ModelRetry("一次提 1-4 个问题")
    for q in questions:
        if not (2 <= len(q.options) <= 4):
            raise ModelRetry(f'问题 "{q.question}" 选项必须 2-4 个')

    # in_terminal 把终端让给 picker Application；结束后常驻 Repl 输入框会自动恢复
    async with in_terminal():
        # 留一行空白把 picker 和上面的 tool_call 行视觉上隔开
        console.print()
        picker = _Picker(questions)
        answers = await picker.run()
        console.print()

    if answers is None:
        return "用户取消了提问，未提供任何回答。请等待用户进一步指示，不要自作主张。"

    # 拼成自然语言喂回模型，对齐 CC mapToolResultToToolResultBlockParam 的格式
    formatted = "; ".join(f'"{q}" → "{a}"' for q, a in answers.items())
    return f"用户回答如下：{formatted}。请基于这些回答继续。"
