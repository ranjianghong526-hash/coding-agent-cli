"""主动提问的参数协议与终端表单；回答由真人输入，不由模型代填。"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator
from prompt_toolkit.application import Application
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.widgets import Frame, TextArea


class QuestionOption(BaseModel):
    """选项名称是提交给模型的值，描述只用于帮助用户判断。"""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    label: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=300)


class Question(BaseModel):
    """没有 options 时直接输入文本；有选项时仍始终提供自定义回答。"""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=1000)
    header: str = Field(default="问题", min_length=1, max_length=12)
    options: list[QuestionOption] = Field(default_factory=list, max_length=4)
    multi_select: StrictBool = False

    @model_validator(mode="after")
    def check_options(self):
        if len(self.options) == 1:
            raise ValueError("请提供 2～4 个选项，或不提供选项让用户自由输入")
        labels = [option.label.casefold() for option in self.options]
        if len(labels) != len(set(labels)):
            raise ValueError("同一问题的选项名称不能重复")
        return self


class QuestionAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    question: str
    selected_options: list[str]
    custom_answer: str


class QuestionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    status: Literal["answered", "cancelled"]
    answers: list[QuestionAnswer]


# 此标记放在 SDK 消息的 metadata 中，不发送给模型，也不是模型参数。
USER_ANSWER_METADATA = {"source": "ask_user_question:terminal"}


class QuestionForm:
    """每题保留草稿，全部答完并提交后才构造工具结果。"""

    def __init__(self, questions: list[Question]):
        if not 1 <= len(questions) <= 4:
            raise ValueError("一次提问必须有 1～4 题")
        self.questions = questions
        self.index = 0  # len(questions) 表示最后的提交页。
        self.cursor = [0] * len(questions)
        self.selected: list[set[int]] = [set() for _ in questions]
        self.custom = [""] * len(questions)
        self.editing = False
        self.notice = ""
        self.editor = TextArea(multiline=False, prompt="自定义回答：")
        self.control = FormattedTextControl(self.render, focusable=True)
        from prompt_toolkit.layout import ConditionalContainer
        body = HSplit([
            Window(self.control, wrap_lines=True),
            ConditionalContainer(self.editor, filter=Condition(lambda: self.editing)),
        ])
        keys = KeyBindings()

        @keys.add("c-c")
        @keys.add("c-d")
        @keys.add("escape")
        def cancel(event):
            # 取消不提交部分答案，避免模型把尚未确认的草稿当作决定。
            event.app.exit(result=QuestionResult(status="cancelled", answers=[]))

        @keys.add("up", filter=Condition(lambda: not self.editing))
        @keys.add("down", filter=Condition(lambda: not self.editing))
        def move(event):
            if self.index < len(self.questions):
                delta = -1 if event.key_sequence[0].key == "up" else 1
                count = len(self.questions[self.index].options) + 1
                self.cursor[self.index] = (self.cursor[self.index] + delta) % count

        @keys.add("left", filter=Condition(lambda: not self.editing))
        @keys.add("right", filter=Condition(lambda: not self.editing))
        def switch(event):
            delta = -1 if event.key_sequence[0].key == "left" else 1
            self.show(max(0, min(len(self.questions), self.index + delta)))

        @keys.add(" ", filter=Condition(lambda: not self.editing))
        def toggle(event):
            if self.index < len(self.questions) and self.questions[self.index].multi_select:
                self.choose(advance=False)

        @keys.add("enter")
        def enter(event):
            if self.editing:
                text = self.editor.text.strip()
                if not text:
                    self.notice = "请输入回答，或按 Esc 取消。"
                    return
                self.custom[self.index] = text
                self.editing = False
                self.app.layout.focus(self.control)
                self.show(self.index + 1)
            elif self.index == len(self.questions):
                missing = next((i for i in range(len(self.questions)) if not self.answered(i)), None)
                if missing is not None:
                    self.show(missing)
                    self.notice = "这题还没有回答，请明确选择或输入。"
                else:
                    event.app.exit(result=self.result())
            elif self.questions[self.index].multi_select:
                # 多选用空格选中；回车确认，空选择不能默认为第一个选项。
                if self.cursor[self.index] == len(self.questions[self.index].options):
                    self.start_edit()
                elif self.answered(self.index):
                    self.show(self.index + 1)
                else:
                    self.notice = "请用空格勾选至少一项，或选择其他输入回答。"
            else:
                self.choose(advance=True)

        self.app = Application(layout=Layout(Frame(body, title="Agent 向你提问"), self.control),
                               key_bindings=keys, full_screen=False)
        if not questions[0].options:
            self.start_edit()

    def answered(self, index: int) -> bool:
        return bool(self.selected[index] or self.custom[index])

    def show(self, index: int) -> None:
        self.index = index
        self.notice = ""
        if index < len(self.questions) and not self.questions[index].options:
            self.start_edit()

    def start_edit(self) -> None:
        self.editing = True
        self.editor.text = self.custom[self.index]
        self.app.layout.focus(self.editor)

    def choose(self, *, advance: bool) -> None:
        question = self.questions[self.index]
        position = self.cursor[self.index]
        if position == len(question.options):
            if question.multi_select and self.custom[self.index] and not advance:
                # 多选的“其他”也能用空格取消，回车则重新编辑已有文本。
                self.custom[self.index] = ""
                return
            if not question.multi_select:
                self.selected[self.index].clear()
            self.start_edit()
        elif question.multi_select:
            selected = self.selected[self.index]
            selected.remove(position) if position in selected else selected.add(position)
        else:
            self.selected[self.index] = {position}
            self.custom[self.index] = ""
            if advance:
                self.show(self.index + 1)

    def result(self) -> QuestionResult:
        return QuestionResult(status="answered", answers=[QuestionAnswer(
            question=question.question,
            selected_options=[option.label for i, option in enumerate(question.options) if i in self.selected[index]],
            custom_answer=self.custom[index],
        ) for index, question in enumerate(self.questions)])

    def render(self):
        # 使用纯文本片段，不把模型提供的内容解释为 HTML 或终端样式标签。
        tabs = " | ".join(f"{'>' if i == self.index else ''}{q.header}" for i, q in enumerate(self.questions))
        tabs += " | " + (">提交" if self.index == len(self.questions) else "提交")
        lines = [tabs, ""]
        if self.index == len(self.questions):
            for i, question in enumerate(self.questions):
                answer = self.result().answers[i]
                lines.append(f"{question.header}：{'、'.join(answer.selected_options + ([answer.custom_answer] if answer.custom_answer else [])) or '(未回答)'}")
            lines.append("回车提交；← 返回检查答案。")
        else:
            question = self.questions[self.index]
            lines.append(question.question)
            for i, option in enumerate(question.options):
                mark = "[x]" if i in self.selected[self.index] else "[ ]"
                lines.append(f"{'>' if i == self.cursor[self.index] else ' '} {mark} {option.label}  {option.description}")
            custom_mark = "[x]" if self.custom[self.index] else "[ ]"
            lines.append(f"{'>' if self.cursor[self.index] == len(question.options) else ' '} {custom_mark} 其他（输入自定义文本）")
            lines.append("↑↓ 移动；←→ 切换题目；空格勾选多选；回车选择/确认；Esc 取消。")
        if self.notice:
            lines.append(self.notice)
        return [("", "\n".join(lines))]

    async def run(self) -> QuestionResult:
        return await self.app.run_async()


async def ask_questions(questions: list[Question]) -> QuestionResult:
    """等待终端表单返回；等待期间 Agent 不会进入下一次模型请求。"""
    return await QuestionForm(questions).run()
