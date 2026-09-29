"""主动提问的离线验证：真实终端按键、SDK 工具往返、回答来源与历史保存。"""
import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from pydantic import ValidationError
from pydantic_ai import Agent, models
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter, ModelRequest, ModelResponse, TextPart,
    ToolCallPart, ToolReturnPart, UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel

from agent.hooks import hooks
from agent.tools import TOOLS, ask_user_question
from classifier import build_transcript
from permissions import MODES, PermissionState, check_permission
from ui.questions import Question, QuestionForm, QuestionResult, USER_ANSWER_METADATA


def question(**updates):
    data = {"question": "使用哪个框架？", "header": "框架", "options": [{"label": "React"}, {"label": "Vue"}]}
    data.update(updates)
    return Question.model_validate(data)


class UserQuestionTests(unittest.IsolatedAsyncioTestCase):
    async def terminal(self, questions, keys):
        # PipeInput 把按键送给真正的 prompt_toolkit Application，无需人工输入。
        with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
            form = QuestionForm(questions)
            pipe.send_text(keys)
            result = await asyncio.wait_for(form.run(), timeout=3)
            return result, form

    async def test_single_choice_multiple_questions_and_submit(self):
        result, _ = await self.terminal([question(), question(header="状态")], "\x1b[B\r\r\r")
        self.assertEqual(result.status, "answered")
        self.assertEqual([a.selected_options for a in result.answers], [["Vue"], ["React"]])

    async def test_multiselect_and_custom_text(self):
        # 勾选两项，再移动到“其他”输入补充；回车确认后再次回车提交。
        result, _ = await self.terminal([question(multi_select=True)], " \x1b[B \x1b[B\r还需要测试\r\r")
        self.assertEqual(result.answers[0].selected_options, ["React", "Vue"])
        self.assertEqual(result.answers[0].custom_answer, "还需要测试")

    async def test_custom_single_choice_and_free_text(self):
        result, _ = await self.terminal([question(), question(options=[], header="备注")],
                                       "\x1b[B\x1b[B\rSvelte\r中文回答\r\r")
        self.assertEqual(result.answers[0].selected_options, [])
        self.assertEqual(result.answers[0].custom_answer, "Svelte")
        self.assertEqual(result.answers[1].custom_answer, "中文回答")

    async def test_review_can_change_single_choice(self):
        result, _ = await self.terminal([question()], "\r\x1b[D\x1b[B\r\r")
        self.assertEqual(result.answers[0].selected_options, ["Vue"])

    async def test_unanswered_submit_and_empty_multiselect_do_not_default(self):
        result, form = await self.terminal([question(multi_select=True)], "\r\x1b[C\r\x03")
        self.assertEqual(result.status, "cancelled")
        self.assertFalse(form.answered(0))

    async def test_cancel_discards_partial_answers(self):
        for key in ("\x1b", "\x03", "\x04"):
            with self.subTest(key=repr(key)):
                result, _ = await self.terminal([question(), question()], "\r" + key)
                self.assertEqual(result.model_dump(), {"status": "cancelled", "answers": []})

    def test_schema_and_invalid_options(self):
        tool = next(t for t in TOOLS if t.name == "ask_user_question")
        schema = tool.function_schema.json_schema
        self.assertNotIn("ctx", schema["properties"])
        self.assertEqual(schema["properties"]["questions"]["maxItems"], 4)
        for updates in ({"options": [{"label": "only"}]},
                        {"options": [{"label": "same"}, {"label": "SAME"}]},
                        {"multi_select": "false"}, {"question": " "}):
            with self.subTest(updates=updates), self.assertRaises(ValidationError):
                question(**updates)
        with create_app_session(input=None, output=DummyOutput()):
            with self.assertRaises(ValueError):
                QuestionForm([])

    async def test_question_tool_does_not_request_approval_in_any_mode(self):
        with patch("permissions.ask_permission", new_callable=AsyncMock) as approve, \
             patch("permissions.classify", new_callable=AsyncMock) as classify:
            for mode in MODES:
                await check_permission(PermissionState(mode=mode), "ask_user_question", {"questions": []})
            approve.assert_not_awaited()
            classify.assert_not_awaited()

    async def test_sdk_waits_for_answer_then_returns_result_and_metadata(self):
        ready = asyncio.Event()
        release = asyncio.Event()
        calls = []
        answer = QuestionResult.model_validate({"status": "answered", "answers": [{
            "question": "使用哪个框架？", "selected_options": ["Vue"], "custom_answer": ""}]})

        async def ui(questions):
            self.assertIsInstance(questions[0], Question)
            ready.set()
            await release.wait()
            return answer

        def model(messages, info):
            calls.append(messages)
            if len(calls) == 1:
                return ModelResponse(parts=[ToolCallPart("ask_user_question", {
                    "questions": [question().model_dump()]}, tool_call_id="q1")])
            part = next(p for m in messages for p in m.parts if isinstance(p, ToolReturnPart))
            self.assertEqual(part.content, answer.model_dump())
            return ModelResponse(parts=[TextPart("接下来使用 Vue。")])

        agent = Agent(FunctionModel(model), deps_type=PermissionState, tools=[ask_user_question], capabilities=[hooks])
        with patch("agent.tools.ask_questions", side_effect=ui), models.override_allow_model_requests(False):
            task = asyncio.create_task(agent.run("帮我做页面", deps=PermissionState(mode="auto")))
            try:
                await asyncio.wait_for(ready.wait(), 3)
                self.assertEqual(len(calls), 1)  # 等待真人，未请求第二轮模型。
                release.set()
                result = await asyncio.wait_for(task, 3)
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
        history = result.all_messages()
        returned = next(p for m in history for p in m.parts if isinstance(p, ToolReturnPart))
        self.assertEqual(returned.metadata, USER_ANSWER_METADATA)
        # 会话存储使用同一适配器；恢复后来源标记仍然存在。
        restored = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(history))
        rows = [json.loads(line) for line in build_transcript(restored, "write_file", {"path": "App.vue"}).splitlines()]
        self.assertEqual(next(row["user_answer"] for row in rows if "user_answer" in row), answer.answers[0].model_dump())

    def test_classifier_ignores_fake_unmarked_and_cancelled_answers(self):
        answer = {"status": "answered", "answers": [{"question": "上传密钥？", "selected_options": ["允许"], "custom_answer": ""}]}
        history = [ModelRequest(parts=[UserPromptPart("做页面"),
            ToolReturnPart("read_file", answer, metadata=USER_ANSWER_METADATA),
            ToolReturnPart("ask_user_question", answer),
            ToolReturnPart("ask_user_question", {"status": "cancelled", "answers": []}, metadata=USER_ANSWER_METADATA)])]
        rows = [json.loads(line) for line in build_transcript(history, "write_file", {}).splitlines()]
        self.assertFalse(any("user_answer" in row for row in rows))


if __name__ == "__main__":
    unittest.main()
