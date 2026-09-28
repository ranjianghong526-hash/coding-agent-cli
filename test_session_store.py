"""持久化与恢复的离线回归测试，所有文件写入临时目录，不调用真实模型。"""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import DummyInput
from prompt_toolkit.output import DummyOutput
from pydantic_ai import Agent, models
from pydantic_ai.messages import (
    ModelRequest, ModelResponse,
    TextPart, ThinkingPart, ToolCallPart, ToolReturnPart, UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel

with patch.dict(os.environ, {"API_KEY": "test-placeholder"}):
    with create_app_session(input=DummyInput(), output=DummyOutput()):
        import main

import session_store as store
from ui import commands
from ui.commands import SessionState


def conversation(prompt="创建你好文件"):
    """包含工具请求与返回的完整一轮，用于验证恢复后结构及调用 ID 不丢失。"""
    return [
        ModelRequest(parts=[UserPromptPart(prompt)]),
        ModelResponse(parts=[ThinkingPart("测试思考"), ToolCallPart("write_file", {"path": "1.txt", "content": "你好"}, tool_call_id="call-1")]),
        ModelRequest(parts=[ToolReturnPart("write_file", "已写入", tool_call_id="call-1")]),
        ModelResponse(parts=[TextPart("完成")]),
    ]


class SessionStoreTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / ".sessions"
        self.directory_patch = patch.object(store, "SESSION_DIR", self.root)
        self.directory_patch.start()
        self.addCleanup(self.directory_patch.stop)
        self.console_patch = patch.object(commands, "console", Mock())
        self.console_patch.start()
        self.addCleanup(self.console_patch.stop)

    def state(self, prompt="创建你好文件"):
        return SessionState(history=conversation(prompt), model_name="test", input_tokens=10, output_tokens=5)

    def path(self, state):
        return self.root / f"{state.session_id}.jsonl"

    async def choose(self, state, choice):
        # 模拟异步菜单输入；普通输入、Ctrl-D 和 Ctrl-C 使用同一个测试入口。
        prompt = AsyncMock(side_effect=choice) if isinstance(choice, BaseException) else AsyncMock(return_value=choice)
        with patch.object(commands, "PromptSession", return_value=Mock(prompt_async=prompt)):
            return await main.handle_command("/resume", state)

    def test_roundtrip_and_append_only_new_messages(self):
        state = self.state()
        store.save_session(state)
        first_line = self.path(state).read_bytes()
        store.save_session(state)
        self.assertEqual(self.path(state).read_bytes(), first_line)
        state.history += [ModelRequest(parts=[UserPromptPart("继续解释")]), ModelResponse(parts=[TextPart("说明")])]
        state.input_tokens += 3
        state.output_tokens += 2
        store.save_session(state)
        lines = self.path(state).read_bytes().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(len(store.StoredTurn.model_validate_json(lines[1]).messages), 2)
        self.assertTrue(self.path(state).read_bytes().startswith(first_line))
        self.assertIn("你好", self.path(state).read_text(encoding="utf-8"))
        restored = store.load_session(state.session_id)
        self.assertEqual(restored.history, state.history)
        self.assertEqual((restored.input_tokens, restored.output_tokens), (13, 7))
        self.assertEqual(restored.title, "创建你好文件")
        self.assertEqual(restored.history[2].parts[0].tool_call_id, "call-1")

    def test_incomplete_tail_is_ignored_and_repaired(self):
        state = self.state()
        store.save_session(state)
        valid = self.path(state).read_bytes()
        with self.path(state).open("ab") as file:
            file.write(b'{"version":1,"messages":')
        self.assertEqual(store.load_session(state.session_id).history, state.history)
        state.history += conversation("下一轮")
        store.save_session(state)
        self.assertTrue(self.path(state).read_bytes().startswith(valid))
        self.assertEqual(len(self.path(state).read_bytes().splitlines()), 2)
        self.assertEqual(store.load_session(state.session_id).history, state.history)

    def test_corrupt_complete_record_is_reported_without_deletion(self):
        valid, broken = self.state("正常会话"), self.state("损坏会话")
        store.save_session(valid)
        store.save_session(broken)
        with self.path(broken).open("ab") as file:
            file.write(b'{"broken":true}\n')
        with self.assertRaises(ValueError):
            store.load_session(broken.session_id)
        sessions, unreadable = store.list_sessions()
        self.assertEqual([s.session_id for s in sessions], [valid.session_id])
        self.assertIn(broken.session_id, unreadable[0])
        self.assertTrue(self.path(broken).exists())
        before = self.path(broken).read_bytes()
        broken.history += conversation("不能覆盖坏文件")
        with self.assertRaises(ValueError):
            store.save_session(broken)
        self.assertEqual(self.path(broken).read_bytes(), before)

    def test_invalid_id_cannot_be_used_as_path(self):
        with self.assertRaises(ValueError):
            store.load_session("../outside")

    def test_failed_write_keeps_save_position_and_next_round_catches_up(self):
        state = self.state()
        with patch.object(store.os, "fsync", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                store.save_session(state)
        self.assertEqual(state.saved_messages, 0)
        # 即使失败发生在整行已写完之后，再次保存也不会重复这一轮。
        store.save_session(state)
        self.assertEqual(len(self.path(state).read_bytes().splitlines()), 1)
        self.assertEqual(state.saved_messages, len(state.history))

    def test_apply_result_save_failure_preserves_memory_and_token_counts(self):
        state = SessionState(model_name="test")
        result = Mock()
        result.all_messages.return_value = conversation()
        result.usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        with patch.object(main, "save_session", side_effect=PermissionError), patch.object(main, "console", Mock()) as console:
            main.apply_result(state, result)
        self.assertEqual(state.history, result.all_messages())
        self.assertEqual((state.input_tokens, state.output_tokens), (10, 5))
        self.assertEqual(state.saved_messages, 0)
        self.assertIn("保存失败", str(console.print.call_args_list))
        result.all_messages.return_value = state.history + conversation("下一轮")
        main.apply_result(state, result)
        self.assertEqual(store.load_session(state.session_id).history, state.history)
        self.assertEqual((state.input_tokens, state.output_tokens), (20, 10))

    def test_new_preserves_old_file_and_assigns_new_identity(self):
        state = self.state()
        old_id, old_history = state.session_id, list(state.history)
        commands.cmd_new(state)
        self.assertEqual(store.load_session(old_id).history, old_history)
        self.assertNotEqual(state.session_id, old_id)
        self.assertEqual((state.history, state.saved_messages, state.input_tokens, state.output_tokens), ([], 0, 0, 0))
        state.history = conversation("新会话")
        store.save_session(state)
        self.assertEqual(len(store.list_sessions()[0]), 2)

    async def test_resume_cancel_invalid_or_empty_does_not_change_state(self):
        state = self.state()
        before = state.__dict__.copy()
        self.assertEqual(await main.handle_command("/resume", state), "continue")
        self.assertEqual(state.__dict__, before)
        store.save_session(self.state("已有历史"))
        for choice in ["", "q", "invalid", "0", "999", EOFError(), KeyboardInterrupt()]:
            with self.subTest(choice=repr(choice)):
                self.assertEqual(await self.choose(state, choice), "continue")
                self.assertEqual(state.__dict__, before)

    async def test_selecting_current_session_preserves_unsaved_tail(self):
        state = self.state()
        store.save_session(state)
        state.history += conversation("尚未落盘")
        expected = list(state.history)
        await self.choose(state, "1")
        self.assertEqual(state.history, expected)
        self.assertEqual(store.load_session(state.session_id).history, expected)

    async def test_save_failure_prevents_new_or_resume_switch(self):
        previous = self.state("上一次")
        store.save_session(previous)
        current = self.state("当前未保存")
        before = current.__dict__.copy()
        with patch.object(commands, "save_session", side_effect=PermissionError):
            with self.assertRaises(PermissionError):
                commands.cmd_new(current)
            with self.assertRaises(PermissionError):
                await self.choose(current, "1")
        self.assertEqual(current.__dict__, before)

    async def test_resumed_history_reaches_next_model_and_appends_same_file(self):
        old = self.state("此前的工作")
        store.save_session(old)
        current = SessionState(model_name="current-model", last_api_calls=["temporary log"])
        await self.choose(current, "1")
        self.assertEqual(current.session_id, old.session_id)
        self.assertEqual(current.model_name, "current-model")
        self.assertEqual(current.last_api_calls, [])
        requests = []

        def respond(messages, info):
            requests.append(messages)
            return ModelResponse(parts=[TextPart("接着说明")])

        test_agent = Agent(FunctionModel(respond))
        with patch.object(models, "ALLOW_MODEL_REQUESTS", False), patch.object(main, "agent", test_agent):
            with patch.object(main, "console", Mock()), patch.object(main, "print_part"):
                result = await main.run_agent("继续", current)
        self.assertEqual(requests[0][:len(old.history)], old.history)
        main.apply_result(current, result)
        self.assertEqual(len(self.path(old).read_bytes().splitlines()), 2)
        self.assertEqual(store.load_session(old.session_id).history, current.history)
        self.assertEqual(len(store.list_sessions()[0]), 1)


if __name__ == "__main__":
    unittest.main()
