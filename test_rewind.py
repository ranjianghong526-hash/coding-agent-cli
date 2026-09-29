"""回退离线测试：真实文件、消息边界、失败轮次、恢复后的分支和冲突保护。"""
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
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel

with patch.dict(os.environ, {"API_KEY": "test-placeholder"}):
    with create_app_session(input=DummyInput(), output=DummyOutput()):
        import main

import session_store
from agent.hooks import hooks
from agent.tools import read_file, edit_file, write_file
from classifier import build_transcript
from memory_store import MemoryStore
from permissions import PermissionState
from rewind import apply_rewind
from rewind_store import RewindStore
from ui import commands
from ui.commands import SessionState


class RewindTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        for patcher in (patch.object(session_store, "SESSION_DIR", self.root / ".sessions"),
                        patch.object(commands, "console", Mock()), patch.object(main, "console", Mock()),
                        patch.object(main, "print_part", Mock())):
            patcher.start()
            self.addCleanup(patcher.stop)
        deps = PermissionState(mode="bypass", rewind=RewindStore(self.root), memory=MemoryStore(self.root))
        self.state = SessionState(model_name="test", permissions=deps)
        self.ctx = SimpleNamespace(deps=deps)
        self.store = deps.rewind
        self.path = self.root / "app.py"
        self.path.write_bytes(b"port = 8000\r\n")

    def begin(self, prompt="修改端口"):
        return self.store.begin(prompt, len(self.state.history), self.state.permissions.tasks.document)

    def write(self, content, path=None):
        path = path or self.path
        if path.exists():
            read_file(self.ctx, str(path), force=True)
        result = write_file(self.ctx, str(path), content)
        self.assertTrue(result.startswith("已写入"), result)

    def finish_turn(self, prompt="修改端口"):
        self.store.end()
        self.state.history.extend([ModelRequest(parts=[UserPromptPart(prompt)]), ModelResponse(parts=[TextPart("完成")])])
        session_store.save_session(self.state)

    def test_each_edit_backed_up_and_reload(self):
        point = self.begin()
        self.write("port = 8080\n")
        read_file(self.ctx, str(self.path), force=True)
        self.assertTrue(edit_file(self.ctx, str(self.path), "8080", "9000").startswith("已编辑"))
        self.finish_turn()
        edits = self.store.document.checkpoints[0].edits
        self.assertEqual(len(edits), 2)
        self.assertEqual(self.store._blob_path(edits[0].before).read_bytes(), b"port = 8000\r\n")
        fresh = RewindStore(self.root)
        fresh.bind(self.state.session_id)
        self.assertEqual(fresh.choices(2)[0].id, point)
        fresh.restore_files(point, 2)
        self.assertEqual(self.path.read_bytes(), b"port = 8000\r\n")

    def test_files_only_restore_existing_remove_new_keep_chat_and_tasks(self):
        point = self.begin()
        new = self.root / "new.txt"
        self.write("changed")
        self.write("new", new)
        self.state.permissions.tasks.create("保留的任务")
        self.finish_turn()
        history = list(self.state.history)
        current_id = self.state.session_id
        self.assertEqual(apply_rewind(self.state, point, "files"), 2)
        self.assertEqual(self.path.read_bytes(), b"port = 8000\r\n")
        self.assertFalse(new.exists())
        self.assertEqual(self.state.session_id, current_id)
        self.assertEqual(self.state.history[:2], history)
        self.assertEqual(self.state.permissions.tasks.list()[0]["subject"], "保留的任务")
        self.assertEqual(self.state.permissions.files.read_file_state, {})
        transcript = build_transcript(self.state.history, "read_file", {"path": str(self.path)})
        self.assertNotIn("用户通过 /rewind", transcript)
        # 恢复操作也登记版本，所以重复回退不是“外部修改”。
        apply_rewind(self.state, point, "files")

    def test_conversation_only_forks_before_selected_user_and_keeps_files(self):
        self.begin("第一轮")
        self.write("first")
        self.finish_turn("第一轮")
        self.state.permissions.tasks.create("第一轮任务")
        point = self.begin("第二轮")
        self.write("second")
        self.state.permissions.tasks.update(1, status="completed")
        self.finish_turn("第二轮")
        original_id = self.state.session_id
        original_jsonl = session_store._session_path(original_id).read_bytes()
        self.state.input_tokens = 20
        apply_rewind(self.state, point, "conversation")
        self.assertNotEqual(self.state.session_id, original_id)
        self.assertEqual(len(self.state.history), 3)  # 两条旧消息 + 代码保留的程序提醒。
        self.assertEqual(self.path.read_text(), "second")
        self.assertEqual(self.state.next_prompt, "第二轮")
        self.assertEqual(self.state.input_tokens, 20)  # 已实际消费的用量不因回退退款。
        self.assertEqual(self.state.permissions.tasks.get(1)["status"], "pending")
        self.assertEqual(session_store._session_path(original_id).read_bytes(), original_jsonl)
        self.assertEqual(len(session_store.load_session(original_id).history), 4)
        self.assertEqual(len(self.state.permissions.rewind.choices(2)), 1)

    def test_both_then_continue_and_rewind_earlier_after_fork(self):
        first = self.begin("第一轮")
        self.write("first")
        self.finish_turn("第一轮")
        second = self.begin("第二轮")
        new = self.root / "new.txt"
        self.write("second")
        self.write("created", new)
        self.finish_turn("第二轮")
        apply_rewind(self.state, second, "both")
        self.assertEqual(self.path.read_text(), "first")
        self.assertFalse(new.exists())
        self.assertEqual(len(self.state.history), 2)
        self.ctx = SimpleNamespace(deps=self.state.permissions)
        self.store = self.state.permissions.rewind
        self.begin("新的第二轮")
        self.write("alternative")
        self.finish_turn("新的第二轮")
        apply_rewind(self.state, first, "both")
        self.assertEqual(self.path.read_bytes(), b"port = 8000\r\n")
        self.assertEqual(self.state.history, [])

    def test_external_edit_blocks_every_file_before_any_restore(self):
        point = self.begin()
        another = self.root / "another.txt"
        self.write("changed")
        self.write("new", another)
        self.finish_turn()
        self.path.write_text("user editor change")
        original_id = self.state.session_id
        for mode in ("files", "both"):
            with self.assertRaisesRegex(ValueError, "外部修改"):
                apply_rewind(self.state, point, mode)
            self.assertEqual(self.path.read_text(), "user editor change")
            self.assertTrue(another.exists())
            self.assertEqual(self.state.session_id, original_id)

    def test_corrupt_or_missing_backup_never_changes_files(self):
        point = self.begin()
        self.write("changed")
        self.finish_turn()
        edit = self.store.document.checkpoints[0].edits[0]
        self.store._blob_path(edit.before).write_bytes(b"bad")
        with self.assertRaisesRegex(ValueError, "备份损坏"):
            apply_rewind(self.state, point, "files")
        self.assertEqual(self.path.read_text(), "changed")

    def test_capture_failure_prevents_file_write(self):
        self.begin()
        read_file(self.ctx, str(self.path))
        with patch.object(self.store, "_commit", side_effect=PermissionError):
            result = write_file(self.ctx, str(self.path), "new")
        self.assertTrue(result.startswith("[错误]"))
        self.assertEqual(self.path.read_bytes(), b"port = 8000\r\n")

    def test_restore_failure_rolls_back_already_restored_files(self):
        point = self.begin()
        other = self.root / "other.txt"
        self.write("changed")
        self.write("created", other)
        self.finish_turn()
        original = self.store._restore
        calls = 0

        def fail_second(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise PermissionError("simulated")
            return original(*args)

        before = self.store.document.model_dump()
        with patch.object(self.store, "_restore", side_effect=fail_second):
            with self.assertRaisesRegex(OSError, "已撤销"):
                apply_rewind(self.state, point, "files")
        self.assertEqual(self.path.read_text(), "changed")
        self.assertEqual(other.read_text(), "created")
        self.assertEqual(self.store.document.model_dump(), before)

    def test_fork_save_failure_preserves_current_session_and_code(self):
        point = self.begin()
        self.write("changed")
        self.finish_turn()
        original_id, original_history = self.state.session_id, list(self.state.history)
        with patch("rewind.save_session", side_effect=[None, PermissionError]):
            with self.assertRaises(PermissionError):
                apply_rewind(self.state, point, "both")
        self.assertEqual(self.state.session_id, original_id)
        self.assertEqual(self.state.history, original_history)
        self.assertEqual(self.path.read_text(), "changed")
        self.assertEqual(len(list(session_store.SESSION_DIR.glob("*.rewind.json"))), 1)

    def test_partial_restore_failure_keeps_recovery_branch(self):
        point = self.begin()
        other = self.root / "other.txt"
        self.write("changed")
        self.write("created", other)
        self.finish_turn()
        original = RewindStore._restore
        calls = 0

        def fail_restore_and_rollback(*args):
            nonlocal calls
            calls += 1
            if calls >= 2:
                raise PermissionError("cannot restore or compensate")
            return original(*args)

        original_id = self.state.session_id
        with patch.object(RewindStore, "_restore", side_effect=fail_restore_and_rollback):
            with self.assertRaisesRegex(OSError, "恢复日志保留"):
                apply_rewind(self.state, point, "both")
        self.assertEqual(self.state.session_id, original_id)
        self.assertEqual(len(list(session_store.SESSION_DIR.glob("*.rewind.json"))), 2)
        sessions, unreadable = session_store.list_sessions()
        self.assertEqual(unreadable, [])
        self.assertEqual(len(sessions), 2)

    def test_tampered_journal_path_rejected_on_bind(self):
        self.begin()
        self.write("changed")
        self.finish_turn()
        document = self.store.document.model_copy(deep=True)
        document.checkpoints[0].edits[0].path = str(self.root.parent / "outside.txt")
        self.store.path(self.state.session_id).write_text(document.model_dump_json(), encoding="utf-8")
        with self.assertRaises(ValueError):
            RewindStore(self.root).bind(self.state.session_id)
        self.assertEqual(self.path.read_text(), "changed")

    async def test_real_sdk_failed_round_is_discoverable_and_recoverable(self):
        count = 0

        def respond(messages, info):
            nonlocal count
            count += 1
            if count == 1:
                return ModelResponse(parts=[ToolCallPart("read_file", {"path": str(self.path)}, "read1")])
            if count == 2:
                return ModelResponse(parts=[ToolCallPart("edit_file", {"path": str(self.path), "old_string": "8000", "new_string": "8080"}, "edit1")])
            raise RuntimeError("model failed after edit")

        test_agent = Agent(FunctionModel(respond), deps_type=PermissionState, tools=[read_file, edit_file], capabilities=[hooks])
        with patch.object(models, "ALLOW_MODEL_REQUESTS", False), patch.object(main, "agent", test_agent):
            with self.assertRaises(RuntimeError):
                await main.run_agent("修改端口", self.state)
        self.assertIn("8080", self.path.read_text())
        self.assertEqual(self.state.history, [])
        self.assertIsNone(self.store.active_id)
        sessions, unreadable = session_store.list_sessions()
        self.assertEqual(unreadable, [])
        self.assertEqual(sessions[0].session_id, self.state.session_id)
        fresh = RewindStore(self.root)
        fresh.bind(sessions[0].session_id)
        fresh.restore_files(fresh.choices(0)[0].id, 0)
        self.assertEqual(self.path.read_bytes(), b"port = 8000\r\n")

    def test_untracked_paths_and_legacy_session(self):
        self.begin()
        self.assertFalse(self.store.tracks(self.root / ".git" / "config"))
        self.assertFalse(self.store.tracks(self.root / ".memory" / "pref.md"))
        self.assertFalse(self.store.tracks(self.root.parent / "outside.txt"))
        with self.assertRaises(ValueError):
            self.store._checked_path(str(self.root / ".." / "outside.txt"))
        legacy = RewindStore(self.root)
        legacy.bind("a" * 32)
        self.assertEqual(legacy.choices(100), [])

    async def test_menu_cancel_and_mode_selection(self):
        point = self.begin()
        self.write("changed")
        self.finish_turn()
        for choices in ([""], ["q"], ["bad"], ["1", ""], ["1", "q"]):
            with patch.object(commands, "PromptSession", return_value=Mock(prompt_async=AsyncMock(side_effect=choices))):
                await commands.cmd_rewind(self.state)
            self.assertEqual(self.path.read_text(), "changed")
        with patch.object(commands, "PromptSession", return_value=Mock(prompt_async=AsyncMock(side_effect=["1", "3"]))):
            await commands.cmd_rewind(self.state)
        self.assertEqual(self.path.read_bytes(), b"port = 8000\r\n")
        self.assertEqual(self.state.next_prompt, "修改端口")

    async def test_rewind_prompt_prefills_next_input(self):
        self.state.next_prompt = "请重新修改端口"
        prompt = AsyncMock(return_value="改成 9090")
        with patch.object(main, "prompt_session", Mock(prompt_async=prompt)), patch.object(main, "print_divider"):
            self.assertEqual(await main.read_user_input(self.state), "改成 9090")
        self.assertEqual(prompt.call_args.kwargs["default"], "请重新修改端口")
        self.assertEqual(self.state.next_prompt, "")


if __name__ == "__main__":
    unittest.main()
