"""可靠文件工具验证：使用临时文件和模拟模型，覆盖数据保护及与权限系统的整条链路。"""
import codecs
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
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

with patch.dict(os.environ, {"API_KEY": "test-placeholder"}):
    with create_app_session(input=DummyInput(), output=DummyOutput()):
        import main

from agent import tools
from agent.hooks import hooks
from permissions import PermissionState
from ui import commands
from ui.commands import SessionState


class FileEditTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "example.txt"
        self.ctx = SimpleNamespace(deps=PermissionState())

    def read(self, **kwargs):
        return tools.read_file(self.ctx, str(self.path), **kwargs)

    def edit(self, old, new):
        return tools.edit_file(self.ctx, str(self.path), old, new)

    def write(self, content):
        return tools.write_file(self.ctx, str(self.path), content)

    def test_numbered_paginated_read_and_repeat_dedup(self):
        self.path.write_text("一\n二\n三\n四\n", encoding="utf-8")
        text = self.read(offset=2, limit=2)
        self.assertIn("2 | 二", text)
        self.assertIn("3 | 三", text)
        self.assertNotIn("1 | 一", text)
        self.assertIn("offset=4", text)
        self.assertIn("文件未变化", self.read(offset=2, limit=2))
        self.assertIn("2 | 二", self.read(offset=2, limit=2, force=True))
        self.assertIn("[错误]", self.read(offset=0))
        self.assertIn("[错误]", self.read(limit=0))
        self.assertIn("超出", self.read(offset=99))

    def test_existing_file_requires_read_even_in_bypass(self):
        self.ctx.deps.mode = "bypass"
        self.path.write_text("原内容", encoding="utf-8")
        self.assertIn("先用 read_file", self.write("覆盖"))
        self.assertIn("先用 read_file", self.edit("原内容", "修改"))
        self.assertEqual(self.path.read_text(encoding="utf-8"), "原内容")

    def test_partial_reads_must_cover_whole_file_before_full_write(self):
        self.path.write_text("a\nb\nc\n", encoding="utf-8")
        self.read(limit=1)
        self.assertIn("完整读取", self.write("replacement"))
        self.read(offset=2, limit=2)
        self.assertIn("已写入", self.write("replacement"))
        self.assertEqual(self.path.read_text(encoding="utf-8"), "replacement")

    def test_partial_read_can_only_edit_seen_lines(self):
        self.path.write_text("a\nb\nc\n", encoding="utf-8")
        self.read(limit=1)
        self.assertIn("尚未读取", self.edit("c", "new c"))
        self.assertIn("已编辑", self.edit("a", "new a"))
        self.assertIn("完整读取", self.write("whole file"))
        self.assertEqual(self.path.read_text(encoding="utf-8"), "new a\nb\nc\n")

    def test_mtime_change_blocks_stale_overwrite(self):
        self.path.write_text("old", encoding="utf-8")
        self.read()
        previous = self.path.stat()
        self.path.write_text("user changed", encoding="utf-8")
        os.utime(self.path, ns=(previous.st_atime_ns, previous.st_mtime_ns + 10_000_000))
        self.assertIn("已变化", self.write("agent changed"))
        self.assertIn("已变化", self.edit("old", "new"))
        self.assertEqual(self.path.read_text(encoding="utf-8"), "user changed")
        self.assertIn("user changed", self.read())
        self.assertIn("已编辑", self.edit("user changed", "merged"))

    def test_changed_content_with_same_mtime_is_also_detected(self):
        self.path.write_bytes(b"old")
        self.read()
        previous = self.path.stat()
        self.path.write_bytes(b"new")
        os.utime(self.path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        self.assertIn("已变化", self.write("replacement"))
        self.assertEqual(self.path.read_bytes(), b"new")

    def test_deleted_file_is_not_silently_recreated_from_stale_state(self):
        self.path.write_text("old", encoding="utf-8")
        self.read()
        self.path.unlink()
        self.assertIn("被删除", self.write("replacement"))
        self.assertFalse(self.path.exists())
        self.assertIn("不存在", self.read())
        self.assertIn("已写入", self.write("new file"))

    def test_edit_requires_unique_match_including_overlapping_matches(self):
        for original, old, expected in [("a a", "a", "多处"), ("aaaa", "aaa", "多处"), ("a", "missing", "未找到"), ("a", "", "不能为空"), ("a", "a", "新旧文本相同")]:
            with self.subTest(original=original, old=old):
                self.path.write_text(original, encoding="utf-8")
                self.read()
                new = old if expected == "新旧文本相同" else "new"
                self.assertIn(expected, self.edit(old, new))
                self.assertEqual(self.path.read_text(encoding="utf-8"), original)

    def test_edit_preserves_unrelated_bytes_crlf_and_utf8_bom(self):
        original = codecs.BOM_UTF8 + "第一行\r\n旧值\r\n下一行\r\n末尾\r\n".encode("utf-8")
        self.path.write_bytes(original)
        self.read()
        self.assertIn("已编辑", self.edit("旧值\n下一行", "新值\n下一行"))
        self.assertEqual(self.path.read_bytes(), original.replace("旧值".encode("utf-8"), "新值".encode("utf-8")))
        self.assertIn("已编辑", self.edit("新值", "再次修改"))
        self.assertIn("文件未变化", self.read())
        self.assertIn("再次修改", self.read(force=True))

    def test_new_and_empty_files_can_be_written(self):
        self.assertIn("已写入", self.write("你好"))
        self.assertEqual(self.path.read_bytes(), "你好".encode("utf-8"))
        self.assertIn("已编辑", self.edit("你好", "新内容"))
        self.assertIn("已写入", self.write(""))
        self.assertIn("空文件", self.read(force=True))
        self.assertIn("已写入", self.write("恢复内容"))

    def test_same_file_has_same_state_for_path_aliases(self):
        self.path.write_text("value", encoding="utf-8")
        self.read()
        alias = str(self.path.parent / "." / self.path.name)
        self.assertIn("已编辑", tools.edit_file(self.ctx, alias, "value", "changed"))
        self.assertEqual(len(self.ctx.deps.files.read_file_state), 1)

    def test_failed_atomic_replace_keeps_original_and_cleans_temporary_file(self):
        self.path.write_text("old", encoding="utf-8")
        self.read()
        before = dict(self.ctx.deps.files.read_file_state)
        with patch.object(tools.os, "replace", side_effect=PermissionError("denied")):
            self.assertIn("[错误]", self.edit("old", "new"))
        self.assertEqual(self.path.read_text(encoding="utf-8"), "old")
        self.assertEqual(self.ctx.deps.files.read_file_state, before)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_last_moment_change_before_replace_is_detected(self):
        self.path.write_text("old", encoding="utf-8")
        self.read()
        original_read = tools._read_disk
        calls = 0

        def changed_before_commit(path):
            nonlocal calls
            calls += 1
            if calls == 2:
                path.write_text("user changed before commit", encoding="utf-8")
            return original_read(path)

        with patch.object(tools, "_read_disk", side_effect=changed_before_commit):
            self.assertIn("写入前发生变化", self.edit("old", "new"))
        self.assertEqual(self.path.read_text(encoding="utf-8"), "user changed before commit")
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_new_file_creation_never_overwrites_racing_user_file(self):
        original_link = tools.os.link

        def race(source, destination):
            Path(destination).write_text("user created", encoding="utf-8")
            return original_link(source, destination)

        with patch.object(tools.os, "link", side_effect=race):
            self.assertIn("[错误]", self.write("agent created"))
        self.assertEqual(self.path.read_text(encoding="utf-8"), "user created")
        self.assertEqual(self.ctx.deps.files.read_file_state, {})
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_session_new_discards_file_state_but_keeps_permission_mode(self):
        self.path.write_text("old", encoding="utf-8")
        state = SessionState(permissions=self.ctx.deps)
        self.ctx.deps.mode = "acceptEdits"
        self.read()
        with patch.object(commands, "console", Mock()), patch.object(commands, "save_session"):
            commands.cmd_new(state)
        self.assertEqual(state.permissions.mode, "acceptEdits")
        self.assertEqual(state.permissions.files.read_file_state, {})
        self.assertIn("先用 read_file", self.write("new"))


class FileEditIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_edit_file_uses_auto_classifier_and_human_rejection_prevents_change(self):
        for blocked in (False, True):
            with self.subTest(blocked=blocked), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "example.txt"
                path.write_text("old", encoding="utf-8")
                calls = []

                def respond(history, info):
                    calls.append(history)
                    if len(calls) == 1:
                        return ModelResponse(parts=[ToolCallPart("read_file", {"path": str(path)}, tool_call_id="read")])
                    if len(calls) == 2:
                        return ModelResponse(parts=[ToolCallPart("edit_file", {"path": str(path), "old_string": "old", "new_string": "new"}, tool_call_id="edit")])
                    return ModelResponse(parts=[TextPart("完成")])

                test_agent = Agent(FunctionModel(respond), tools=tools.TOOLS, capabilities=[hooks])
                state = SessionState(permissions=PermissionState(mode="auto"))
                with patch.object(models, "ALLOW_MODEL_REQUESTS", False), patch.object(main, "agent", test_agent):
                    with patch.object(main, "console", Mock()), patch.object(main, "print_part"), patch("permissions.console", Mock()):
                        with patch("permissions.classify", AsyncMock(return_value={"should_block": blocked, "reason": "测试裁决"})) as classify:
                            with patch("permissions.ask_permission", AsyncMock(return_value=(False, False, "拒绝"))) as ask:
                                await main.run_agent("将 old 改为 new", state)
                self.assertEqual(classify.await_count, 1)
                self.assertEqual(classify.call_args.args[1], "edit_file")
                self.assertEqual(classify.call_args.args[2]["old_string"], "old")
                self.assertEqual(ask.await_count, int(blocked))
                self.assertEqual(path.read_text(encoding="utf-8"), "old" if blocked else "new")

    async def test_registered_tools_queue_multiple_edits_and_hide_context_parameter(self):
        self.assertTrue(all(tool.sequential for tool in tools.TOOLS))
        for tool in tools.TOOLS:
            self.assertNotIn("ctx", tool.function_schema.json_schema.get("properties", {}))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.txt"
            path.write_text("value=1\n", encoding="utf-8")
            requests = []

            def respond(history, info):
                requests.append(history)
                if len(requests) == 1:
                    return ModelResponse(parts=[ToolCallPart("read_file", {"path": str(path)}, tool_call_id="read")])
                if len(requests) == 2:
                    return ModelResponse(parts=[
                        ToolCallPart("edit_file", {"path": str(path), "old_string": "value=1", "new_string": "value=2"}, tool_call_id="edit1"),
                        ToolCallPart("edit_file", {"path": str(path), "old_string": "value=2", "new_string": "value=3"}, tool_call_id="edit2"),
                    ])
                return ModelResponse(parts=[TextPart("完成")])

            test_agent = Agent(FunctionModel(respond), tools=tools.TOOLS, capabilities=[hooks])
            state = SessionState(permissions=PermissionState(mode="acceptEdits"))
            with patch.object(models, "ALLOW_MODEL_REQUESTS", False), patch.object(main, "agent", test_agent):
                with patch.object(main, "console", Mock()), patch.object(main, "print_part"), patch("permissions.ask_permission", AsyncMock()) as ask:
                    result = await main.run_agent("将 value 改成 3", state)
            ask.assert_not_awaited()
            self.assertEqual(result.output, "完成")
            self.assertEqual(path.read_text(encoding="utf-8"), "value=3\n")
            edits = [p for m in requests[-1] for p in m.parts if p.part_kind == "tool-return" and p.tool_name == "edit_file"]
            self.assertEqual(len(edits), 2)
            self.assertTrue(all("已编辑" in p.content for p in edits))

    async def test_resume_requires_fresh_file_read(self):
        state = SessionState()
        state.permissions.files.read_file_state["old-file"] = Mock()
        saved = SimpleNamespace(session_id="2" * 32, history=[], input_tokens=0, output_tokens=0, updated_at=Mock(), title="title")
        with patch.object(commands, "console", Mock()), patch.object(commands, "save_session"):
            with patch.object(commands, "list_sessions", return_value=([saved], [])), patch.object(commands, "load_session", return_value=saved):
                with patch.object(commands, "PromptSession", return_value=Mock(prompt_async=AsyncMock(return_value="1"))):
                    await commands.cmd_resume(state)
        self.assertEqual(state.permissions.files.read_file_state, {})


if __name__ == "__main__":
    unittest.main()
