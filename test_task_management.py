"""任务管理离线测试：真实 SDK 工具链、独立持久化、失败恢复与最新提醒。"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import DummyInput
from prompt_toolkit.output import DummyOutput
from pydantic_ai import Agent, models
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

with patch.dict(os.environ, {"API_KEY": "test-placeholder"}):
    with create_app_session(input=DummyInput(), output=DummyOutput()):
        import main

import session_store
import permissions
from agent.hooks import hooks
from agent.tools import task_create, task_get, task_list, task_update
from classifier import build_transcript
from context_injection import is_system_reminder
from task_store import TaskStore
from ui import commands
from ui.commands import SessionState


class TaskManagementTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / ".sessions"
        for patcher in (patch.object(session_store, "SESSION_DIR", self.root),
                        patch.object(commands, "console", Mock()),
                        patch.object(main, "console", Mock()),
                        patch.object(main, "print_part", Mock())):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.state = SessionState(model_name="test")
        self.tasks = self.state.permissions.tasks

    def test_crud_roundtrip_and_deleted_ids_not_reused(self):
        first = self.tasks.create("读取代码", "先理解现状")
        self.assertEqual(first["status"], "pending")
        self.tasks.create("实现修改")
        self.tasks.update(1, status="in_progress")
        self.tasks.update(1, status="completed", description="已阅读")
        fresh = TaskStore()
        fresh.bind(self.state.session_id)
        self.assertEqual(fresh.list(), self.tasks.list())
        self.assertEqual(fresh.get(1)["description"], "已阅读")
        fresh.update(2, status="deleted")
        self.assertEqual(fresh.create("运行测试")["id"], 3)
        with self.assertRaises(ValueError):
            fresh.get(2)

    def test_invalid_changes_do_not_modify_memory_or_disk(self):
        self.tasks.create("有效任务")
        path = self.tasks.path(self.state.session_id)
        original = path.read_bytes()
        for operation in (lambda: self.tasks.create(" "),
                          lambda: self.tasks.update(99, status="completed"),
                          lambda: self.tasks.update(1),
                          lambda: self.tasks.update(1, status="unknown"),
                          lambda: self.tasks.update(1, status="deleted", subject="新名")):
            with self.assertRaises(ValueError):
                operation()
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(self.tasks.get(1)["status"], "pending")

    def test_atomic_write_failure_preserves_old_state_and_cleans_temporary(self):
        self.tasks.create("原任务")
        path = self.tasks.path(self.state.session_id)
        before = path.read_bytes()
        with patch("task_store.os.replace", side_effect=PermissionError):
            with self.assertRaises(PermissionError):
                self.tasks.update(1, status="completed")
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self.tasks.get(1)["status"], "pending")
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_corrupt_task_file_does_not_replace_current_binding(self):
        self.tasks.create("当前任务")
        other_id = uuid4().hex
        path = self.tasks.path(other_id)
        path.write_text('{"tasks": "broken"}', encoding="utf-8")
        with self.assertRaises(ValueError):
            self.tasks.bind(other_id)
        self.assertEqual(self.tasks.session_id, self.state.session_id)
        self.assertEqual(self.tasks.get(1)["subject"], "当前任务")
        sessions, bad = session_store.list_sessions()
        self.assertEqual([s.session_id for s in sessions], [self.state.session_id])
        self.assertIn(other_id, str(bad))
        self.assertTrue(path.exists())

    async def test_new_and_resume_isolate_and_restore_tasks_without_execution(self):
        self.tasks.create("旧会话任务")
        old_id = self.state.session_id
        commands.cmd_new(self.state)
        self.assertEqual(self.tasks.list(), [])
        self.assertNotEqual(self.state.session_id, old_id)
        # 即使没有成功聊天轮次，独立保存的计划仍可在 /resume 中发现。
        session = session_store.load_session(old_id)
        self.assertEqual(session.history, [])
        with patch.object(commands, "PromptSession", return_value=Mock(prompt_async=AsyncMock(return_value="1"))):
            await commands.cmd_resume(self.state)
        self.assertEqual(self.state.session_id, old_id)
        self.assertEqual(self.tasks.get(1)["subject"], "旧会话任务")
        self.assertEqual(self.state.history, [])

    async def test_four_tools_and_current_reminders_through_real_sdk(self):
        requests = []

        def respond(messages, info):
            requests.append(messages)
            number = len(requests)
            if number == 1:
                return ModelResponse(parts=[ToolCallPart("task_create", {
                    "subject": "运行测试", "description": "验证功能"}, tool_call_id="t1")])
            reminder = [str(p.content) for m in messages for p in m.parts
                        if p.part_kind == "user-prompt" and "当前会话的任务清单" in str(p.content)][-1]
            self.assertIn('"subject": "运行测试"', reminder)
            if number == 2:
                self.assertIn('"status": "pending"', reminder)
                return ModelResponse(parts=[ToolCallPart("task_update", {
                    "task_id": 1, "status": "in_progress"}, tool_call_id="t2")])
            if number == 3:
                self.assertIn('"status": "in_progress"', reminder)
                return ModelResponse(parts=[ToolCallPart("task_get", {"task_id": 1}, tool_call_id="t3"),
                                            ToolCallPart("task_list", {}, tool_call_id="t4")])
            return ModelResponse(parts=[TextPart("测试尚未完成。")])

        self.state.permissions.mode = "auto"
        agent = Agent(FunctionModel(respond), deps_type=permissions.PermissionState,
                      tools=[task_create, task_get, task_update, task_list], capabilities=[hooks])
        with models.override_allow_model_requests(False), patch.object(main, "agent", agent), \
             patch.object(permissions, "ask_permission", AsyncMock()) as approve, \
             patch.object(permissions, "classify", AsyncMock()) as classify:
            result = await main.run_agent("帮我改代码并验证", self.state)
        approve.assert_not_awaited()
        classify.assert_not_awaited()
        history = result.all_messages()
        returned = [p for m in history for p in m.parts if p.part_kind == "tool-return"]
        self.assertEqual([p.tool_name for p in returned], ["task_create", "task_update", "task_get", "task_list"])
        self.assertEqual(returned[-1].content["tasks"][0]["status"], "in_progress")
        reminders = [m for m in history if is_system_reminder(m)]
        self.assertGreaterEqual(len(reminders), 3)
        transcript = build_transcript(history, "run_command", {"command": "python -m unittest"})
        rows = [json.loads(row) for row in transcript.splitlines()]
        self.assertEqual(sum("user" in row for row in rows), 1)
        self.assertNotIn("当前会话的任务清单", transcript)
        main.apply_result(self.state, result)
        restored = session_store.load_session(self.state.session_id)
        self.assertTrue(any(is_system_reminder(m) for m in restored.history))

    async def test_failed_round_keeps_independent_plan_and_next_request_receives_it(self):
        count = 0

        def failing(messages, info):
            nonlocal count
            count += 1
            if count == 1:
                return ModelResponse(parts=[ToolCallPart("task_create", {"subject": "未完计划"}, tool_call_id="failed")])
            raise RuntimeError("模拟模型服务失败")

        agent = Agent(FunctionModel(failing), tools=[task_create], capabilities=[hooks])
        with models.override_allow_model_requests(False), patch.object(main, "agent", agent):
            with self.assertRaises(RuntimeError):
                await main.run_agent("完成一个多步任务", self.state)
        self.assertEqual(self.state.history, [])
        self.assertEqual(self.tasks.get(1)["status"], "pending")
        self.assertFalse(session_store._session_path(self.state.session_id).exists())
        fresh = SessionState(session_id=self.state.session_id)
        self.assertEqual(fresh.permissions.tasks.list(), self.tasks.list())
        seen = []

        def continuation(messages, info):
            seen.extend(str(p.content) for m in messages for p in m.parts if p.part_kind == "user-prompt")
            return ModelResponse(parts=[TextPart("继续检查未完计划")])

        with models.override_allow_model_requests(False), patch.object(main, "agent", Agent(FunctionModel(continuation), capabilities=[hooks])):
            await main.run_agent("继续", fresh)
        self.assertIn("未完计划", "\n".join(seen))

    async def test_task_permissions_all_modes_and_local_tasks_command(self):
        for mode in permissions.MODES:
            for name in ("task_create", "task_get", "task_update", "task_list"):
                self.assertFalse(permissions.requires_approval(mode, name))
        self.tasks.create("[red]文字不能变成样式[/red]")
        self.assertEqual(await main.handle_command("/tasks", self.state), "continue")
        table = commands.console.print.call_args.args[0]
        self.assertIn("[red]文字不能变成样式[/red]", str(table.columns[2]._cells[0]))


if __name__ == "__main__":
    unittest.main()
