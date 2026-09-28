"""权限审批离线验证：真实 SDK 工具链、临时文件与模拟终端，不访问模型服务。"""
import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from prompt_toolkit import PromptSession
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import DummyInput, create_pipe_input
from prompt_toolkit.output import DummyOutput
from pydantic_ai import Agent, models
from pydantic_ai.exceptions import SkipToolExecution
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

with patch.dict(os.environ, {"API_KEY": "test-placeholder"}):
    with create_app_session(input=DummyInput(), output=DummyOutput()):
        import main

import permissions
from agent.hooks import _approve_tool, hooks
from agent.tools import read_file, run_command, write_file
from ui import commands
from ui.commands import SessionState


class PermissionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        for target in (permissions, commands):
            console = patch.object(target, "console", Mock())
            console.start()
            self.addCleanup(console.stop)

    async def run_tool(self, tool, args, state=None, approval=(True, False, "")):
        """模拟模型先调用工具、再根据结果回复，验证审批实际位于执行之前。"""
        requests = []

        def respond(messages, info):
            requests.append(messages)
            if len(requests) == 1:
                return ModelResponse(parts=[ToolCallPart(tool.__name__, args, tool_call_id="permission-1")])
            return ModelResponse(parts=[TextPart("收到工具结果")])

        test_agent = Agent(FunctionModel(respond), tools=[tool], capabilities=[hooks])
        with patch.object(models, "ALLOW_MODEL_REQUESTS", False), patch.object(main, "agent", test_agent):
            with patch.object(main, "console", Mock()), patch.object(main, "print_part"):
                mock_approval = AsyncMock(side_effect=approval) if callable(approval) else AsyncMock(return_value=approval)
                with patch.object(permissions, "ask_permission", mock_approval) as ask:
                    result = await main.run_agent("测试工具审批", state or SessionState())
        return result, requests, ask

    def test_permission_modes_and_unknown_tools(self):
        expected = {
            "default": [False, True, True, True],
            "acceptEdits": [False, False, True, True],
            "auto": [False, True, True, True],
            "bypass": [False, False, False, False],
        }
        for mode, decisions in expected.items():
            self.assertEqual([permissions.requires_approval(mode, name) for name in ("read_file", "write_file", "run_command", "new_tool")], decisions)
        with self.assertRaises(ValueError):
            permissions.requires_approval("unknown", "read_file")

    async def test_denied_write_has_no_side_effect_and_is_returned_to_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.txt"
            path.write_text("原内容", encoding="utf-8")
            result, requests, ask = await self.run_tool(write_file, {"path": str(path), "content": "新内容"}, approval=(False, False, "不要覆盖"))
            self.assertEqual(path.read_text(encoding="utf-8"), "原内容")
        returned = [p for m in requests[-1] for p in m.parts if p.part_kind == "tool-return"]
        self.assertIn("[权限拒绝]", returned[-1].content)
        self.assertIn("不要覆盖", returned[-1].content)
        self.assertFalse(any(p.part_kind == "retry-prompt" for m in requests[-1] for p in m.parts))
        self.assertEqual(result.output, "收到工具结果")
        ask.assert_awaited_once()

    async def test_allowed_write_waits_for_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "created.txt"
            observed = []

            async def approve(name, args):
                observed.append(path.exists())
                return True, False, ""

            _, _, ask = await self.run_tool(write_file, {"path": str(path), "content": "你好"}, approval=approve)
            self.assertEqual(observed, [False])
            self.assertEqual(path.read_text(encoding="utf-8"), "你好")
            ask.assert_awaited_once()

    async def test_read_and_accept_edits_skip_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.txt"
            path.write_text("你好", encoding="utf-8")
            _, _, read_ask = await self.run_tool(read_file, {"path": str(path)})
            read_ask.assert_not_awaited()
            state = SessionState(permissions=permissions.PermissionState(mode="acceptEdits"))
            _, _, write_ask = await self.run_tool(write_file, {"path": str(path), "content": "修改"}, state)
            write_ask.assert_not_awaited()
            self.assertEqual(path.read_text(encoding="utf-8"), "修改")

    async def test_commands_require_confirmation_unless_bypass(self):
        with patch("agent.tools.subprocess.run", return_value=SimpleNamespace(stdout="ok", stderr="", returncode=0)) as execute:
            for mode in ("default", "acceptEdits"):
                _, _, ask = await self.run_tool(run_command, {"command": "demo"}, SessionState(permissions=permissions.PermissionState(mode=mode)), approval=(False, False, "不运行"))
                ask.assert_awaited_once()
                execute.assert_not_called()
            _, _, ask = await self.run_tool(run_command, {"command": "demo"}, SessionState(permissions=permissions.PermissionState(mode="bypass")))
            ask.assert_not_awaited()
            execute.assert_called_once()

    async def test_remember_matches_full_arguments_only_for_current_runtime(self):
        state = permissions.PermissionState()
        with patch.object(permissions, "ask_permission", AsyncMock(return_value=(True, True, ""))) as ask:
            await permissions.check_permission(state, "run_command", {"command": "npm test"})
            await permissions.check_permission(state, "run_command", {"command": "npm test"})
            self.assertEqual(ask.await_count, 1)
            await permissions.check_permission(state, "run_command", {"command": "npm test && echo changed"})
            await permissions.check_permission(state, "another_tool", {"command": "npm test"})
            await permissions.check_permission(permissions.PermissionState(), "run_command", {"command": "npm test"})
            self.assertEqual(ask.await_count, 4)

    async def test_once_and_denial_are_not_remembered(self):
        state = permissions.PermissionState()
        with patch.object(permissions, "ask_permission", AsyncMock(side_effect=[(True, False, ""), (False, False, "拒绝"), (True, False, "")])) as ask:
            await permissions.check_permission(state, "run_command", {"command": "demo"})
            with self.assertRaises(SkipToolExecution):
                await permissions.check_permission(state, "run_command", {"command": "demo"})
            await permissions.check_permission(state, "run_command", {"command": "demo"})
        self.assertEqual(ask.await_count, 3)
        self.assertEqual(state.allowed_calls, set())

    async def test_parallel_calls_do_not_overlap_prompts(self):
        state = permissions.PermissionState()
        active = 0
        peak = 0

        async def approve(name, args):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            active -= 1
            return True, True, ""

        with patch.object(permissions, "ask_permission", AsyncMock(side_effect=approve)) as ask:
            await asyncio.gather(*(permissions.check_permission(state, "run_command", {"command": "same"}) for _ in range(3)))
        self.assertEqual(peak, 1)
        self.assertEqual(ask.await_count, 1)

    async def test_prompt_choices_invalid_input_and_cancellation(self):
        cases = [(["y"], (True, False, "")), (["a"], (True, True, "")), (["n 先读文件"], (False, False, "先读文件")), ([""], (False, False, "用户拒绝了本次操作")), (["bad", "n"], (False, False, "用户拒绝了本次操作")), ([EOFError()], (False, False, "用户取消了审批")), ([KeyboardInterrupt()], (False, False, "用户取消了审批"))]
        for inputs, expected in cases:
            with self.subTest(inputs=inputs):
                with patch.object(permissions, "PromptSession", return_value=Mock(prompt_async=AsyncMock(side_effect=inputs))):
                    self.assertEqual(await permissions.ask_permission("write_file", {"path": "[red]你好.txt", "content": "完整内容"}), expected)

    async def test_missing_context_prevents_execution(self):
        with self.assertRaises(SkipToolExecution):
            await _approve_tool(SimpleNamespace(deps=None), call=ToolCallPart("write_file", {}), tool_def=None, args={})

    async def test_shift_tab_changes_mode_without_losing_input(self):
        state = SessionState()
        with create_pipe_input() as pipe:
            with create_app_session(input=pipe, output=DummyOutput()):
                session = PromptSession()
                # 真实终端按键序列：输入前半句，Shift+Tab，再输入后半句及回车。
                pipe.send_text("hello\x1b[Z world\r")
                with patch.object(main, "prompt_session", session), patch.object(main, "print_divider"):
                    self.assertEqual(await main.read_user_input(state), "hello world")
        self.assertEqual(state.permissions.mode, "acceptEdits")
        state.permissions.cycle_mode()
        self.assertEqual(state.permissions.mode, "auto")
        state.permissions.cycle_mode()
        self.assertEqual(state.permissions.mode, "bypass")
        state.permissions.cycle_mode()
        self.assertEqual(state.permissions.mode, "default")

    async def test_new_and_resume_keep_current_runtime_permissions(self):
        state = SessionState(permissions=permissions.PermissionState(mode="acceptEdits"))
        state.permissions.allowed_calls.add(("run_command", "example"))
        original = state.permissions
        with patch.object(commands, "save_session"):
            commands.cmd_new(state)
            saved = SimpleNamespace(session_id="old", history=[], input_tokens=1, output_tokens=2, updated_at=Mock(), title="title")
            with patch.object(commands, "list_sessions", return_value=([saved], [])), patch.object(commands, "load_session", return_value=saved):
                with patch.object(commands, "PromptSession", return_value=Mock(prompt_async=AsyncMock(return_value="1"))):
                    await commands.cmd_resume(state)
        self.assertIs(state.permissions, original)
        self.assertEqual(state.permissions.mode, "acceptEdits")
        self.assertEqual(len(state.permissions.allowed_calls), 1)


if __name__ == "__main__":
    unittest.main()
