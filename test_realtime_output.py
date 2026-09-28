"""离线验证逐步输出，不请求真实模型、不读写用户文件。

运行：.venv/Scripts/python.exe -m unittest -v test_realtime_output
"""
import os
import unittest
from unittest.mock import AsyncMock, Mock, patch

from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import DummyInput
from prompt_toolkit.output import DummyOutput
from pydantic_ai import Agent, models
from pydantic_ai.messages import ModelResponse, TextPart, ThinkingPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

# 模拟终端避免 Windows 在非交互测试环境中寻找真实控制台。
# 导入时只使用测试占位密钥，不读取用户的真实密钥去发起模型请求。
with patch.dict(os.environ, {"API_KEY": "test-placeholder"}):
    with create_app_session(input=DummyInput(), output=DummyOutput()):
        import main

from agent.hooks import api_call_log, hooks
from ui import commands
from ui.commands import SessionState


class RealtimeOutputTests(unittest.IsolatedAsyncioTestCase):
    async def test_steps_are_visible_before_completion_and_not_repeated(self):
        """验证工具执行前可见调用，下一次模型请求前可见工具结果，收尾不重复输出。"""
        displayed = []
        requests = []
        tool_calls = []

        async def read_demo() -> str:
            """返回固定测试内容。"""
            self.assertIn("tool-call", [part.part_kind for part in displayed])
            tool_calls.append("read_demo")
            return "demo contents"

        def respond(messages, info):
            requests.append(messages)
            if len(requests) == 1:
                return ModelResponse(
                    parts=[
                        ThinkingPart("先读取测试内容"),
                        TextPart("准备读取"),
                        ToolCallPart("read_demo", {}, tool_call_id="demo-1"),
                    ],
                    usage=RequestUsage(input_tokens=10, output_tokens=5),
                    finish_reason="tool_call",
                )
            self.assertIn("tool-return", [part.part_kind for part in displayed])
            return ModelResponse(
                parts=[TextPart("读取完成")],
                usage=RequestUsage(input_tokens=20, output_tokens=8),
                finish_reason="stop",
            )

        test_agent = Agent(FunctionModel(respond), tools=[read_demo], capabilities=[hooks])
        state = SessionState(model_name="test-model")
        api_call_log.clear()
        with patch.object(models, "ALLOW_MODEL_REQUESTS", False):
            with patch.object(main, "agent", test_agent), patch.object(main, "console", Mock()):
                with patch.object(main, "print_part", side_effect=displayed.append):
                    result = await main.run_agent("读取测试内容", state)
                    self.assertEqual(result.output, "读取完成")
                    self.assertEqual(tool_calls, ["read_demo"])
                    self.assertEqual(
                        [part.part_kind for part in displayed],
                        ["thinking", "text", "tool-call", "tool-return", "text"],
                    )
                    main.apply_result(state, result)
                    self.assertEqual(len(displayed), 5)
                    self.assertEqual(state.input_tokens, 30)
                    self.assertEqual(state.output_tokens, 13)
                    self.assertEqual(len(state.last_api_calls), 2)
                    self.assertEqual(state.last_api_calls[0].parts_kinds, ["thinking", "text", "tool-call"])
                    self.assertEqual(state.last_api_calls[1].output_tokens, 8)
                    previous_history = list(state.history)
                    api_call_log.clear()
                    self.assertEqual(len(state.last_api_calls), 2)
                    second = await main.run_agent("继续解释", state)
                    self.assertEqual(requests[-1][:len(previous_history)], previous_history)
                    main.apply_result(state, second)
                    self.assertEqual(state.input_tokens, 50)
                    self.assertEqual(state.output_tokens, 21)
                    self.assertEqual(len(state.last_api_calls), 1)

    async def test_input_commands_and_exit_do_not_start_agent(self):
        """空输入跳过，/status 在本地处理，/exit 正常退出异步主循环。"""
        with patch.object(main, "prompt_session", Mock(prompt_async=AsyncMock(side_effect=["  ", "/status", "/exit"]))):
            with patch.object(main, "run_agent", AsyncMock()) as run:
                with patch.object(main, "print_divider"), patch.object(main, "print_welcome_banner"):
                    with patch.object(main, "console", Mock()), patch.object(commands, "console", Mock()):
                        await main.main()
                run.assert_not_awaited()

    async def test_eof_returns_exit_signal(self):
        with patch.object(main, "prompt_session", Mock(prompt_async=AsyncMock(side_effect=EOFError))):
            with patch.object(main, "print_divider"):
                self.assertIsNone(await main.read_user_input())


if __name__ == "__main__":
    unittest.main()
