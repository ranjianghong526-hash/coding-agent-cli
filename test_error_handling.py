"""错误处理的离线回归测试：模拟 HTTP、工具故障和继续输入，不调用真实模型。"""
import asyncio
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from openai import AsyncOpenAI
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import DummyInput
from prompt_toolkit.output import DummyOutput
from pydantic_ai import Agent, models
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

# 不同 OpenAI SDK 版本使用不同的 HTTPX 包名称，仅用于模拟传输。
try:
    import httpx2 as httpx
except ImportError:
    import httpx

with patch.dict(os.environ, {"API_KEY": "test-placeholder"}):
    with create_app_session(input=DummyInput(), output=DummyOutput()):
        import main

from agent.core import client as configured_client
from agent.hooks import ApiCall, api_call_log, hooks
from agent.tools import read_file, run_command, write_file
from ui import commands
from ui.commands import SessionState
from permissions import PermissionState


class ToolErrorTests(unittest.TestCase):
    def setUp(self):
        self.ctx = SimpleNamespace(deps=PermissionState())

    def test_known_file_errors_return_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertIn("不存在", read_file(self.ctx, str(root / "missing.txt")))
            self.assertIn("[错误]", read_file(self.ctx, str(root)))
            invalid = root / "invalid.txt"
            invalid.write_bytes(b"\xff")
            self.assertIn("[错误]", read_file(self.ctx, str(invalid)))
            self.assertIn("[错误]", write_file(self.ctx, str(root / "missing" / "file.txt"), "hello"))
            good = root / "good.txt"
            self.assertIn("已写入", write_file(self.ctx, str(good), "你好"))
            self.assertIn("你好", read_file(self.ctx, str(good), force=True))

    def test_permissions_and_command_errors_are_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.txt"
            path.write_text("original", encoding="utf-8")
            with patch("builtins.open", side_effect=PermissionError("denied")):
                self.assertIn("[错误]", read_file(self.ctx, str(path)))
                self.assertIn("[错误]", write_file(self.ctx, str(path), "hello"))
            self.assertEqual(path.read_text(encoding="utf-8"), "original")
        for error in [OSError("cannot start"), subprocess.TimeoutExpired("demo", 10)]:
            with self.subTest(error=type(error).__name__):
                with patch("agent.tools.subprocess.run", side_effect=error):
                    self.assertIn("[错误]", run_command("demo"))


class AgentErrorTests(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_tool_error_is_sent_back_to_model(self):
        displayed = []
        requests = []

        async def broken_tool() -> str:
            """模拟工具的未知异常。"""
            raise ValueError("sensitive-detail")

        def respond(messages, info):
            requests.append(messages)
            if len(requests) == 1:
                return ModelResponse(parts=[ToolCallPart("broken_tool", {}, tool_call_id="broken-1")])
            retry_parts = [p for m in messages for p in m.parts if p.part_kind == "retry-prompt"]
            self.assertTrue(retry_parts)
            self.assertIn("ValueError", str(retry_parts[-1].content))
            self.assertNotIn("sensitive-detail", str(retry_parts[-1].content))
            return ModelResponse(parts=[TextPart("工具失败，停止操作")])

        test_agent = Agent(FunctionModel(respond), tools=[broken_tool], capabilities=[hooks], retries=2)
        with patch.object(models, "ALLOW_MODEL_REQUESTS", False):
            with patch.object(main, "agent", test_agent), patch.object(main, "console", Mock()):
                with patch.object(main, "print_part", side_effect=displayed.append):
                    result = await main.run_agent("测试", SessionState(permissions=PermissionState(mode="bypass")))
        self.assertEqual(result.output, "工具失败，停止操作")
        self.assertIn("retry-prompt", [p.part_kind for p in displayed])

    async def test_tool_retries_are_bounded(self):
        executions = []

        async def broken_tool() -> str:
            """总是失败的工具。"""
            executions.append(1)
            raise RuntimeError("broken")

        def respond(messages, info):
            return ModelResponse(parts=[ToolCallPart("broken_tool", {}, tool_call_id=f"call-{len(executions)}")])

        test_agent = Agent(FunctionModel(respond), tools=[broken_tool], capabilities=[hooks], retries=2)
        with patch.object(models, "ALLOW_MODEL_REQUESTS", False):
            with patch.object(main, "agent", test_agent), patch.object(main, "console", Mock()):
                with patch.object(main, "print_part"):
                    with self.assertRaises(UnexpectedModelBehavior):
                        await main.run_agent("测试", SessionState(permissions=PermissionState(mode="bypass")))
        self.assertEqual(len(executions), 3)

    async def test_main_continues_after_failure_and_keeps_previous_history(self):
        state = SessionState(history=["previous history"], input_tokens=7)
        error = ModelHTTPError(status_code=401, model_name="test", body="sensitive-detail")
        result = Mock()
        result.all_messages.return_value = ["next successful history"]
        result.usage = SimpleNamespace(input_tokens=2, output_tokens=3)

        async def execute(user_input, current_state):
            self.assertEqual(current_state.history, ["previous history"])
            if user_input == "fail":
                # 模拟本轮已有一次成功模型调用、下一次请求失败。
                api_call_log.append(ApiCall("test", 1, None, [], input_tokens=10, output_tokens=5))
                raise error
            self.assertEqual(current_state.input_tokens, 17)
            self.assertEqual(len(current_state.last_api_calls), 1)
            return result

        with patch.object(main, "SessionState", return_value=state):
            with patch.object(main, "read_user_input", AsyncMock(side_effect=["fail", "/status", "succeed", "/exit"])):
                with patch.object(main, "run_agent", AsyncMock(side_effect=execute)) as run:
                    with patch.object(main, "print_welcome_banner"), patch.object(main, "console", Mock()) as console:
                        with patch.object(commands, "console", Mock()), patch.object(main, "save_session"):
                            await main.main()
        self.assertEqual(run.await_count, 2)
        self.assertEqual(state.history, ["next successful history"])
        self.assertEqual(state.input_tokens, 19)
        shown = str(console.print.call_args_list)
        self.assertIn("API Key", shown)
        self.assertNotIn("sensitive-detail", shown)

    async def test_empty_slash_and_user_cancellation(self):
        with patch.object(main, "console", Mock()):
            self.assertEqual(await main.handle_command("/", SessionState()), "continue")
            with patch.object(main, "read_user_input", AsyncMock(return_value="run")):
                with patch.object(main, "run_agent", AsyncMock(side_effect=asyncio.CancelledError)) as run:
                    with patch.object(main, "print_welcome_banner"):
                        await main.main()
                    self.assertEqual(run.await_count, 1)

    async def test_command_error_does_not_count_previous_tokens_again(self):
        state = SessionState(input_tokens=7)
        api_call_log[:] = [ApiCall("test", 1, None, [], input_tokens=10)]
        failing = commands.Command("bad", "模拟错误", Mock(side_effect=ValueError("broken")))
        with patch.object(main, "SessionState", return_value=state):
            with patch.dict(main.COMMANDS, {"bad": failing}):
                with patch.object(main, "read_user_input", AsyncMock(side_effect=["/bad", "/exit"])):
                    with patch.object(main, "console", Mock()), patch.object(commands, "console", Mock()):
                        with patch.object(main, "print_welcome_banner"):
                            await main.main()
        self.assertEqual(state.input_tokens, 7)

    async def test_api_transient_errors_retry_only_request(self):
        calls = []

        def handle(request):
            calls.append(request)
            status = [429, 503, 200][len(calls) - 1]
            if status != 200:
                return httpx.Response(status, json={"error": {"message": "temporary"}})
            return httpx.Response(200, json={
                "id": "mock-response", "object": "chat.completion", "created": 0,
                "model": "deepseek-flash",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "recovered"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            })

        async with AsyncOpenAI(
            api_key="test-placeholder", base_url="https://mock.invalid",
            max_retries=configured_client.max_retries, timeout=configured_client.timeout,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
        ) as client:
            # 测试不实际等待退避，但由真实 SDK 执行重试和响应解析。
            with patch.object(client, "_calculate_retry_timeout", return_value=0):
                test_agent = Agent(OpenAIChatModel("deepseek-flash", provider=DeepSeekProvider(openai_client=client)), capabilities=[hooks])
                with patch.object(main, "agent", test_agent), patch.object(main, "console", Mock()), patch.object(main, "print_part"):
                    result = await main.run_agent("测试", SessionState())
        self.assertEqual(result.output, "recovered")
        self.assertEqual(len(calls), 3)

    async def test_api_authentication_error_is_not_retried_and_is_logged(self):
        calls = []

        def handle(request):
            calls.append(request)
            return httpx.Response(401, json={"error": {"message": "sensitive-detail"}})

        api_call_log.clear()
        async with AsyncOpenAI(
            api_key="test-placeholder", base_url="https://mock.invalid",
            max_retries=configured_client.max_retries,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
        ) as client:
            test_agent = Agent(OpenAIChatModel("deepseek-flash", provider=DeepSeekProvider(openai_client=client)), capabilities=[hooks])
            with patch.object(main, "agent", test_agent), patch.object(main, "console", Mock()):
                with self.assertRaises(ModelHTTPError):
                    await main.run_agent("测试", SessionState())
        self.assertEqual(len(calls), 1)
        self.assertEqual(api_call_log[-1].finish_reason, "error: 401")


if __name__ == "__main__":
    unittest.main()
