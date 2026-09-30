"""图片输入、图文顺序、SDK 请求映射和历史恢复；不调用真实模型或更改真实剪贴板。"""
import asyncio
import base64
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

os.environ.setdefault("API_KEY", "test-placeholder")
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
try:
    import httpx2 as httpx
except ImportError:
    # 兼容使用 httpx 的旧版 OpenAI SDK，不额外添加项目依赖。
    import httpx
from openai import AsyncOpenAI
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import DummyInput
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from rich.console import Console

import classifier
import images
import main
import mentions
import session
from agent.deps import AgentDeps
from agent.file_state import ReadFileState
from agent.tools.file import read_file
from ui import commands
from ui.commands import SessionState
from ui.input_ui import Repl
from file_history import FileHistory

# 一张完整的 1×1 PNG，包含非 UTF-8 字节，能发现误用文本读取或保存的问题。
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aX1cAAAAASUVORK5CYII=")


class ImageTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.image_path = self.root / "shot.PNG"
        self.image_path.write_bytes(PNG)
        self.image = images.load_image(str(self.image_path))
        self.state = SessionState(session_id="image-test")
        self.state.tasks_store = SimpleNamespace(list=lambda: [])
        self.ctx = SimpleNamespace(deps=AgentDeps(self.state.read_file_state, None))
        for item in [patch.object(session, "STORAGE_ROOT", self.root / "projects"),
                     patch.object(images, "CLIPBOARD_DIR", self.root / "clipboard")]:
            item.start()
            self.addCleanup(item.stop)
        app = create_app_session(input=DummyInput(), output=DummyOutput())
        app.__enter__()
        self.addCleanup(app.__exit__, None, None, None)

    async def test_load_tool_and_read_errors(self):
        self.assertEqual(self.image.data, PNG)
        self.assertEqual(self.image.media_type, "image/png")
        returned = read_file(self.ctx, str(self.image_path), offset=999, limit=1)
        self.assertEqual(returned.data, PNG)
        self.assertIsNone(self.state.read_file_state.get(str(self.image_path)))
        for path in [self.root / "missing.png", self.root / "empty.png"]:
            if path.name == "empty.png":
                path.write_bytes(b"")
            with self.assertRaises(ModelRetry):
                read_file(self.ctx, str(path))
        with self.assertRaises(ValueError):
            images.load_image(str(self.root / "file.txt"))
        text = self.root / "file.txt"
        text.write_text("普通文本", encoding="utf-8")
        self.assertIn("普通文本", read_file(self.ctx, str(text)))

    async def test_order_deleted_repeated_and_invalid_placeholders(self):
        second = BinaryContent(data=b"second", media_type="image/jpeg")
        content = images.build_user_content("前[Image #2]中[Image #1]后[Image #2]", [self.image, second])
        self.assertEqual(content, ["前", second, "中", self.image, "后", second])
        self.assertEqual(images.build_user_content("只有文字", [self.image]), ["只有文字"])
        with self.assertRaises(ValueError):
            images.build_user_content("[Image #3]", [self.image])

    async def test_mentions_mix_clipboard_and_text_files(self):
        text = self.root / "code.py"
        text.write_text("print('hello')", encoding="utf-8")
        original = f"[Image #1]截图， @{self.image_path} 设计， @{text} 代码，再看 @{self.image_path}"
        attachments = [self.image]
        with patch.object(main, "print_part"):
            prompt = main.inject_at_mentions(original, self.state, attachments)
        self.assertEqual(len(attachments), 2)
        self.assertEqual(prompt.count("[Image #2]"), 2)
        self.assertEqual(len(self.state.history), 2)
        self.assertIn("hello", self.state.history[1].parts[0].content)
        content = images.build_user_content(prompt, attachments)
        self.assertEqual(sum(isinstance(item, BinaryContent) for item in content), 3)
        self.assertEqual(len(session.load_history(self.state.session_id)), 2)

    async def test_clipboard_success_missing_tool_and_no_image(self):
        def run(command, **options):
            if command == images._CHECK_COMMANDS["Windows"]:
                return SimpleNamespace(returncode=0, stdout=b"True\r\n")
            # 保存脚本含生成的完整路径，模拟系统工具将图片写到这个位置。
            import re
            path = re.search(r"\$img.Save\('([^']+)'", command[-1]).group(1)
            Path(path).write_bytes(PNG)
            return SimpleNamespace(returncode=0, stdout=b"")
        with patch("images.platform.system", return_value="Windows"), patch("images.subprocess.run", side_effect=run):
            first, second = images.read_clipboard_image(), images.read_clipboard_image()
        self.assertEqual(first.data, PNG)
        self.assertEqual(second.data, PNG)
        self.assertEqual(len(list((self.root / "clipboard").glob("*.png"))), 2)
        for outcome in [SimpleNamespace(returncode=0, stdout=b"False"), FileNotFoundError(), subprocess.TimeoutExpired("probe", 5)]:
            with patch("images.platform.system", return_value="Windows"), \
                 patch("images.subprocess.run", **({"side_effect": outcome} if isinstance(outcome, Exception) else {"return_value": outcome})):
                self.assertIsNone(images.read_clipboard_image())
        with patch("images.platform.system", return_value="Windows"):
            command = images._save_command("C:/a'b/image.png")
            self.assertIn("a''b", command[-1])
            self.assertIn("-STA", command)

    async def test_history_and_terminal_and_classifier_preserve_only_text_authority(self):
        message = ModelRequest(parts=[UserPromptPart(["解释截图", self.image]),
                                      ToolReturnPart("read_file", self.image, "picture")])
        session.append_messages(self.state.session_id, [message])
        restored = session.load_history(self.state.session_id)[0]
        self.assertEqual(restored.parts[0].content[1].data, PNG)
        self.assertEqual(restored.parts[1].content.data, PNG)
        self.assertEqual(session.first_prompt(session.session_file(self.state.session_id)), "解释截图")
        session.rewrite_messages(self.state.session_id, [restored])
        self.assertEqual(session.load_history(self.state.session_id)[0].parts[0].content[1].data, PNG)
        transcript = classifier.build_transcript([message], "read_file", {})
        self.assertIn('"user": "解释截图"', transcript)
        self.assertNotIn("BinaryContent", transcript)
        output = io.StringIO()
        with patch.object(commands, "console", Console(file=output, width=120, color_system=None)):
            commands.print_part(message.parts[0])
            commands.print_part(message.parts[1])
        self.assertIn("image/png", output.getvalue())
        self.assertNotIn("IHDR", output.getvalue())
        self.assertNotIn("BinaryContent", output.getvalue())

    async def test_ui_paste_submit_reset_and_escape_during_paste(self):
        repl = Repl(self.state)
        repl._on_submit = AsyncMock()
        with patch.object(images, "read_clipboard_image", return_value=self.image):
            await repl._paste_image()
        self.assertEqual(repl._buffer.text, "[Image #1]")
        repl._buffer.insert_text("解释")
        with patch.object(repl.app, "create_background_task", side_effect=asyncio.create_task):
            repl._on_enter()
            await repl._task
        repl._on_submit.assert_awaited_once_with("[Image #1]解释", attachments=[self.image])
        self.assertEqual(repl._attachments, [])
        with patch.object(images, "read_clipboard_image", return_value=self.image):
            await repl._paste_image()
        self.assertEqual(repl._buffer.text, "[Image #1]")
        kb = repl._build_key_bindings()
        from prompt_toolkit.keys import Keys
        escape = kb.get_bindings_for_keys((Keys.Escape,))[0]
        escape.handler(SimpleNamespace(app=repl.app))
        self.assertEqual(repl._attachments, [])
        self.assertEqual(repl._buffer.text, "")
        # 模拟剪贴板命令尚未返回时清空草稿，迟到的图片不能重新出现。
        def clipboard_after_clear():
            repl._draft_generation += 1
            return self.image
        with patch.object(images, "read_clipboard_image", side_effect=clipboard_after_clear):
            await repl._paste_image()
        self.assertEqual(repl._attachments, [])

    async def test_restore_multimodal_draft(self):
        text, attachments = images.restore_prompt(["前", self.image, "后"])
        self.assertEqual(images.build_user_content(text, attachments), ["前", self.image, "后"])
        self.state.pending_input, self.state.pending_images = text, attachments
        repl = Repl(self.state)
        repl._on_submit = AsyncMock()
        await repl._process("/rewind")
        self.assertEqual(repl._attachments[0].data, PNG)
        self.assertEqual(repl._buffer.text, "前[Image #1]后")
        self.assertEqual(self.state.pending_images, [])

    async def test_rewind_restores_selected_turn_images_only(self):
        for picked in [0, 1]:
            state = SessionState(session_id=f"rewind-{picked}")
            state.file_history = FileHistory(state.session_id)
            state.file_history.make_checkpoint(0, "普通输入")
            state.history = [ModelRequest(parts=[UserPromptPart("普通输入")]),
                             ModelResponse(parts=[TextPart("答复")])]
            state.file_history.make_checkpoint(2, "[Image #1]看图")
            state.history += [ModelRequest(parts=[UserPromptPart([self.image, "看图"])]),
                              ModelResponse(parts=[TextPart("图像答复")])]
            session.append_messages(state.session_id, state.history)
            with patch.object(commands._ListPicker, "run", AsyncMock(side_effect=[picked, "conversation"])):
                await commands.cmd_rewind(state)
            self.assertEqual(len(state.pending_images), 1 if picked == 0 else 0)
            self.assertEqual(state.pending_input, "[Image #1]看图" if picked == 0 else "普通输入")

    async def test_real_keyboard_paste_and_main_loop_multimodal_input(self):
        with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
            repl = Repl(self.state)
            submitted = []
            async def submit(text, attachments=None):
                submitted.append((text, attachments))
                repl.exit()
            with patch.object(images, "read_clipboard_image", return_value=self.image), \
                 patch("ui.input_ui.platform.system", return_value="Windows"):
                repl.app.key_bindings = repl._build_key_bindings()
                running = asyncio.create_task(repl.run(submit))
                pipe.send_text("\x1bv")  # Alt+V 在终端中编码为 Escape + v。
                async with asyncio.timeout(3):
                    while not repl._attachments:
                        await asyncio.sleep(.01)
                pipe.send_text("看图\r")
                await asyncio.wait_for(running, 3)
        self.assertEqual(submitted[0][0], "[Image #1]看图")
        self.assertEqual(submitted[0][1][0].data, PNG)
        seen = []
        def respond(messages, info):
            seen.extend(messages)
            return ModelResponse(parts=[TextPart("收到图片")])
        with patch.object(main, "agent", Agent(FunctionModel(respond))), \
             patch.object(main.mcp_servers, "active_toolsets", return_value=[]), patch.object(main, "print_part"):
            await main.run_agent_loop(submitted[0][0], self.state, submitted[0][1])
        user = next(p for m in seen for p in m.parts if p.part_kind == "user-prompt")
        self.assertEqual(user.content[0].data, PNG)
        self.assertEqual(user.content[1], "看图")
        self.assertEqual(session.load_history(self.state.session_id)[0].parts[0].content[0].data, PNG)

    async def test_actual_sdk_chat_payload_and_image_tool_result(self):
        captured = []
        def respond(request):
            body = json.loads(request.content)
            captured.append(body)
            if len(captured) == 1:
                message = {"role": "assistant", "content": None, "tool_calls": [{"id": "picture", "type": "function",
                           "function": {"name": "read_file", "arguments": json.dumps({"path": str(self.image_path)})}}]}
                reason = "tool_calls"
            else:
                message, reason = {"role": "assistant", "content": "收到图像"}, "stop"
            return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 0,
                "model": "test-vision", "choices": [{"index": 0, "message": message, "finish_reason": reason}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            client = AsyncOpenAI(api_key="test", base_url="https://test.invalid", http_client=http)
            model = OpenAIChatModel("test-vision", provider=OpenAIProvider(openai_client=client))
            agent = Agent(model, tools=[read_file], deps_type=AgentDeps)
            result = await agent.run([self.image, "这是什么？", self.image, "这个呢？"], deps=self.ctx.deps)
        content = captured[0]["messages"][0]["content"]
        self.assertEqual([item["type"] for item in content], ["image_url", "text", "image_url", "text"])
        uri = content[0]["image_url"]["url"]
        self.assertEqual(base64.b64decode(uri.split(",", 1)[1]), PNG)
        returned_image = [part for message in captured[1]["messages"] if message["role"] == "user"
                          for part in message["content"] if isinstance(part, dict) and part["type"] == "image_url"]
        self.assertGreaterEqual(len(returned_image), 3)
        self.assertEqual(result.output, "收到图像")


if __name__ == "__main__":
    unittest.main()
