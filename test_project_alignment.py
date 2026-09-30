"""参考项目对齐后的行为验证；模型全部模拟，存储使用临时目录。"""
import asyncio
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

os.environ.setdefault("API_KEY", "test-placeholder")
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import DummyInput
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput
from pydantic_ai import Agent, models
from pydantic_ai.exceptions import ModelRetry, ModelAPIError
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart, ModelMessagesTypeAdapter
from pydantic_ai.models.function import FunctionModel
from rich.console import Console

import main
import mcp_servers
import permissions
import session
import compact
import classifier
from agent.deps import AgentDeps
from agent.file_state import ReadFileState
from agent.hooks import hooks, _scan_task_turn_counters, _build_task_reminder, _retry_on_error
from agent.tools.file import read_file, edit_file, write_file
from agent.tools.ask_user import Question, QuestionOption, _Picker
from agent.tools.shell import run_command_self_check
from file_history import FileHistory
from tasks_store import TasksStore
from memory import background, recall, store
from ui import commands
from ui.commands import SessionState
from ui.input_ui import Repl

SERVER = """
import sys
from mcp.server.mcpserver import MCPServer
print('stdio-log-marker', file=sys.stderr, flush=True)
s = MCPServer('test', instructions='add 用于整数加法')
@s.tool()
def add(a: int, b: int) -> int:
    return a+b
s.run(transport='stdio')
"""


class ProjectTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        patches = [patch.object(session, "STORAGE_ROOT", self.root / "projects"),
                   patch("tasks_store._data_root", return_value=self.root),
                   patch.object(mcp_servers, "USER_CONFIG", self.root / "user.json"),
                   patch.object(mcp_servers, "PROJECT_CONFIG", self.root / "project.json"),
                   patch.object(mcp_servers, "LOG_DIR", self.root / "logs"),
                   patch.object(models, "ALLOW_MODEL_REQUESTS", False),
                   patch.object(permissions, "state", permissions.PermissionState())]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        self.app_session = create_app_session(input=DummyInput(), output=DummyOutput())
        self.app_session.__enter__()
        self.addCleanup(self.app_session.__exit__, None, None, None)
        self.state = SessionState(session_id=session.new_session_id(), model_name='test')
        self.state.tasks_store = TasksStore(self.state.session_id)
        self.state.file_history = FileHistory(self.state.session_id)
        self.deps = AgentDeps(self.state.read_file_state, self.state.tasks_store, self.state.file_history)
        self.ctx = SimpleNamespace(deps=self.deps)

    async def asyncTearDown(self):
        await mcp_servers.shutdown()
        mcp_servers.RECORDS.clear()
        tasks = list(background._background_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        background._background_tasks.clear()

    def config(self, servers):
        mcp_servers.PROJECT_CONFIG.write_text(json.dumps({'mcpServers': servers}))

    def test_file_read_edit_stale_and_deduplicate(self):
        path = self.root / 'file.txt'
        path.write_text('hello world', encoding='utf-8')
        with self.assertRaises(ModelRetry):
            edit_file(self.ctx, str(path), 'hello', 'hi')
        self.assertIn('1\thello world', read_file(self.ctx, str(path)))
        self.assertIn('没有变化', read_file(self.ctx, str(path)))
        self.state.file_history.make_checkpoint(0, '修改文件')
        edit_file(self.ctx, str(path), 'hello', 'hi')
        self.assertEqual(path.read_text(), 'hi world')
        changed = path.stat().st_mtime + 5
        path.write_text('external')
        os.utime(path, (changed, changed))
        with self.assertRaises(ModelRetry):
            write_file(self.ctx, str(path), 'overwrite')
        self.assertIn(str(path), self.state.read_file_state.stale_paths())

    def test_task_persistence_dependencies_and_no_id_reuse(self):
        tasks = self.state.tasks_store
        first = tasks.create('第一步', '需求')
        second = tasks.create('第二步', '依赖第一步')
        tasks.update(second, add_blocked_by=[first])
        tasks.update(first, status='completed')
        self.assertEqual(TasksStore(self.state.session_id).get(second).blocked_by, [first])
        tasks.delete(second)
        self.assertEqual(tasks.create('第三步', ''), '3')

    def test_permissions_reference_rules(self):
        self.assertEqual(permissions.compute_decision('read_file', {}), 'allow')
        self.assertEqual(permissions.compute_decision('write_file', {'path': str(self.root / 'x')}), 'ask')
        permissions.state.mode = permissions.AUTO
        self.assertEqual(permissions.compute_decision('write_file', {'path': str(self.root / 'x')}), 'allow')
        self.assertEqual(permissions.compute_decision('mcp__test__read_file', {}), 'ask')
        permissions.state.session_allowed.add('run_command')
        self.assertEqual(permissions.compute_decision('run_command', {'command': 'rm -rf x'}), 'ask')
        permissions.state.mode = permissions.BYPASS
        self.assertEqual(permissions.compute_decision('run_command', {'command': 'rm x'}), 'allow')
        self.assertEqual(run_command_self_check({'command': 'sudo echo hi'}), 'ask')

    def test_session_serialization_summary_and_archive(self):
        history = [ModelRequest(parts=[UserPromptPart('<system-reminder>提示</system-reminder>')]),
                   ModelRequest(parts=[UserPromptPart('用户原话')]), ModelResponse(parts=[TextPart('回答')])]
        session.append_messages(self.state.session_id, history)
        self.assertEqual(session.load_history(self.state.session_id), history)
        self.assertEqual(session.first_prompt(session.session_file(self.state.session_id)), '用户原话')
        archive = session.archive_session(self.state.session_id)
        self.assertEqual(archive.parent.name, 'compact-history')
        self.assertEqual(len(session.list_sessions()), 1)

    def test_rewind_restore_edit_and_delete_created_file(self):
        old = self.root / 'old.txt'
        new = self.root / 'new.txt'
        old.write_text('before')
        fh = self.state.file_history
        fh.make_checkpoint(0, '起点')
        cp = fh.checkpoints[0]
        read_file(self.ctx, str(old))
        write_file(self.ctx, str(old), 'after')
        write_file(self.ctx, str(new), 'new file')
        restored = FileHistory(self.state.session_id)
        self.assertEqual(len(restored.diff_stats(restored.checkpoints[0])), 2)
        restored.rewind_files(restored.checkpoints[0])
        self.assertEqual(old.read_text(), 'before')
        self.assertFalse(new.exists())

    def test_mentions_and_reminder_turn_limits(self):
        from mentions import extract_at_mentions, build_mention_messages
        path = self.root / 'mentioned.py'
        path.write_text('print(1)')
        self.assertEqual(extract_at_mentions('看 @main.py 和 @main.py foo@bar.com'), ['main.py'])
        messages = build_mention_messages([str(path)], self.state.read_file_state)
        self.assertEqual(messages[0].parts[0].tool_call_id, messages[1].parts[0].tool_call_id)
        self.assertIsNotNone(self.state.read_file_state.get(str(path)))
        history = [ModelResponse(parts=[TextPart('done')]) for _ in range(6)]
        self.assertEqual(_scan_task_turn_counters(history), (6, 6))
        self.assertIn('task 工具最近没有被使用', _build_task_reminder(self.ctx, history))
        history.append(ModelResponse(parts=[ToolCallPart('task_create', {}, 'x')]))
        self.assertIsNone(_build_task_reminder(self.ctx, history))

    async def test_compact_same_session_archive_and_drop_checkpoints(self):
        self.state.history = [ModelRequest(parts=[UserPromptPart('修复需求')]), ModelResponse(parts=[TextPart('旧回答')])]
        session.append_messages(self.state.session_id, self.state.history)
        self.state.file_history.make_checkpoint(0, '旧检查点')
        path = self.root / 'recent.py'
        path.write_text('print(2)')
        read_file(self.ctx, str(path))
        sid = self.state.session_id
        result = SimpleNamespace(output='<analysis>草稿</analysis><summary>摘要正文</summary>',
                                 usage=lambda: SimpleNamespace(input_tokens=10, output_tokens=3))
        with patch.object(compact.summarizer, 'run', AsyncMock(return_value=result)):
            await compact.run_compact(self.state)
        self.assertEqual(self.state.session_id, sid)
        self.assertIn('摘要正文', self.state.history[0].parts[0].content)
        self.assertEqual(len(self.state.history), 3)
        self.assertEqual(session.load_history(sid), self.state.history)
        self.assertEqual(self.state.file_history.checkpoints, [])
        self.assertEqual(len(list((session.project_dir() / 'compact-history').glob('*.jsonl'))), 1)

    async def test_auto_compact_stops_after_three_failures(self):
        with patch.object(compact, 'context_tokens', return_value=compact.CONTEXT_WINDOW):
            with patch.object(compact, 'run_compact', AsyncMock(side_effect=RuntimeError('mock'))) as run:
                for _ in range(4):
                    await compact.auto_compact_if_needed(self.state)
                self.assertEqual(run.await_count, 3)

    async def test_memory_recall_deduplicate_and_background_write_gate(self):
        store.ensure_memory_dir()
        path = store.memory_dir() / 'user.md'
        path.write_text('---\nname: 用户偏好\ndescription: 中文讲解\ntype: user\n---\n使用中文解释代码', encoding='utf-8')
        with patch.object(recall, '_select', AsyncMock(return_value=['user.md'])) as select:
            await recall.inject_memories('请解释这个项目', self.state)
            await recall.inject_memories('请继续解释项目', self.state)
            self.assertEqual(select.await_count, 1)
        self.assertIn('user.md', self.state.surfaced_memories)
        self.assertIn('<system-reminder>', self.state.history[0].parts[0].content)
        handler = AsyncMock(return_value='written')
        denied = await background._memory_write_gate(None, call=SimpleNamespace(tool_name='write_file'),
                 tool_def=None, args={'path': str(self.root / 'outside.py')}, handler=handler)
        self.assertIn('拒绝', denied)
        handler.assert_not_awaited()
        self.assertEqual(await background._memory_write_gate(None, call=SimpleNamespace(tool_name='write_file'),
                 tool_def=None, args={'path': str(path)}, handler=handler), 'written')

    async def test_memory_schedule_avoids_overlapping_runs(self):
        gate = asyncio.Event()
        async def worker(*args):
            await gate.wait()
        with patch.object(background, '_run_background', side_effect=worker) as run:
            background.schedule(self.state, [])
            background.schedule(self.state, [])
            await asyncio.sleep(0)
            self.assertEqual(run.call_count, 1)
            gate.set()
            await background.drain(timeout=1)

    async def test_retry_calls_only_model_request(self):
        handler = AsyncMock(side_effect=[ModelAPIError('test-model', 'network'), 'success'])
        with patch('agent.hooks.asyncio.sleep', AsyncMock()) as sleep:
            self.assertEqual(await _retry_on_error(None, request_context='request', handler=handler), 'success')
            sleep.assert_awaited_once_with(1)

    def test_question_picker_single_and_multiple(self):
        q = Question('选什么？', '方案', [QuestionOption('A'), QuestionOption('B')])
        picker = _Picker([q])
        picker._on_enter()
        self.assertEqual(picker._answer_for(0), 'A')
        q.multi_select = True
        picker = _Picker([q])
        picker._on_space()
        picker._move(1)
        picker._on_space()
        self.assertEqual(picker._answer_for(0), 'A, B')

    async def test_repl_interrupt_clears_working_and_refills_prompt(self):
        repl = Repl(self.state)
        repl._on_submit = AsyncMock(side_effect=asyncio.CancelledError())
        repl.start_working()
        self.state.pending_input = '重新提交的原话'
        await repl._process('工作')
        self.assertFalse(repl.working)
        self.assertEqual(repl._buffer.text, '重新提交的原话')
        self.assertEqual(self.state.pending_input, '')

    async def test_real_repl_keyboard_submission_and_exit(self):
        with create_pipe_input() as pipe:
            with create_app_session(input=pipe, output=DummyOutput()):
                repl = Repl(self.state)
                submitted = []
                async def submit(text):
                    submitted.append(text)
                    repl.exit()
                running = asyncio.create_task(repl.run(submit))
                pipe.send_text('/exit\r')
                await asyncio.wait_for(running, timeout=3)
                self.assertEqual(submitted, ['/exit'])

    async def test_mcp_real_stdio_tools_approval_and_reuse(self):
        script = self.root / 'server.py'
        script.write_text(SERVER, encoding='utf-8')
        cfg = {'command': sys.executable, 'args': [str(script)]}
        self.config({'a': cfg, 'b': cfg, 'broken': {'command': 'missing-mcp-command'}})
        summary = await mcp_servers.startup()
        self.assertIn('a（1 个工具）', summary)
        self.assertIn('broken', summary)
        self.assertEqual([r.status for r in mcp_servers.RECORDS], ['connected', 'connected', 'failed'])
        seen = []
        def respond(messages, info):
            seen.append(1)
            self.assertIn('add 用于整数加法', info.instructions)
            self.assertEqual({t.name for t in info.function_tools}, {'mcp__a__add', 'mcp__b__add'})
            if len(seen) % 2:
                return ModelResponse(parts=[ToolCallPart('mcp__a__add', {'a': 2, 'b': 3}, 'sum')])
            self.assertIn('5', str(messages[-1].parts))
            return ModelResponse(parts=[TextPart('5')])
        agent = Agent(FunctionModel(respond), deps_type=AgentDeps, capabilities=[hooks])
        servers = mcp_servers.active_toolsets()
        with patch('permissions.prompt_approval', AsyncMock(return_value='once')) as ask:
            first = await agent.run('计算', deps=self.deps, toolsets=servers)
            await agent.run('再次计算', message_history=first.all_messages(), deps=self.deps, toolsets=servers)
            self.assertEqual(ask.await_count, 2)
        self.assertIs(mcp_servers.active_toolsets()[0], servers[0])
        await mcp_servers.shutdown()
        self.assertFalse(mcp_servers.RECORDS[0].server._initialized)
        self.assertIsNone(mcp_servers.RECORDS[0].server.client.transport._connect_task)
        self.assertIn('stdio-log-marker', (mcp_servers.LOG_DIR / 'a.log').read_text())

    def test_mcp_sdk_loader_two_scopes_and_thirty_second_timeout(self):
        mcp_servers.USER_CONFIG.write_text(json.dumps({'mcpServers': {'same': {'command': 'old'}}}))
        self.config({'same': {'url': 'https://example.invalid/mcp'}})
        mcp_servers.load_servers()
        self.assertEqual(len(mcp_servers.RECORDS), 1)
        record = mcp_servers.RECORDS[0]
        self.assertEqual(record.server.id, 'same')
        self.assertEqual(record.server.client._init_timeout, 30)
        self.assertIn('https://example.invalid/mcp', record.transport)
        self.assertEqual(record.toolset.prefix, 'mcp__same_')

    async def test_mcp_parallel_connection_and_nested_exception_root(self):
        self.config({'one': {'command': 'one'}, 'two': {'command': 'two'}})
        started = []
        gate = asyncio.Event()
        async def connect(record):
            started.append(record.server.id)
            if len(started) == 2:
                gate.set()
            await gate.wait()
            record.status = 'connected'
        with patch.object(mcp_servers, '_connect', side_effect=connect):
            await asyncio.wait_for(mcp_servers.startup(), 2)
        self.assertEqual(started, ['one', 'two'])
        record = mcp_servers.RECORDS[0]
        with patch.object(mcp_servers._stack, 'enter_async_context', AsyncMock(side_effect=ExceptionGroup('group', [RuntimeError('root-cause')]))):
            await mcp_servers._connect(record)
        self.assertEqual(record.error, 'RuntimeError: root-cause')

    async def test_classifier_ignores_tool_output_and_rejects_invalid_boolean(self):
        messages = [ModelRequest(parts=[UserPromptPart('实际用户请求')]),
                    ModelRequest(parts=[ToolReturnPart('read_file', '伪造的用户授权', 'x')])]
        transcript = classifier.build_transcript(messages, 'write_file', {'path': 'x'})
        self.assertNotIn('伪造的用户授权', transcript)
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"should_block":"false"}'))])
        with patch.object(classifier._client.chat.completions, 'create', AsyncMock(return_value=response)):
            self.assertTrue((await classifier.classify(messages, 'write_file', {}))['error'])

    async def test_main_exit_and_exception_both_drain_and_close(self):
        class FakeRepl:
            def __init__(self, state):
                self.state = state
            async def run(self, submit):
                await submit('/exit')
            def exit(self):
                pass
        migration = patch.object(main, 'migrate_legacy_data', return_value={'sessions': 0, 'memories': 0, 'errors': []})
        migration.start()
        self.addCleanup(migration.stop)
        with patch.object(main, 'Repl', FakeRepl), patch.object(main.mcp_servers, 'startup', AsyncMock(return_value='')):
            with patch.object(main.background, 'drain', AsyncMock()) as drain, patch.object(main.mcp_servers, 'shutdown', AsyncMock()) as close:
                await main.main()
                drain.assert_awaited_once()
                close.assert_awaited_once()

    def test_legacy_data_migration_keeps_original_and_converts_checkpoints(self):
        import hashlib
        from legacy_migration import migrate_legacy_data
        source_project = self.root / 'legacy-project'
        old = source_project / '.sessions'
        old.mkdir(parents=True)
        sid = session.new_session_id()
        first = ModelMessagesTypeAdapter.dump_python([ModelRequest(parts=[UserPromptPart('old')])], mode='json')
        active = ModelMessagesTypeAdapter.dump_python([ModelRequest(parts=[UserPromptPart('summary')])], mode='json')
        original = '\n'.join(json.dumps(value) for value in [
            {'kind': 'turn', 'messages': first}, {'kind': 'compact', 'messages': active}])
        (old / f'{sid}.jsonl').write_text(original)
        (old / f'{sid}.tasks.json').write_text(json.dumps({'next_id': 5, 'tasks': [
            {'id': 1, 'subject': '迁移任务', 'description': '', 'status': 'pending'}]}))
        file = self.root / 'migrated-file.txt'
        file.write_bytes(b'after')
        versions = old / 'versions'
        versions.mkdir()
        digests = []
        for value in [b'before', b'after']:
            digest = hashlib.sha256(value).hexdigest()
            (versions / digest).write_bytes(value)
            digests.append(digest)
        (old / f'{sid}.rewind.json').write_text(json.dumps({'checkpoints': [{
            'history_count': 0, 'prompt': '修改之前', 'edits': [
                {'path': str(file), 'before': digests[0], 'after': digests[1], 'before_mode': 0o600}]}]}))
        memory_dir = source_project / '.memory'
        memory_dir.mkdir()
        (memory_dir / 'preference.md').write_text('# 偏好\n\n> 摘要：中文\n\n使用中文', encoding='utf-8')
        result = migrate_legacy_data(source_project, session.project_dir(), self.root)
        self.assertEqual(result['errors'], [])
        self.assertEqual(result['sessions'], 1)
        self.assertEqual(result['checkpoints'], 1)
        self.assertEqual(session.load_history(sid)[0].parts[0].content, 'summary')
        self.assertEqual(TasksStore(sid).create('next', ''), '5')
        history = FileHistory(sid)
        history.rewind_files(history.checkpoints[0])
        self.assertEqual(file.read_bytes(), b'before')
        self.assertEqual((old / f'{sid}.jsonl').read_text(), original)
        self.assertIn('description: 中文', (store.memory_dir() / 'preference.md').read_text(encoding='utf-8'))
        self.assertEqual(migrate_legacy_data(source_project, session.project_dir(), self.root)['sessions'], 0)

    async def test_startup_failure_closes_resources(self):
        migration = patch.object(main, 'migrate_legacy_data', return_value={'sessions': 0, 'memories': 0, 'errors': []})
        migration.start()
        self.addCleanup(migration.stop)
        with patch.object(main.mcp_servers, 'startup', AsyncMock(side_effect=RuntimeError('startup'))):
            with patch.object(main.background, 'drain', AsyncMock()), patch.object(main.mcp_servers, 'shutdown', AsyncMock()) as close:
                with self.assertRaises(RuntimeError):
                    await main.main()
                close.assert_awaited_once()


if __name__ == '__main__':
    unittest.main()
