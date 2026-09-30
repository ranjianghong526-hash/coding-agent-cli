"""Skills 三层加载、覆盖规则、动态清单与工具链；不使用真实模型 API。"""
import os
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path
from unittest.mock import AsyncMock, patch

os.environ.setdefault("API_KEY", "test-placeholder")
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ModelResponse, ToolCallPart, TextPart
from pydantic_ai.models.function import FunctionModel

import permissions
import session
import skills
from agent.core import project_context
from agent.deps import AgentDeps
from agent.file_state import ReadFileState
from agent.hooks import hooks
from agent.tools.skills import load_skill
from agent.tools.file import read_file
from agent.tools import TOOLS
from memory import store


class SkillTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.user = self.root / "personal"
        self.project = self.root / ".my-claude-code" / "skills"
        original = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, original)
        for item in [patch.object(skills, "USER_SKILLS_DIR", self.user),
                     patch.object(store, "read_index", return_value=""),
                     patch.object(session, "STORAGE_ROOT", self.root / "history"),
                     patch.object(permissions, "state", permissions.PermissionState())]:
            item.start()
            self.addCleanup(item.stop)

    def write_skill(self, root, name, description="用途说明", body="正文只在加载后出现", header_name=None):
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "SKILL.md"
        path.write_text(f"---\nname: {header_name or name}\ndescription: {description}\n---\n{body}\n", encoding="utf-8")
        return path

    async def test_discovery_metadata_only_precedence_and_depth(self):
        self.write_skill(self.user, "same", "个人说明", "个人正文")
        self.write_skill(self.user, "other")
        selected = self.write_skill(self.project, "same", "项目说明", "正文秘密标记")
        self.write_skill(self.project / "same" / "references", "nested")
        # 发现阶段只用 readline，不使用读全文件的 read_text。
        with patch.object(Path, "read_text", side_effect=AssertionError("发现阶段不能读全文")):
            found = skills.discover_skills()
        self.assertEqual([s.name for s in found], ["other", "same"])
        self.assertEqual(found[1].source, "project")
        self.assertEqual(found[1].path, selected.resolve())
        self.assertEqual({f.name for f in fields(skills.SkillInfo)}, {"name", "description", "path", "source"})
        listing = skills.format_skill_listing(found)
        self.assertIn("项目说明", listing)
        self.assertNotIn("正文秘密标记", listing)
        self.assertEqual(skills.discover_skills(self.root / "missing", self.root / "absent"), [])
        self.assertEqual(skills.format_skill_listing([]), "")

    async def test_validation_quotes_and_malformed_metadata(self):
        self.write_skill(self.project, "quoted", '"用途：审查代码"')
        self.write_skill(self.project, "single", "'用户的 ''review'' 流程'")
        for name, desc, header in [("mismatch", "描述", "another"), ("BadName", "描述", None),
                                    ("double--dash", "描述", None), ("empty", "", None),
                                    ("tagged", "<fake>", None), ("long", "x" * 1025, None),
                                    ("block", ">", None)]:
            self.write_skill(self.project, name, desc, header_name=header)
        broken = self.write_skill(self.project, "broken")
        broken.write_text("---\nname: broken\ndescription: never ends", encoding="utf-8")
        invalid = self.write_skill(self.project, "invalid")
        invalid.write_bytes(b"\xff")
        result = skills.discover_skills()
        self.assertEqual([s.name for s in result], ["quoted", "single"])
        self.assertEqual(result[0].description, "用途：审查代码")
        self.assertEqual(result[1].description, "用户的 'review' 流程")

    async def test_read_current_body_unknown_skill_and_resources_stay_unloaded(self):
        path = self.write_skill(self.project, "review", body="先读 diff，再按需读取 references/database.md。")
        ref = path.parent / "references" / "database.md"
        ref.parent.mkdir()
        ref.write_text("资源秘密标记", encoding="utf-8")
        content = skills.read_skill(" review ")
        self.assertIn(str(path.parent), content)
        self.assertIn("先读 diff", content)
        self.assertNotIn("description:", content)
        self.assertNotIn("资源秘密标记", content)
        self.write_skill(self.project, "review", body="修改后的流程")
        self.assertIn("修改后的流程", skills.read_skill("review"))
        for name in ["missing", "../review", str(path)]:
            with self.assertRaisesRegex(ModelRetry, "未知 skill"):
                skills.read_skill(name)
        self.write_skill(self.project, "review", body="")
        with self.assertRaisesRegex(ModelRetry, "正文为空"):
            skills.read_skill("review")

    async def test_three_layers_and_updated_catalog_in_one_agent_run(self):
        path = self.write_skill(self.project, "review", "初始说明", "正文步骤标记，需要时读取 references/database.md。")
        ref = path.parent / "references" / "database.md"
        ref.parent.mkdir()
        ref.write_text("数据库资源标记", encoding="utf-8")
        calls = 0
        def respond(messages, info):
            nonlocal calls
            calls += 1
            text = str(messages)
            if calls == 1:
                self.assertIn("初始说明", info.instructions)
                self.assertNotIn("正文步骤标记", info.instructions)
                self.assertNotIn("数据库资源标记", text)
                definition = next(tool for tool in info.function_tools if tool.name == "load_skill")
                self.assertEqual(definition.parameters_json_schema["required"], ["name"])
                self.write_skill(self.project, "added", "本轮新技能")
                return ModelResponse(parts=[ToolCallPart("load_skill", {"name": "review"}, "skill")])
            if calls == 2:
                self.assertIn("本轮新技能", info.instructions)
                self.assertIn("正文步骤标记", text)
                self.assertNotIn("数据库资源标记", text)
                return ModelResponse(parts=[ToolCallPart("read_file", {"path": str(ref)}, "reference")])
            self.assertIn("数据库资源标记", text)
            return ModelResponse(parts=[TextPart("已经按需加载三层")])
        agent = Agent(FunctionModel(respond), instructions=project_context, tools=[load_skill, read_file],
                      deps_type=AgentDeps, capabilities=[hooks])
        with patch.object(permissions, "prompt_approval", AsyncMock()) as approval:
            result = await agent.run("审查涉及数据库的改动", deps=AgentDeps(ReadFileState(), None))
        approval.assert_not_awaited()
        self.assertEqual(calls, 3)
        self.assertEqual(result.output, "已经按需加载三层")
        session.append_messages("skills-test", result.new_messages())
        restored = session.load_history("skills-test")
        self.assertIn("正文步骤标记", str(restored))
        self.assertIn("数据库资源标记", str(restored))

    async def test_permissions_and_registration_do_not_grant_script_execution(self):
        self.assertIn(load_skill, TOOLS)
        for mode in permissions.MODES:
            permissions.state.mode = mode
            self.assertEqual(permissions.compute_decision("load_skill", {"name": "review"}), "allow")
        permissions.state.mode = permissions.DEFAULT
        self.assertEqual(permissions.compute_decision("run_command", {"command": "python script.py"}), "ask")
        # 新增技能后，不必重启即可重新生成动态清单。
        self.assertNotIn("新增用途", project_context())
        path = self.write_skill(self.project, "new-skill", "新增用途")
        self.assertIn("新增用途", project_context())
        path.unlink()
        self.assertNotIn("新增用途", project_context())


if __name__ == "__main__":
    unittest.main()
