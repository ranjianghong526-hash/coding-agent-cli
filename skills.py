"""渐进式 Skills：发现阶段只读元数据，需要时才读取正文与外部资源。"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai.exceptions import ModelRetry

USER_SKILLS_DIR = Path.home() / ".my-claude-code" / "skills"
FRONTMATTER_MAX_LINES = 128
MAX_NAME_CHARS = 64
MAX_DESCRIPTION_CHARS = 1024
SKILL_NAME_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


@dataclass(frozen=True)
class SkillInfo:
    """只保存目录信息，不缓存正文；工具加载时重新读取当前文件。"""
    name: str
    description: str
    path: Path
    source: str


def _decode_scalar(value: str) -> str:
    """支持单行普通文本、JSON 双引号和 YAML 单引号，不引入 YAML 解析依赖。"""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        try:
            return str(json.loads(value))
        except (json.JSONDecodeError, TypeError):
            return value[1:-1]
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1].replace("''", "'")
    return value


def _read_frontmatter(path: Path) -> dict[str, str]:
    """读到第二个 --- 就返回，正文不参与发现；坏文件不影响其他技能。"""
    try:
        with path.open(encoding="utf-8") as file:
            if file.readline().strip() != "---":
                return {}
            fields = {}
            for _ in range(FRONTMATTER_MAX_LINES):
                line = file.readline()
                if not line:
                    return {}
                if line.strip() == "---":
                    return fields
                if line.lstrip().startswith("#"):
                    continue
                key, sep, value = line.partition(":")
                if sep:
                    fields[key.strip()] = _decode_scalar(value)
    except (OSError, UnicodeError):
        return {}
    return {}


def _is_valid_metadata(fields: dict[str, str], directory_name: str) -> bool:
    name, description = fields.get("name", ""), fields.get("description", "")
    return (name == directory_name and 0 < len(name) <= MAX_NAME_CHARS
            and SKILL_NAME_PATTERN.fullmatch(name) is not None
            and 0 < len(description) <= MAX_DESCRIPTION_CHARS
            and description not in {"|", ">"}
            and not any(char in name + description for char in "<>"))


def _scan_root(root: Path, source: str) -> list[SkillInfo]:
    """只扫描 root 的直接子目录；references/scripts 中的文件不是新技能。"""
    try:
        directories = sorted(path for path in root.iterdir() if path.is_dir())
    except OSError:
        return []
    result = []
    for directory in directories:
        path = directory / "SKILL.md"
        fields = _read_frontmatter(path)
        if _is_valid_metadata(fields, directory.name):
            result.append(SkillInfo(fields["name"], fields["description"], path.resolve(), source))
    return result


def discover_skills(cwd: Path | None = None, user_skills_dir: Path | None = None) -> list[SkillInfo]:
    """个人级先扫描，项目级按名称覆盖，最后稳定排序；每次发现反映当前磁盘。"""
    cwd = Path(cwd) if cwd is not None else Path.cwd()
    user_root = Path(user_skills_dir) if user_skills_dir is not None else USER_SKILLS_DIR
    found = {item.name: item for item in _scan_root(user_root, "user")}
    found.update({item.name: item for item in _scan_root(cwd / ".my-claude-code" / "skills", "project")})
    return [found[name] for name in sorted(found)]


def format_skill_listing(items: list[SkillInfo]) -> str:
    if not items:
        return ""
    return "\n".join([
        "可用 skills（这里只包含名称和描述）：",
        *(f"- {item.name}: {item.description}" for item in items),
        "", "当某个 skill 匹配用户任务时，继续处理前先调用 load_skill 读取它；不要加载无关 skill。",
    ])


def _read_body(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ModelRetry(f"无法读取 skill 文件 {path}：{error}") from error
    if lines and lines[0].strip() == "---":
        for index, line in enumerate(lines[1:], 1):
            if line.strip() == "---":
                body = "\n".join(lines[index + 1:]).strip()
                if body:
                    return body
                break
    raise ModelRetry(f"skill 文件 {path} 的 frontmatter 或正文为空/无效，请修正后重试")


def read_skill(name: str, cwd: Path | None = None, user_skills_dir: Path | None = None) -> str:
    """通过目录查找准确名称，避免把模型输入直接当作任意文件路径使用。"""
    available = {item.name: item for item in discover_skills(cwd, user_skills_dir)}
    skill = available.get(name.strip())
    if skill is None:
        raise ModelRetry(f"未知 skill：{name}。可用 skills：{', '.join(available) or '(none)'}")
    return f"skill {skill.name} 的根目录：{skill.path.parent}\n\n{_read_body(skill.path)}"
