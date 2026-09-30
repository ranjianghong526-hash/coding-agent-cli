"""
文件式长期记忆存储：每条记忆是一个带 frontmatter 的 markdown 文件，MEMORY.md 是它们的扁平索引。
"""
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import session

# MEMORY.md 索引每轮全量注入上下文，必须封顶：超过任一上限就截断
MAX_INDEX_LINES = 200
MAX_INDEX_BYTES = 25_000

# 单条记忆召回注入时的截断上限
MAX_MEMORY_LINES = 200
MAX_MEMORY_BYTES = 4096

# 解析 frontmatter 只看文件头这么多行，不必读完正文
FRONTMATTER_MAX_LINES = 30


def memory_dir() -> Path:
    """
    当前项目的记忆目录，和会话文件放在同一个项目目录下，记忆按项目隔离。
    """
    return session.project_dir() / "memory"


def index_path() -> Path:
    return memory_dir() / "MEMORY.md"


def ensure_memory_dir() -> None:
    # 启动时建好目录，system prompt 里才能告诉模型「目录已存在，不要 mkdir」
    memory_dir().mkdir(parents=True, exist_ok=True)


@dataclass
class MemoryHeader:
    """
    单个记忆文件的元信息，从 frontmatter 解析而来；召回只凭它判断相关性，不必读正文。
    """
    filename: str
    path: Path
    mtime: float
    name: str
    description: str
    type: str


def _parse_frontmatter(path: Path) -> dict:
    """
    解析记忆文件头部 frontmatter 里的 name / description / type 字段。
    """
    try:
        with open(path, encoding="utf-8") as f:
            lines = [f.readline() for _ in range(FRONTMATTER_MAX_LINES)]
    except OSError:
        return {}
    if not lines or lines[0].strip() != "---":
        return {}
    fields = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, sep, value = line.partition(":")
        if sep:
            fields[key.strip()] = value.strip()
    return fields


def scan_memory_files() -> list[MemoryHeader]:
    """
    扫描所有记忆文件（排除索引 MEMORY.md 和空文件），按修改时间从新到旧排序。
    """
    directory = memory_dir()
    if not directory.exists():
        return []
    headers = []
    for path in directory.glob("*.md"):
        if path.name == "MEMORY.md":
            continue
        # 空文件是合并整理时被清空的旧记忆，跳过
        if path.stat().st_size == 0:
            continue
        fields = _parse_frontmatter(path)
        headers.append(MemoryHeader(
            filename=path.name,
            path=path,
            mtime=path.stat().st_mtime,
            name=fields.get("name", path.stem),
            description=fields.get("description", ""),
            type=fields.get("type", ""),
        ))
    headers.sort(key=lambda h: h.mtime, reverse=True)
    return headers


def format_manifest(headers: list[MemoryHeader]) -> str:
    """
    把记忆元信息拼成每条一行的清单，给召回和提炼的 prompt 用。
    """
    lines = []
    for h in headers:
        day = datetime.fromtimestamp(h.mtime).strftime("%Y-%m-%d")
        lines.append(f"- [{h.type}] {h.filename}（{day}）：{h.description}")
    return "\n".join(lines)


def truncate_content(content: str, max_lines: int, max_bytes: int, notice: str) -> str:
    """
    先按行数截断，再按字节截断（回退到最后一个完整行），有删减就在末尾附上提示。
    """
    lines = content.splitlines()
    truncated = False
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated = True
    text = "\n".join(lines)
    if len(text.encode("utf-8")) > max_bytes:
        text = text.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")
        if "\n" in text:
            text = text[: text.rfind("\n")]
        truncated = True
    if truncated:
        text += f"\n\n> {notice}"
    return text


def read_index() -> str | None:
    """
    读取 MEMORY.md 索引，超过行数或字节上限就截断，并附一条提醒模型保持索引精炼的警告。
    """
    path = index_path()
    if not path.is_file():
        return None
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if not content.strip():
        return None
    return truncate_content(
        content, MAX_INDEX_LINES, MAX_INDEX_BYTES,
        "警告：MEMORY.md 超过上限（200 行 / 25KB），只加载了一部分。请保持索引每条一行，把细节挪进记忆文件。",
    )


def age_days(mtime: float) -> int:
    return max(0, int((time.time() - mtime) // 86400))


def age_text(days: int) -> str:
    # 用「今天 / 昨天 / N 天前」这种自然语言描述新鲜度，模型对相对天数比对日期算术敏感得多
    if days == 0:
        return "今天"
    if days == 1:
        return "昨天"
    return f"{days} 天前"


def staleness_text(days: int) -> str:
    """
    超过一天的记忆注入时开头附带的新鲜度警告。
    """
    return (
        f"这条记忆已保存 {days} 天。记忆是写入当时的快照，不是实时状态——"
        "其中关于代码行为、文件位置的描述可能已经过期，先核对当前代码再当作事实使用。"
    )


def is_memory_path(path: str) -> bool:
    """
    路径是否落在记忆目录内，权限放行和后台 agent 的写入闸门共用这个判断。
    """
    if not path:
        return False
    try:
        return Path(path).resolve().is_relative_to(memory_dir().resolve())
    except (OSError, ValueError):
        return False


def has_memory_writes(messages) -> bool:
    """
    这轮消息里是否已有落在记忆目录内的写文件调用；主对话自己记过了，后台提炼就不必再跑。
    """
    for message in messages:
        for part in message.parts:
            if getattr(part, "part_kind", "") != "tool-call":
                continue
            if part.tool_name not in ("write_file", "edit_file"):
                continue
            if is_memory_path(str(part.args_as_dict().get("path", ""))):
                return True
    return False
