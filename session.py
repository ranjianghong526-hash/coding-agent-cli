"""
会话持久化：把对话历史写成 jsonl 文件，支持扫描和恢复历史会话。
"""
import json
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_core import to_jsonable_python

# 所有会话记录的根目录
STORAGE_ROOT = Path.home() / ".my-claude-code" / "projects"


def sanitize_path(path: str) -> str:
    """
    把项目绝对路径转码成合法的目录名：非字母数字字符一律换成 -
    """
    return re.sub(r"[^a-zA-Z0-9]", "-", path)


def project_dir() -> Path:
    """
    当前项目（工作目录）对应的会话存储目录。
    """
    return STORAGE_ROOT / sanitize_path(str(Path.cwd()))


def new_session_id() -> str:
    return str(uuid.uuid4())


def session_file(session_id: str) -> Path:
    return project_dir() / f"{session_id}.jsonl"


def append_messages(session_id: str, messages) -> None:
    """
    把本轮新增的消息追加到会话文件末尾，一行一条。
    """
    path = session_file(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(to_jsonable_python(msg), ensure_ascii=False) + "\n")


def rewrite_messages(session_id: str, messages) -> None:
    """
    用内存里的对话历史整体重写会话文件，/rewind 截断对话后用它落盘。
    全量重写以内存为准，不依赖磁盘行数和内存条目一一对应。
    """
    path = session_file(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(to_jsonable_python(msg), ensure_ascii=False) + "\n")


def archive_session(session_id: str) -> Path:
    """
    压缩重写会话文件之前先存档：把当前会话文件拷贝进 compact-history/ 子目录，返回存档路径。
    存档保留了压缩前的完整对话记录，摘要不够用时模型可以回头读它。
    放子目录是为了避开 list_sessions() 的 *.jsonl 扫描，存档不会出现在 /resume 列表里。
    """
    archive_dir = project_dir() / "compact-history"
    archive_dir.mkdir(parents=True, exist_ok=True)
    path = archive_dir / f"{session_id}-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    shutil.copy2(session_file(session_id), path)
    return path


def load_history(session_id: str) -> list:
    """
    读取整个会话文件，把每行 JSON 还原成 SDK 的消息对象列表。
    """
    lines = session_file(session_id).read_text(encoding="utf-8").splitlines()
    return ModelMessagesTypeAdapter.validate_python(
        [json.loads(line) for line in lines]
    )


def first_prompt(path: Path) -> str:
    """
    从文件开头找到首条真正的用户输入，作为这个会话的摘要。
    召回的记忆、@ 引用可能排在它前面，所以要跳过 system-reminder 内容，必要时多读几行。
    """
    with open(path, encoding="utf-8") as f:
        for _ in range(20):
            line = f.readline()
            if not line:
                break
            msg = json.loads(line)
            if (msg.get("metadata") or {}).get("origin") in {"dynamic-reminder", "background-job-notification"}:
                continue
            for part in msg.get("parts", []):
                if part.get("part_kind") != "user-prompt":
                    continue
                content = str(part.get("content", ""))
                if content.startswith(("<system-reminder>", "<task-notification>", "<monitor-event>")):
                    continue
                return content
    return "(空会话)"


def list_sessions() -> list:
    """
    扫描当前项目的所有会话文件，按修改时间从新到旧返回
    (session_id, 修改时间, 首条用户输入) 列表。
    """
    if not project_dir().exists():
        return []
    files = sorted(
        project_dir().glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return [
        (p.stem, datetime.fromtimestamp(p.stat().st_mtime), first_prompt(p))
        for p in files
    ]
