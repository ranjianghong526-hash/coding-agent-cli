"""一次性导入原项目数据；保留原文件，已存在的新会话/记忆不覆盖。"""
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from pydantic_ai.messages import ModelMessagesTypeAdapter

import session
import tasks_store


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _rewind_data(source: Path, sid: str, history: list, destination: Path) -> int:
    metadata = source / f"{sid}.rewind.json"
    if not metadata.exists():
        return 0
    document = json.loads(metadata.read_text(encoding="utf-8"))
    context_id = (history[0].metadata or {}).get("compact_id", "") if history else ""
    versions, latest, checkpoints = {}, {}, []
    directory = destination / "file-history" / sid
    directory.mkdir(parents=True, exist_ok=True)

    def version(path, digest, mode=None):
        if digest is not None:
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("旧版本编号无效")
            data = (source / "versions" / digest).read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError("旧文件备份校验失败")
            name = f"{hashlib.sha256(path.encode()).hexdigest()[:16]}@v{len(versions[path]) + 1}"
            target = directory / name
            target.write_bytes(data)
            if mode is not None:
                os.chmod(target, mode)
        else:
            name = None
        versions[path].append(name)
        latest[path] = len(versions[path])

    for point in document.get("checkpoints", []):
        backups = dict(latest)
        for edit in point.get("edits", []):
            path = edit["path"]
            if path not in versions:
                versions[path] = []
                version(path, edit.get("before"), edit.get("before_mode"))
            if path not in backups:
                backups[path] = latest[path]
            version(path, edit.get("after"))
        if (point.get("selectable", True) and point.get("context_id", "") == context_id
                and point.get("history_count", 0) <= len(history)):
            checkpoints.append({"history_index": point["history_count"], "prompt": point["prompt"],
                                "timestamp": metadata.stat().st_mtime, "backups": backups})
    _write_json(directory / "checkpoints.json", {"checkpoints": checkpoints, "versions": versions})
    return len(checkpoints)


def migrate_legacy_data(project: Path | None = None, destination: Path | None = None,
                        task_root: Path | None = None) -> dict:
    """第一次启动自动调用；迁移失败单独报告，不改旧数据、不影响其他会话。"""
    project = project or Path(__file__).resolve().parent
    destination = destination or session.project_dir()
    task_root = task_root or tasks_store._data_root()
    source = project / ".sessions"
    counts = {"sessions": 0, "tasks": 0, "checkpoints": 0, "memories": 0, "errors": []}
    for old in source.glob("*.jsonl"):
        sid = old.stem
        # 源文件名只用作一个会话 ID，不接收目录分隔符或任意用户路径。
        if not sid.replace("-", "").isalnum() or (destination / old.name).exists():
            continue
        try:
            messages = []
            for line in old.read_text(encoding="utf-8").splitlines():
                value = json.loads(line)
                if "messages" in value:
                    if value.get("kind") == "compact":
                        messages = []
                    messages.extend(value["messages"])
                else:
                    messages.append(value)
            history = ModelMessagesTypeAdapter.validate_python(messages)
            # 所有转换先写临时目录，完成后才发布新的会话文件；旧文件永不覆盖。
            destination.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=destination) as temporary:
                staging = Path(temporary)
                checkpoint_count = _rewind_data(source, sid, history, staging)
                task_file = source / f"{sid}.tasks.json"
                tasks = json.loads(task_file.read_text(encoding="utf-8")) if task_file.exists() else {}
                prepared_tasks = [{**task, "id": str(task["id"]), "active_form": None, "blocks": [], "blocked_by": []}
                                  for task in tasks.get("tasks", [])]
                archive = destination / "compact-history" / f"{sid}-legacy-original.jsonl"
                archive.parent.mkdir(parents=True, exist_ok=True)
                if not archive.exists():
                    shutil.copy2(old, archive)
                staged_history = staging / old.name
                staged_history.write_text("".join(json.dumps(msg, ensure_ascii=False) + "\n"
                                          for msg in ModelMessagesTypeAdapter.dump_python(history, mode="json")), encoding="utf-8")
                new_history_dir = destination / "file-history" / sid
                if (staging / "file-history" / sid).exists() and not new_history_dir.exists():
                    new_history_dir.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(staging / "file-history" / sid, new_history_dir)
                new_task_dir = task_root / "tasks" / sid
                if tasks and not new_task_dir.exists():
                    for task in prepared_tasks:
                        _write_json(new_task_dir / f"{task['id']}.json", task)
                    new_task_dir.mkdir(parents=True, exist_ok=True)
                    (new_task_dir / ".highwatermark").write_text(str(tasks.get("next_id", 1) - 1))
                shutil.copy2(staged_history, destination / old.name)
                counts["sessions"] += 1
                counts["tasks"] += len(prepared_tasks)
                counts["checkpoints"] += checkpoint_count
        except (OSError, ValueError, TypeError, KeyError) as error:
            counts["errors"].append(f"{old.name}（{type(error).__name__}）")
    directory = destination / "memory"
    for old in (project / ".memory").glob("*.md"):
        target = directory / old.name
        if target.exists() or old.is_symlink():
            continue
        try:
            text = old.read_text(encoding="utf-8")
            lines = text.splitlines()
            if len(lines) >= 5 and lines[0].startswith("# ") and lines[2].startswith("> 摘要："):
                text = (f"---\nname: {lines[0][2:]}\ndescription: {lines[2][5:]}\ntype: project\n---\n\n"
                        + "\n".join(lines[4:]))
            directory.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            counts["memories"] += 1
        except (OSError, UnicodeError) as error:
            counts["errors"].append(f"{old.name}（{type(error).__name__}）")
    if counts["memories"] and not (directory / "MEMORY.md").exists():
        (directory / "MEMORY.md").write_text("\n".join(
            f"- [{path.stem}]({path.name}) — 从原项目迁移的记忆" for path in directory.glob("*.md")), encoding="utf-8")
    return counts
