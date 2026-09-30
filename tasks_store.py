"""
Task 持久化层：每个 task 是一个独立 JSON，落在 ~/.my-claude-code/tasks/<session_id>/<id>.json 下。
.highwatermark 文件守住「曾经分配过的最大 id」，被删掉的 id 不会被复用。

所有状态在 __init__ 时一次性读入内存，mutation 时同步写盘并更新内存，list()/get() 不再触盘。
TUI 面板每个 spinner tick（~100ms）都会调 list()，没有缓存就是每帧打盘。

单进程使用，mutation 不加文件锁；两个进程同时改同一个 session_id 的目录会有竞态。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterator, Literal


# 落盘状态只承认这 3 个；'deleted' 是 update 时的伪状态信号，会触发删除而不会被写进文件
TaskStatus = Literal["pending", "in_progress", "completed"]


@dataclass
class Task:
    id: str
    subject: str
    description: str
    active_form: str | None = None
    status: TaskStatus = "pending"
    blocks: list[str] = field(default_factory=list)
    blocked_by: list[str] = field(default_factory=list)


_HIGHWATERMARK = ".highwatermark"


def _data_root() -> Path:
    return Path.home() / ".my-claude-code"


class TasksStore:
    """
    按 session_id 隔离的文件式 task 存储。对外暴露 create / list / get / update / delete；每次 mutation 都同步写盘，后续 /resume 时可以恢复历史 task。
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.dir = _data_root() / "tasks" / session_id
        self.dir.mkdir(parents=True, exist_ok=True)
        # __init__ 时把磁盘上所有 task 一次性灌进 _tasks 缓存，后续 list/get/update/delete 都走内存 + 同步写盘
        self._tasks: dict[str, Task] = {}
        for path in self._json_paths():
            try:
                task = Task(**json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, TypeError):
                continue
            self._tasks[task.id] = task
        # 同样的，highwatermark 一次读入，create/delete 直接维护内存里的整数
        self._highwatermark = self._read_highwatermark()

    # ---------- 公共 API ----------

    def create(self, subject: str, description: str, active_form: str | None = None) -> str:
        # id = max(当前所有 id, 历史最大 id) + 1，删过的 id 永不复用
        current_max = max((int(tid) for tid in self._tasks), default=0)
        next_id = max(current_max, self._highwatermark) + 1
        task = Task(
            id=str(next_id),
            subject=subject,
            description=description,
            active_form=active_form,
        )
        self._tasks[task.id] = task
        self._write(task)
        return task.id

    def list(self) -> list[Task]:
        # 按 id 数值升序返回，UI 面板和 task_list 工具的输出顺序一致；走内存 dict 不触盘
        return sorted(self._tasks.values(), key=lambda t: int(t.id))

    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def update(self, task_id: str, **fields) -> Task:
        # add_blocks / add_blocked_by 是追加语义，单独处理；其余字段用 dataclasses.replace 一次性覆盖，享受类型检查
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"task #{task_id} not found")
        add_blocks = fields.pop("add_blocks", None)
        add_blocked_by = fields.pop("add_blocked_by", None)
        new_blocks = list(dict.fromkeys(task.blocks + list(add_blocks))) if add_blocks else task.blocks
        new_blocked_by = list(dict.fromkeys(task.blocked_by + list(add_blocked_by))) if add_blocked_by else task.blocked_by
        updated = replace(task, **fields, blocks=new_blocks, blocked_by=new_blocked_by)
        self._tasks[task_id] = updated
        self._write(updated)
        return updated

    def delete(self, task_id: str) -> None:
        if task_id not in self._tasks:
            raise KeyError(f"task #{task_id} not found")
        # 删前把当前最大 id 提进 highwatermark，下次 create 不会拿到一个"用过"的 id
        current_max = max((int(tid) for tid in self._tasks), default=0)
        if current_max > self._highwatermark:
            self._highwatermark = current_max
            self._write_highwatermark(self._highwatermark)
        del self._tasks[task_id]
        self._task_path(task_id).unlink()

    # ---------- 内部工具 ----------

    def _json_paths(self) -> Iterator[Path]:
        # 抽出来给 __init__ 复用；只挑 *.json，过滤掉 .highwatermark 这类隐藏文件
        for path in self.dir.iterdir():
            if path.suffix == ".json" and not path.name.startswith("."):
                yield path

    def _task_path(self, task_id: str) -> Path:
        return self.dir / f"{task_id}.json"

    def _highwatermark_path(self) -> Path:
        return self.dir / _HIGHWATERMARK

    def _read_highwatermark(self) -> int:
        path = self._highwatermark_path()
        if not path.exists():
            return 0
        try:
            return int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return 0

    def _write_highwatermark(self, value: int) -> None:
        self._highwatermark_path().write_text(str(value), encoding="utf-8")

    def _write(self, task: Task) -> None:
        self._task_path(task.id).write_text(
            json.dumps(asdict(task), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
