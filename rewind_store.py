"""会话检查点和文件版本：修改前备份字节，不依赖 Git，也不执行历史工具。"""
import hashlib
import os
import stat
import tempfile
from pathlib import Path
from threading import RLock
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from task_store import TaskDocument


class FileEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    before: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    after: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    before_mode: int | None = Field(default=None, ge=0, le=0o7777)


class Checkpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(default_factory=lambda: uuid4().hex, pattern=r"^[a-f0-9]{32}$")
    prompt: str
    history_count: int = Field(ge=0, strict=True)
    tasks: TaskDocument = Field(default_factory=TaskDocument)
    selectable: bool = True
    edits: list[FileEdit] = Field(default_factory=list)


class RewindDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int = Field(default=1, ge=1, le=1)
    root: str
    checkpoints: list[Checkpoint] = Field(default_factory=list)


def _atomic_save(path: Path, data: bytes) -> None:
    """日志/备份先可靠写盘；失败时不能让文件工具继续修改代码。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class RewindStore:
    """按串行 CLI 使用；日志属于会话，内容哈希命名的版本文件按项目共享。"""
    def __init__(self, root: Path | None = None):
        self.root = (root or Path.cwd()).resolve()
        self.session_id = uuid4().hex
        self.document = RewindDocument(root=str(self.root))
        self.active_id: str | None = None
        self.lock = RLock()

    @staticmethod
    def path(session_id: str) -> Path:
        from session_store import _session_path
        return _session_path(session_id).with_suffix(".rewind.json")

    def bind(self, session_id: str) -> None:
        with self.lock:
            if session_id == self.session_id:
                return
            path = self.path(session_id)
            document = RewindDocument.model_validate_json(path.read_bytes()) if path.exists() else RewindDocument(root=str(self.root))
            if Path(document.root) != self.root:
                raise ValueError("检查点属于其他项目路径，不能在当前目录恢复")
            for checkpoint in document.checkpoints:
                for edit in checkpoint.edits:
                    self._checked_path(edit.path)
            self.session_id, self.document, self.active_id = session_id, document, None

    def _checked_path(self, value: str) -> Path:
        path = Path(value)
        if not path.is_absolute() or path.resolve() != path or not path.is_relative_to(self.root):
            raise ValueError("检查点路径越出项目目录或包含符号链接")
        relative = path.relative_to(self.root)
        excluded = {".git", ".sessions", ".memory", ".venv", "venv", "node_modules", ".codex", ".agents"}
        if not relative.parts or any(part.casefold() in excluded for part in relative.parts[:-1]):
            raise ValueError("该目录不参与代码回退")
        return path

    def tracks(self, path: Path) -> bool:
        try:
            self._checked_path(str(path))
            return True
        except ValueError:
            return False

    def _commit(self, document: RewindDocument) -> None:
        _atomic_save(self.path(self.session_id), document.model_dump_json(indent=2).encode("utf-8"))
        self.document = document

    def begin(self, prompt: str, history_count: int, tasks: TaskDocument) -> str:
        with self.lock:
            checkpoint = Checkpoint(prompt=prompt, history_count=history_count, tasks=tasks.model_copy(deep=True))
            document = self.document.model_copy(deep=True)
            document.checkpoints.append(checkpoint)
            self._commit(document)
            self.active_id = checkpoint.id
            return checkpoint.id

    def end(self) -> None:
        self.active_id = None

    def _blob_path(self, digest: str) -> Path:
        from session_store import SESSION_DIR
        # digest 也来自严格校验过的 FileEdit，不接受任意文件名。
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("无效版本编号")
        return SESSION_DIR / "versions" / digest

    def _backup(self, data: bytes | None) -> str | None:
        if data is None:
            return None  # 原来不存在：回退时应该删除本轮新建文件。
        digest = hashlib.sha256(data).hexdigest()
        path = self._blob_path(digest)
        if not path.exists():
            _atomic_save(path, data)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("文件版本备份损坏")
        return digest

    def capture(self, path: Path, before: bytes | None, after: bytes) -> bool:
        """在真实修改前保存旧/新版本及顺序；修改失败也能识别磁盘仍是旧版本。"""
        with self.lock:
            if self.active_id is None or not self.tracks(path):
                return False
            mode = stat.S_IMODE(path.stat().st_mode) if before is not None else None
            edit = FileEdit(path=str(path), before=self._backup(before), after=self._backup(after), before_mode=mode)
            document = self.document.model_copy(deep=True)
            next(point for point in document.checkpoints if point.id == self.active_id).edits.append(edit)
            self._commit(document)
            return True

    def choices(self, history_count: int) -> list[Checkpoint]:
        return [point.model_copy(deep=True) for point in self.document.checkpoints
                if point.selectable and point.history_count <= history_count]

    def fork(self, session_id: str, checkpoint_id: str) -> "RewindStore":
        """保留旧日志；分支中被撤销的消息不可再选，但其文件版本仍可供更早的回退使用。"""
        new = RewindStore(self.root)
        new.bind(session_id)
        document = self.document.model_copy(deep=True)
        selected = False
        for point in document.checkpoints:
            selected = selected or point.id == checkpoint_id
            if selected:
                point.selectable = False
        if not selected:
            raise ValueError("检查点不存在")
        new._commit(document)
        return new

    def plan(self, checkpoint_id: str) -> list[tuple[Path, bytes | None, int | None, bytes | None, int | None]]:
        """先校验所有文件和备份，再改任何一个文件；有外部修改就整次拒绝。"""
        index = next((i for i, point in enumerate(self.document.checkpoints) if point.id == checkpoint_id), None)
        if index is None:
            raise ValueError("检查点不存在")
        edits = [edit for point in self.document.checkpoints[index:] for edit in point.edits]
        first, last = {}, {}
        for edit in edits:
            first.setdefault(edit.path, edit)
            last[edit.path] = edit
        plan = []
        for name, initial in first.items():
            path = self._checked_path(name)
            current = path.read_bytes() if path.exists() else None
            current_digest = hashlib.sha256(current).hexdigest() if current is not None else None
            if current_digest not in (last[name].before, last[name].after):
                raise ValueError(f"{path.name} 有未登记的外部修改，未回退；请先自行保存这些修改")
            target = self._blob_path(initial.before).read_bytes() if initial.before is not None else None
            if target is not None and hashlib.sha256(target).hexdigest() != initial.before:
                raise ValueError("文件版本备份损坏，未回退")
            mode = stat.S_IMODE(path.stat().st_mode) if current is not None else None
            plan.append((path, target, initial.before_mode, current, mode))
        return plan

    @staticmethod
    def _restore(path: Path, target: bytes | None, mode: int | None, expected: bytes | None) -> None:
        from agent.tools import _atomic_write, _read_disk
        current, version = _read_disk(path) if path.exists() else (None, None)
        if current != expected:
            raise ValueError(f"{path.name} 在恢复期间变化，停止回退")
        if target is None:
            if current is not None:
                path.unlink()
        else:
            _atomic_write(path, target, version)
            if mode is not None:
                os.chmod(path, mode)

    def restore_files(self, checkpoint_id: str, history_count: int) -> int:
        with self.lock:
            plan = self.plan(checkpoint_id)
            if not plan:
                return 0
            before = self.document.model_copy(deep=True)
            restore = Checkpoint(prompt="程序执行代码回退", history_count=history_count, selectable=False)
            for path, target, _, current, mode in plan:
                restore.edits.append(FileEdit(path=str(path), before=self._backup(current), after=self._backup(target), before_mode=mode))
            document = before.model_copy(deep=True)
            document.checkpoints.append(restore)
            self._commit(document)  # 先登记恢复本身，下一次 rewind 仍能正确校验当前版本。
            changed = []
            try:
                for entry in plan:
                    changed.append(entry)
                    path, target, mode, current, _ = entry
                    self._restore(path, target, mode, current)
            except Exception as error:
                rollback_failed = False
                for path, target, _, current, old_mode in reversed(changed):
                    try:
                        actual = path.read_bytes() if path.exists() else None
                        if actual == current:
                            if actual is not None and old_mode is not None:
                                os.chmod(path, old_mode)
                            continue
                        self._restore(path, current, old_mode, target)
                    except Exception:
                        rollback_failed = True
                if not rollback_failed:
                    self._commit(before)
                raise OSError("代码回退失败；" + ("部分文件未能恢复，请检查版本备份" if rollback_failed else "已撤销本次恢复操作")) from error
            return len(plan)
