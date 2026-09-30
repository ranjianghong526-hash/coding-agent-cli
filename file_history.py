"""
文件检查点：在每条用户输入时给被跟踪的文件记一个检查点，/rewind 就能把代码和对话回退到过去某个时间点。

两个记录时机：track_edit 在工具第一次碰某个文件时，把改动前的内容备份成 v1；
make_checkpoint 在每条用户输入时执行，给内容有变化的跟踪文件存一个新版本。

回退到某个检查点时：检查点里记过的文件恢复到对应版本；检查点里没记的跟踪文件
回退到它的 v1（被碰过之前的状态）；v1 为 None 表示「当时还不存在」，回退时删除。

备份文件放在会话文件旁边的 file-history/<session_id>/ 目录；检查点元数据持久化在 checkpoints.json，
所以 /resume 恢复会话后，过去的检查点照样能用。run_command 改动的文件不在跟踪范围内。
"""
import difflib
import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import session


@dataclass
class Checkpoint:
    """
    一个检查点，在用户输入进入对话历史之前建立。
    """
    # 检查点时刻对话历史的长度，回退对话就是截断到这个下标
    history_index: int
    # 用户输入原文，回退后回填输入框；列表展示时再截断
    prompt: str
    timestamp: float
    # path -> 检查点时刻该文件的版本号
    backups: dict = field(default_factory=dict)


@dataclass
class FileChange:
    """
    一个文件的改动：路径、发生了什么、行数增减。
    """
    path: str
    # 回退计划里取 restore（恢复到目标版本）/ delete（目标版本不存在、删除）；单轮改动统计里取 create / edit / delete
    action: str
    insertions: int
    deletions: int


def _count_diff(current, target) -> tuple:
    """
    统计从 current 回退到 target 会带来的行数增减。
    """
    cur = (current or b"").decode("utf-8", "replace").splitlines()
    tgt = (target or b"").decode("utf-8", "replace").splitlines()
    insertions = deletions = 0
    matcher = difflib.SequenceMatcher(None, cur, tgt, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            deletions += i2 - i1
        if tag in ("replace", "insert"):
            insertions += j2 - j1
    return insertions, deletions


class FileHistory:
    """
    会话级的文件检查点存储：版本备份放磁盘，检查点列表存 checkpoints.json。
    """

    def __init__(self, session_id: str):
        self.dir = session.project_dir() / "file-history" / session_id
        # 检查点列表，从旧到新
        self.checkpoints: list[Checkpoint] = []
        # path -> 各版本的备份文件名列表，下标 i 对应版本 i+1；None 表示该版本文件不存在
        self.versions: dict[str, list] = {}
        self._load()

    # —— 记录 ——

    def track_edit(self, path: str) -> None:
        """
        工具写文件之前调用：第一次碰这个文件时，把改动前的内容备份成 v1。
        """
        path = os.path.abspath(path)
        if path in self.versions:
            return
        self.versions[path] = [self._backup(path, 1)]
        self._save()

    def make_checkpoint(self, history_index: int, prompt: str) -> None:
        """
        在用户输入时建检查点：给内容有变化的跟踪文件各存一个新版本。
        没有跟踪文件时也照样记检查点，回退对话要用它。
        """
        backups = {}
        for path, names in self.versions.items():
            latest = len(names)
            if self._current(path) != self._read_version(path, latest):
                names.append(self._backup(path, latest + 1))
            backups[path] = len(names)
        self.checkpoints.append(
            Checkpoint(history_index, prompt, time.time(), backups)
        )
        self._save()

    # —— 回退 ——

    def diff_stats(self, cp: Checkpoint) -> list:
        """
        只读的回退执行计划：每个会被动到的文件一条 FileChange，内容没变的文件不进清单。
        这是从当前磁盘状态回到检查点的累计差异。
        """
        plan = []
        for path in self.versions:
            target = self._content_at(cp, path)
            current = self._current(path)
            if target == current:
                continue
            action = "delete" if target is None else "restore"
            insertions, deletions = _count_diff(current, target)
            plan.append(FileChange(path, action, insertions, deletions))
        return plan

    def turn_stats(self, cp: Checkpoint) -> list:
        """
        该检查点开启的那一轮对话产生的文件改动：和下一个检查点比较，
        最新的检查点则和当前磁盘状态比较。
        """
        index = self.checkpoints.index(cp)
        nxt = self.checkpoints[index + 1] if index + 1 < len(self.checkpoints) else None
        changes = []
        for path in self.versions:
            before = self._content_at(cp, path)
            after = self._content_at(nxt, path) if nxt else self._current(path)
            if before == after:
                continue
            if before is None:
                action = "create"
            elif after is None:
                action = "delete"
            else:
                action = "edit"
            insertions, deletions = _count_diff(before, after)
            changes.append(FileChange(path, action, insertions, deletions))
        return changes

    def rewind_files(self, cp: Checkpoint) -> list:
        """
        把执行计划应用到磁盘，返回实际执行的 FileChange 清单。
        """
        plan = self.diff_stats(cp)
        for change in plan:
            if change.action == "delete":
                os.remove(change.path)
            else:
                # copy2 连文件权限一起从备份带回来
                version = self._target_version(cp, change.path)
                name = self.versions[change.path][version - 1]
                shutil.copy2(self.dir / name, change.path)
        return plan

    def drop_from(self, cp: Checkpoint) -> None:
        """
        对话回退之后，从这个检查点起的所有检查点都指向已不存在的消息，整段丢弃。
        """
        self.checkpoints = self.checkpoints[: self.checkpoints.index(cp)]
        self._save()

    # —— 内部辅助 ——

    def _target_version(self, cp: Checkpoint, path: str) -> int:
        # 检查点里没记过这个文件，说明它在检查点之后才被首次改动，回退到 v1
        return cp.backups.get(path, 1)

    def _content_at(self, cp: Checkpoint, path: str):
        # 某文件在检查点时刻的内容，None 表示当时不存在
        return self._read_version(path, self._target_version(cp, path))

    def _backup(self, path: str, version: int):
        """
        把文件当前内容拷进备份目录，返回备份文件名；文件不存在时返回 None。
        """
        if not os.path.exists(path):
            return None
        digest = hashlib.sha256(path.encode()).hexdigest()[:16]
        name = f"{digest}@v{version}"
        self.dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, self.dir / name)
        return name

    def _read_version(self, path: str, version: int):
        # 读某个版本的备份内容；统一用 bytes，非文本文件也不出错
        name = self.versions[path][version - 1]
        return None if name is None else (self.dir / name).read_bytes()

    def _current(self, path: str):
        try:
            return Path(path).read_bytes()
        except FileNotFoundError:
            return None

    # —— 持久化 ——

    def _save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        data = {
            "checkpoints": [asdict(s) for s in self.checkpoints],
            "versions": self.versions,
        }
        (self.dir / "checkpoints.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _load(self) -> None:
        # 恢复会话时从 checkpoints.json 重建检查点；新会话没有这个文件，从空白开始
        path = self.dir / "checkpoints.json"
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        self.checkpoints = [Checkpoint(**s) for s in data["checkpoints"]]
        self.versions = data["versions"]
