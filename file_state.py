"""会话级文件读取记录：用于先读后改、版本核对与重复读取去重，不写入聊天 JSONL。"""
from dataclasses import dataclass, field
from threading import RLock
from typing import Any


@dataclass(frozen=True)
class FileVersion:
    """mtime 使用纳秒；内容摘要也能发现修改时间被保留的内容变动。"""
    mtime_ns: int
    size: int
    inode: int
    digest: str


@dataclass
class ReadFileState:
    """只记版本和已展示行区间，不另缓存一份可能过期的完整文件内容。"""
    version: FileVersion
    total_lines: int
    ranges: list[tuple[int, int]] = field(default_factory=list)
    # 提醒去重与已读版本分开：发过提醒不等于模型读过变化后的正文。
    notified_version: FileVersion | str | None = None

    def contains(self, start: int, end: int) -> bool:
        return any(left <= start and end <= right for left, right in self.ranges)

    def record(self, start: int, end: int) -> None:
        """合并相邻分页，读完所有页后才算完整读取，可用于整体覆盖。"""
        merged = []
        for left, right in sorted(self.ranges + [(start, end)]):
            if merged and left <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], right))
            else:
                merged.append((left, right))
        self.ranges = merged

    @property
    def fully_read(self) -> bool:
        return self.total_lines == 0 or self.contains(1, self.total_lines)


@dataclass
class FileContext:
    """复用现有 deps 注入；同步工具在线程中执行，因此使用线程锁而非 asyncio 锁。"""
    read_file_state: dict[str, ReadFileState] = field(default_factory=dict)
    lock: Any = field(default_factory=RLock, repr=False)

    def remember(self, path: str, record: ReadFileState) -> None:
        """按最近实际读取/写入顺序排列，重复读老文件也刷新顺序。"""
        with self.lock:
            self.read_file_state.pop(path, None)
            self.read_file_state[path] = record

    def paths(self) -> list[str]:
        with self.lock:
            return list(self.read_file_state)

    def clear(self) -> None:
        """新建或恢复会话必须重新读磁盘，不能用历史消息冒充当前文件快照。"""
        with self.lock:
            self.read_file_state.clear()
