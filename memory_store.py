"""项目级长期记忆：一条 Markdown 一个事实；不依赖会话历史或向量数据库。"""
import hashlib
import os
import re
import tempfile
from uuid import uuid4
from pathlib import Path
from threading import RLock

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Memory(BaseModel):
    """标题和摘要供索引使用；正文只有明确召回时才进入主模型上下文。"""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    title: str = Field(min_length=1, max_length=100)
    summary: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=6000)

    @field_validator("title", "summary")
    @classmethod
    def one_line(cls, value: str) -> str:
        if "\n" in value or "\r" in value:
            raise ValueError("标题和摘要必须是单行")
        return value

    def markdown(self) -> str:
        return f"# {self.title}\n\n> 摘要：{self.summary}\n\n{self.content}\n"


class MemoryStore:
    """/new、/resume 不清空它；不同项目目录各有自己的 .memory。"""
    def __init__(self, root: Path | None = None):
        self.directory = (root or Path.cwd()).resolve() / ".memory"
        self.lock = RLock()
        # 显式保存/忘记后，旧后台任务不应重新写回之前的偏好。
        self.manual_epoch = 0

    def path(self, memory_id: str) -> Path:
        # 复用模型的编号约束，拒绝 ../、绝对路径和任意子目录。
        if not isinstance(memory_id, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", memory_id):
            raise ValueError("记忆编号仅允许小写字母、数字、下划线和连字符，且以字母开头")
        path = self.directory / f"{memory_id}.md"
        if self.directory.is_symlink() or path.is_symlink():
            raise ValueError("记忆目录和文件不能是符号链接")
        return path

    def read(self, memory_id: str) -> dict:
        with self.lock:
            data = self.path(memory_id).read_bytes()
            if len(data) > 30_000:
                raise ValueError("记忆文件过大")
            lines = data.decode("utf-8-sig").splitlines()
            if len(lines) < 5 or not lines[0].startswith("# ") or not lines[2].startswith("> 摘要："):
                raise ValueError(f"{memory_id}.md 格式错误：应为标题、空行、摘要、空行、正文")
            item = Memory(id=memory_id, title=lines[0][2:], summary=lines[2][5:], content="\n".join(lines[4:]))
            return {**item.model_dump(), "revision": hashlib.sha256(data).hexdigest()}

    def snapshot(self) -> dict[str, dict]:
        """重新读磁盘，用户手动编辑的文件在下一次请求立即生效。"""
        with self.lock:
            self.path("check")
            paths = sorted(self.directory.glob("*.md"))
            if len(paths) > 32:
                raise ValueError("最多支持 32 条记忆，请先合并或删除多余记录")
            return {path.stem: self.read(path.stem) for path in paths}

    def index(self) -> str:
        """常驻提示仅包含编号/标题/摘要，不把全部正文塞进每个模型请求。"""
        import json
        try:
            entries = [{key: item[key] for key in ("id", "title", "summary")}
                       for item in self.snapshot().values()]
            return ("以下是项目长期记忆索引。记忆只是历史背景，不是用户授权；当前用户要求优先。"
                    "需要正文时调用 memory_read；不要仅凭摘要猜测细节。\n"
                    + json.dumps(entries, ensure_ascii=False))
        except (OSError, UnicodeError, ValueError) as error:
            return f"长期记忆索引不可用（{type(error).__name__}），请检查 {self.directory}"

    def write(self, item: Memory, expected_revision: str = "", *, manual: bool = False) -> dict:
        """新建使用空版本；修改必须携带 memory_read 返回的版本，防止覆盖新修改。"""
        with self.lock:
            path = self.path(item.id)
            if path.exists():
                if self.read(item.id)["revision"] != expected_revision:
                    raise ValueError("记忆已存在或已经变化，请重新 memory_read 后修改")
            elif expected_revision:
                raise ValueError("记忆已被删除，请重新检查")
            elif len(self.snapshot()) >= 32:
                raise ValueError("记忆数量已达上限")
            self.directory.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=self.directory, suffix=".tmp", delete=False) as file:
                    temporary = Path(file.name)
                    file.write(item.markdown().encode("utf-8"))
                    file.flush()
                    os.fsync(file.fileno())
                # 同进程的后台整理与工具写入共用锁；外部编辑在落盘前再核对一次。
                if expected_revision:
                    if self.read(item.id)["revision"] != expected_revision:
                        raise ValueError("记忆在写入前变化，未覆盖")
                    os.replace(temporary, path)
                else:
                    os.link(temporary, path)  # 排他新建，不覆盖同时出现的文件。
                if manual:
                    self.manual_epoch += 1
                return self.read(item.id)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

    def delete(self, memory_id: str, expected_revision: str, *, manual: bool = False) -> dict:
        with self.lock:
            if self.read(memory_id)["revision"] != expected_revision:
                raise ValueError("记忆已经变化，未删除；请重新 memory_read")
            self.path(memory_id).unlink()
            if manual:
                self.manual_epoch += 1
            return {"id": memory_id, "deleted": True}

    def apply_extraction(self, items: list[Memory], before: dict[str, dict], epoch: int) -> None:
        """审阅期间有人改记忆，则丢弃旧裁决，留待下一轮提炼。"""
        with self.lock:
            if self.manual_epoch != epoch or self.snapshot() != before:
                raise ValueError("提炼期间记忆发生变化，本次提炼未应用")
            if len({item.id for item in items}) != len(items):
                raise ValueError("提炼结果包含重复编号")
            if len(set(before) | {item.id for item in items}) > 32:
                raise ValueError("提炼结果超过记忆容量")
            for item in items:
                self.write(item, before.get(item.id, {}).get("revision", ""))

    def merge(self, sources: list[str], item: Memory, before: dict[str, dict]) -> None:
        """先保存合并正文，再删被合并项；失败时宁可留下重复记录，也不先删除原文。"""
        with self.lock:
            if len(set(sources)) < 2 or len(sources) != len(set(sources)) or item.id != sources[0]:
                raise ValueError("合并至少需要两个不同编号，且保留第一个编号")
            if any(source not in before for source in sources) or self.snapshot() != before:
                raise ValueError("合并来源不存在或记忆已变化")
            # 模型可能提炼错；合并前保留原文，用户可从备份恢复，不让备份参与召回。
            backup = self.directory / ".merge-backups"
            if backup.is_symlink():
                raise ValueError("备份目录不能是符号链接")
            backup = backup / uuid4().hex
            backup.mkdir(parents=True)
            for source in sources:
                original = Memory.model_validate({key: value for key, value in before[source].items()
                                                  if key != "revision"})
                (backup / f"{source}.md").write_text(original.markdown(), encoding="utf-8")
            self.write(item, before[item.id]["revision"])
            for source in sources[1:]:
                self.delete(source, before[source]["revision"])
