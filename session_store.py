"""本地会话存储：一个会话一个 JSONL 文件，每行保存一轮新增消息和累计用量。

消息使用 SDK 的序列化适配器，保留工具调用 ID、工具结果、思考片段等结构。
记录位于项目根目录 .sessions/，不同项目自然分开，不保存 API Key 或 HTTP 日志。
"""
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from uuid import UUID, uuid4

from pydantic import AwareDatetime, BaseModel, Field
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

if TYPE_CHECKING:
    # 只供类型检查使用，运行时不导入 UI，避免 commands -> 存储 -> commands 循环。
    from ui.commands import SessionState

SESSION_DIR = Path(__file__).resolve().parent / ".sessions"


class StoredTurn(BaseModel):
    """一行就是一轮完整记录；消息和统计一起落盘，避免只保存半轮消息。"""
    version: Literal[1] = 1
    # compact 是同一会话中的上下文替换记录；缺省 turn 兼容旧 JSONL。
    kind: Literal["turn", "compact"] = "turn"
    model_name: str
    saved_at: AwareDatetime
    input_tokens: int = Field(ge=0, strict=True)
    output_tokens: int = Field(ge=0, strict=True)
    # 先保存 SDK 转换出的 JSON 字典，恢复时再交给 SDK 校验并重建消息对象。
    messages: list[dict]


@dataclass
class SavedSession:
    """已校验的历史会话，同时提供选择列表需要的标题和更新时间。"""
    session_id: str
    history: list[ModelMessage]
    input_tokens: int
    output_tokens: int
    updated_at: datetime
    title: str


def _session_path(session_id: str) -> Path:
    """只接受程序生成的 UUID，不能把任意用户路径拼到存储目录里。"""
    if UUID(session_id).hex != session_id:
        raise ValueError("会话编号不是有效的 UUID")
    return SESSION_DIR / f"{session_id}.jsonl"


def _read_turns(path: Path) -> tuple[list[StoredTurn], int]:
    """校验完整行，返回记录及有效字节长度；崩溃留下的未完成末行不恢复。"""
    turns = []
    valid_bytes = 0
    for line in path.read_bytes().splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break
        turn = StoredTurn.model_validate_json(line)
        # 元数据与消息结构都合法才算完整记录，不能把损坏内容静默传给模型。
        messages = ModelMessagesTypeAdapter.validate_python(turn.messages)
        if turn.kind == "compact":
            from context_injection import is_compact_summary
            if not messages or not is_compact_summary(messages[0]):
                raise ValueError("压缩记录缺少带来源标记的摘要")
            compact_id = messages[0].metadata.get("compact_id", "")
            if not isinstance(compact_id, str) or not compact_id or UUID(compact_id).hex != compact_id:
                raise ValueError("压缩记录缺少有效边界编号")
        turns.append(turn)
        valid_bytes += len(line)
    return turns, valid_bytes


def _active_messages(turns: list[StoredTurn]) -> list[dict]:
    """磁盘保留所有轮次，当前模型历史从最后一个压缩边界重新开始。"""
    messages = []
    for turn in turns:
        if turn.kind == "compact":
            messages = []
        messages.extend(turn.messages)
    return messages


def history_context_id(history: list[ModelMessage]) -> str:
    """检查点的位置属于某一段有效历史，不能跨压缩边界使用旧偏移。"""
    from context_injection import is_compact_summary
    if history and is_compact_summary(history[0]):
        return history[0].metadata.get("compact_id", "")
    return ""


def session_context_id(session_id: str) -> str:
    path = _session_path(session_id)
    if not path.exists():
        return ""
    turns, _ = _read_turns(path)
    return history_context_id(ModelMessagesTypeAdapter.validate_python(_active_messages(turns)))


def save_session(state: "SessionState") -> None:
    """追加还未落盘的消息；写入失败时不推进保存位置，下次仍可补存。"""
    if len(state.history) <= state.saved_messages:
        return
    path = _session_path(state.session_id)
    # 先序列化，失败时不创建文件，也不会破坏已经保存的会话。
    history = ModelMessagesTypeAdapter.dump_python(state.history, mode="json")
    valid_bytes = 0
    saved_count = 0
    if path.exists():
        turns, valid_bytes = _read_turns(path)
        saved = _active_messages(turns)
        saved_count = len(saved)
        if saved != history[:saved_count] or saved_count > len(history):
            raise ValueError("磁盘会话与当前历史不一致，请用 /resume 重新选择会话")
        # 上次写完后同步磁盘失败，也可能留下完整记录；识别已存在内容以免重复追加。
    turn = StoredTurn(
        model_name=state.model_name,
        saved_at=datetime.now(timezone.utc),
        input_tokens=state.input_tokens,
        output_tokens=state.output_tokens,
        messages=history[saved_count:],
    )
    payload = (turn.model_dump_json() + "\n").encode("utf-8")
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as file:
        # 只去掉上次未完成的末行；已经校验的完整历史保持不变。
        file.truncate(valid_bytes)
        if turn.messages:
            file.write(payload)
        # 重试时即使整行已经存在，也要再次同步，确认落盘后才推进保存位置。
        file.flush()
        os.fsync(file.fileno())
    state.saved_messages = len(state.history)


def save_compacted_history(state: "SessionState", history: list[ModelMessage]) -> None:
    """原日志后追加一条完整压缩记录；单文件原子替换避免边界与摘要只保存一半。"""
    from context_injection import is_compact_summary
    from rewind_store import _atomic_save
    if not history or not is_compact_summary(history[0]) or not history_context_id(history):
        raise ValueError("压缩历史缺少摘要或边界编号")
    path = _session_path(state.session_id)
    turns, valid_bytes = _read_turns(path)
    original = ModelMessagesTypeAdapter.dump_python(state.history, mode="json")
    if _active_messages(turns) != original:
        raise ValueError("磁盘会话与压缩前历史不一致，请重新恢复会话")
    turn = StoredTurn(kind="compact", model_name=state.model_name,
                      saved_at=datetime.now(timezone.utc), input_tokens=state.input_tokens,
                      output_tokens=state.output_tokens,
                      messages=ModelMessagesTypeAdapter.dump_python(history, mode="json"))
    # 写临时文件、fsync、replace；失败时旧文件仍完整，成功后保留原日志字节前缀。
    _atomic_save(path, path.read_bytes()[:valid_bytes] + (turn.model_dump_json() + "\n").encode("utf-8"))


def load_session(session_id: str) -> SavedSession:
    """恢复完整消息对象和累计统计，不调用模型，也不重新执行历史中的工具。"""
    from task_store import TaskDocument, TaskStore
    from rewind_store import RewindDocument, RewindStore
    from context_injection import is_system_reminder, is_compact_summary

    path = _session_path(session_id)
    task_path = TaskStore.path(session_id)
    tasks = TaskDocument.model_validate_json(task_path.read_bytes()) if task_path.exists() else None
    rewind_path = RewindStore.path(session_id)
    rewind = RewindDocument.model_validate_json(rewind_path.read_bytes()) if rewind_path.exists() else None
    turns, _ = _read_turns(path) if path.exists() else ([], 0)
    if not turns:
        if tasks is not None:
            # 第一轮模型失败时任务已经落盘，但聊天还没有完整轮次，仍允许恢复计划。
            title = tasks.tasks[0].subject if tasks.tasks else "仅任务记录"
            updated = datetime.fromtimestamp(task_path.stat().st_mtime, timezone.utc)
            return SavedSession(session_id, [], 0, 0, updated, title)
        if rewind is not None and rewind.checkpoints:
            # 失败轮次可能只有检查点及文件备份，仍要让 /resume 找回并 /rewind。
            title = rewind.checkpoints[0].prompt[:60] or "仅检查点记录"
            updated = datetime.fromtimestamp(rewind_path.stat().st_mtime, timezone.utc)
            return SavedSession(session_id, [], 0, 0, updated, title)
        raise ValueError("会话没有完整记录")
    history = ModelMessagesTypeAdapter.validate_python(_active_messages(turns))
    # 标题沿用原会话第一条真实用户输入，不因摘要或下一轮问题而改变。
    title_history = ModelMessagesTypeAdapter.validate_python([m for turn in turns for m in turn.messages])
    title = rewind.checkpoints[0].prompt[:60] if rewind is not None and rewind.checkpoints else "未命名会话"
    found = False
    for message in title_history:
        if is_compact_summary(message):
            from classifier import compact_authorizations
            users = [record["user"] for record in compact_authorizations(message) if "user" in record]
            if users and not found:
                title = "压缩：" + " ".join(users[0].split())[:55]
            continue
        if is_system_reminder(message):
            continue
        for part in message.parts:
            if part.part_kind == "user-prompt":
                title = " ".join(str(part.content).split())[:60] or title
                found = True
                break
        if found:
            break
    last = turns[-1]
    updated = max(last.saved_at, datetime.fromtimestamp(task_path.stat().st_mtime, timezone.utc)) if tasks is not None else last.saved_at
    if rewind is not None:
        updated = max(updated, datetime.fromtimestamp(rewind_path.stat().st_mtime, timezone.utc))
    return SavedSession(session_id, history, last.input_tokens, last.output_tokens, updated, title)


def list_sessions() -> tuple[list[SavedSession], list[str]]:
    """扫描当前项目会话；单个文件损坏不阻止其他会话恢复，也不删除坏文件。"""
    sessions = []
    unreadable = []
    # 独立任务在失败轮次也会保存，所以同时发现尚无聊天记录的任务会话。
    ids = {path.stem for path in SESSION_DIR.glob("*.jsonl")}
    ids.update(path.name.removesuffix(".tasks.json") for path in SESSION_DIR.glob("*.tasks.json"))
    ids.update(path.name.removesuffix(".rewind.json") for path in SESSION_DIR.glob("*.rewind.json"))
    for session_id in sorted(ids):
        try:
            sessions.append(load_session(session_id))
        except (OSError, ValueError) as error:
            unreadable.append(f"{session_id}（{type(error).__name__}）")
    sessions.sort(key=lambda session: session.updated_at, reverse=True)
    return sessions, unreadable


def archive_session(state: "SessionState") -> Path:
    """先补存完整历史，再保留原始 JSONL 和侧车日志；名字唯一，存档不进入 /resume 扫描。"""
    from rewind_store import _atomic_save
    save_session(state)
    source = _session_path(state.session_id)
    path = SESSION_DIR / "compact-history" / (
        f"{state.session_id}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid4().hex}.jsonl"
    )
    _atomic_save(path, source.read_bytes())
    for suffix in (".tasks.json", ".rewind.json"):
        sidecar = source.with_suffix(suffix)
        if sidecar.exists():
            _atomic_save(path.with_suffix(suffix), sidecar.read_bytes())
    return path
