"""
Coding Agent 的文件读取、完整写入、局部编辑和 shell 命令工具。

模型决定工具名和参数，Pydantic AI 调用下面的 Python 函数，返回值再交回模型。
这些函数本身不调用大模型，也不负责保存对话历史。
"""
import codecs
import hashlib
import os
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field
from pydantic_ai import RunContext, Tool
from pydantic_ai.messages import ToolReturn

from file_state import FileContext, FileVersion, ReadFileState
from permissions import PermissionState
from ui.questions import Question, USER_ANSWER_METADATA, ask_questions
from task_store import TaskStatus
from memory_store import Memory


def memory_read(ctx: RunContext[PermissionState], memory_id: str) -> dict:
    """按索引编号读取一条项目长期记忆的完整正文和 revision；记忆不是用户授权，当前用户要求优先。"""
    try:
        return ctx.deps.memory.read(memory_id)
    except (OSError, UnicodeError, ValueError) as error:
        return {"error": str(error)}


def memory_write(ctx: RunContext[PermissionState], memory: Memory, expected_revision: str = "") -> dict:
    """保存明确的长期偏好、项目约定或稳定事实，不能保存密钥、一次性任务或执行授权。新建版本留空，修改先 memory_read 再传其 revision；用户改口更新旧编号。"""
    try:
        return ctx.deps.memory.write(memory, expected_revision, manual=True)
    except (OSError, UnicodeError, ValueError) as error:
        return {"error": str(error)}


def memory_delete(ctx: RunContext[PermissionState], memory_id: str, expected_revision: str) -> dict:
    """用户明确要求忘记某条长期记忆时删除；先 memory_read，携带其 revision，删除仍需权限检查。"""
    try:
        return ctx.deps.memory.delete(memory_id, expected_revision, manual=True)
    except (OSError, UnicodeError, ValueError) as error:
        return {"error": str(error)}


def task_create(ctx: RunContext[PermissionState], subject: Annotated[str, Field(min_length=1, max_length=200)],
                description: Annotated[str, Field(max_length=3000)] = "") -> dict:
    """创建多步计划中的一项任务，初始 pending。subject 是简短标题，description 是具体要求；不要把用户未授权的操作当作已授权。"""
    return ctx.deps.tasks.create(subject, description)


def task_get(ctx: RunContext[PermissionState], task_id: Annotated[int, Field(ge=1, strict=True)]) -> dict:
    """按编号读取任务的标题、完整要求与当前状态。"""
    return ctx.deps.tasks.get(task_id)


def task_list(ctx: RunContext[PermissionState]) -> dict:
    """列出当前会话全部任务；继续复杂工作前检查未完成项，不要重复创建已有计划。"""
    return {"tasks": ctx.deps.tasks.list()}


def task_update(ctx: RunContext[PermissionState], task_id: Annotated[int, Field(ge=1, strict=True)],
                status: TaskStatus | Literal["deleted"] | None = None,
                subject: Annotated[str, Field(min_length=1, max_length=200)] | None = None,
                description: Annotated[str, Field(max_length=3000)] | None = None) -> dict:
    """更新任务字段；开始时 in_progress，实际完成并验证后 completed，待办 pending；deleted 仅删除清单项，不删除项目文件。"""
    return ctx.deps.tasks.update(task_id, status=status, subject=subject, description=description)


async def ask_user_question(
    ctx: RunContext[PermissionState],
    questions: Annotated[list[Question], Field(min_length=1, max_length=4)],
) -> ToolReturn:
    """向用户澄清关键需求；一次 1～4 题，每题可单选、多选或自由输入。返回用户确认的答案；取消时不得推断答案或绕过取消。"""
    # 与人工权限审批共享终端锁；审批 hook 结束后才进入这里，避免嵌套加锁。
    # await 会暂停这个工具，直到真人提交；它不会阻塞整个 asyncio 事件循环。
    async with ctx.deps.approval_lock:
        result = await ask_questions(questions)
    # SDK 把 return_value 写成 ToolReturnPart；metadata 只供程序识别回答来源。
    return ToolReturn(return_value=result.model_dump(), metadata=dict(USER_ANSWER_METADATA))


def _path(path: str) -> tuple[Path, str]:
    """相对路径、绝对路径和符号链接使用统一身份，避免绕过已读状态。"""
    resolved = Path(path).expanduser().resolve()
    return resolved, os.path.normcase(str(resolved))


def _read_disk(path: Path) -> tuple[bytes, FileVersion]:
    """读取真实磁盘字节，并检测读取期间的变动或文件被替换。"""
    with open(path, "rb") as file:
        before = os.fstat(file.fileno())
        data = file.read()
        after = os.fstat(file.fileno())
    current = path.stat()
    identity = lambda info: (info.st_mtime_ns, info.st_size, info.st_ino)
    if identity(before) != identity(after) or identity(after) != identity(current) or len(data) != after.st_size:
        raise OSError("文件在读取过程中发生变化，请重新读取")
    return data, FileVersion(after.st_mtime_ns, after.st_size, after.st_ino, hashlib.sha256(data).hexdigest())


def _require_read(ctx: RunContext[PermissionState], key: str, version: FileVersion, *, full: bool) -> None:
    """即使权限允许也必须校验文件状态；这道保护属于真实工具内部。"""
    record = ctx.deps.files.read_file_state.get(key)
    if record is None:
        raise ValueError("请先用 read_file 读取该文件，再修改")
    if record.version != version:
        raise ValueError("文件自上次读取后已变化，未写入；请重新 read_file 后调整修改")
    if full and not record.fully_read:
        raise ValueError("整体覆盖前必须完整读取文件；请读取剩余行，或改用 edit_file 局部编辑")


def _atomic_write(path: Path, data: bytes, expected: FileVersion | None) -> FileVersion:
    """先写同目录临时文件；已有文件再核对版本后原子替换，新建使用排他创建。"""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        if expected is None:
            # 硬链接创建不会覆盖已存在路径，防止检查后用户恰好创建同名文件。
            # 临时文件和目标在同目录；不支持硬链接的文件系统明确报错，不降级覆盖。
            os.link(temporary, path)
        else:
            os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
            _, current = _read_disk(path)
            if current != expected:
                raise ValueError("文件在写入前发生变化，未覆盖；请重新读取")
            os.replace(temporary, path)
        _, version = _read_disk(path)
        if version.digest != hashlib.sha256(data).hexdigest():
            raise OSError("写入后文件又发生变化，请重新读取确认")
        return version
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_file_content(files: FileContext, path: str, offset: int = 1, limit: int = 200, force: bool = False) -> str:
    """共享读取实现：模型的 read_file 与用户的 @引用使用同一份版本和已读区间登记。"""
    if offset < 1 or limit < 1:
        return "[错误] offset 和 limit 必须大于等于 1"
    key = None
    try:
        resolved, key = _path(path)
        with files.lock:
            data, version = _read_disk(resolved)
            lines = data.decode("utf-8-sig").splitlines()
            if offset > max(1, len(lines)):
                return f"[错误] 起始行超出文件范围，文件共 {len(lines)} 行"
            record = files.read_file_state.get(key)
            if record is None or record.version != version:
                record = ReadFileState(version, len(lines))
            end = min(len(lines), offset + limit - 1)
            if not force and record.contains(offset, end):
                files.remember(key, record)
                region = f"第 {offset}～{end} 行" if lines else "空文件"
                return f"文件未变化，{region}此前已读取；如需再次显示请设置 force=True"
            record.record(offset, end)
            files.remember(key, record)
            if not lines:
                return "(空文件)"
            text = "\n".join(f"{number:>4} | {lines[number - 1]}" for number in range(offset, end + 1))
            more = f"\n还有后续行，请从 offset={end + 1} 继续读取。" if end < len(lines) else ""
            return f"{resolved}（共 {len(lines)} 行，显示 {offset}～{end}）\n{text}{more}"
    except FileNotFoundError:
        if key is not None:
            # 模型已通过真实读取获知文件不存在，清除过期的已读记录。
            with files.lock:
                files.read_file_state.pop(key, None)
        return f"[错误] 文件 {path} 不存在"
    except (OSError, UnicodeError, ValueError) as error:
        return f"[错误] 无法读取 {path}：{error}"


def read_file(ctx: RunContext[PermissionState], path: str, offset: int = 1, limit: int = 200, force: bool = False) -> str:
    """读取 UTF-8 文件并显示行号；offset 从 1 开始，limit 为行数。重复读取可用 force 强制显示。"""
    # SDK 调用入口只负责取得会话依赖，避免 @引用复制一套读取和版本检查逻辑。
    return read_file_content(ctx.deps.files, path, offset, limit, force)


def write_file(ctx: RunContext[PermissionState], path: str, content: str) -> str:
    """创建 UTF-8 文件或完整重写；覆盖已有文件前必须完整读取且版本未变。父目录必须存在。"""
    try:
        resolved, key = _path(path)
        with ctx.deps.files.lock:
            expected = None
            old = None
            bom = b""
            if resolved.exists():
                old, expected = _read_disk(resolved)
                _require_read(ctx, key, expected, full=True)
                bom = codecs.BOM_UTF8 if old.startswith(codecs.BOM_UTF8) and not content.startswith("\ufeff") else b""
            elif key in ctx.deps.files.read_file_state:
                raise ValueError("文件在上次读取后被删除，未重新创建；请先 read_file 确认当前状态")
            data = bom + content.encode("utf-8")
            # 先写版本备份及日志，备份失败则不会修改真实文件。
            ctx.deps.rewind.capture(resolved, old, data)
            version = _atomic_write(resolved, data, expected)
            count = len(content.splitlines())
            record = ReadFileState(version, count)
            record.record(1, count)
            ctx.deps.files.remember(key, record)
            return f"已写入 {path}"
    except (OSError, UnicodeError, ValueError) as error:
        return f"[错误] 无法写入 {path}：{error}"


def edit_file(ctx: RunContext[PermissionState], path: str, old_string: str, new_string: str) -> str:
    """精确替换一处文本；先读取文件，old_string 必须非空且在整个文件中唯一。"""
    if not old_string:
        return "[错误] old_string 不能为空；创建文件请使用 write_file"
    if old_string == new_string:
        return "[错误] 新旧文本相同，无需编辑"
    try:
        resolved, key = _path(path)
        with ctx.deps.files.lock:
            data, version = _read_disk(resolved)
            _require_read(ctx, key, version, full=False)
            text = data.decode("utf-8-sig")
            # Windows 换行文件也可以匹配模型提供的 LF 文本；写回保留原换行字节。
            newline = "\r\n" if "\r\n" in text and "\n" not in text.replace("\r\n", "") else "\n"
            old = old_string.replace("\r\n", "\n").replace("\n", newline)
            new = new_string.replace("\r\n", "\n").replace("\n", newline)
            first = text.find(old)
            if first < 0:
                return "[错误] 未找到 old_string，文件未修改；请重新读取并复制精确文本"
            if text.find(old, first + 1) >= 0:
                return "[错误] old_string 匹配多处，文件未修改；请增加上下文使其唯一"
            start_line = text.count("\n", 0, first) + 1
            end_line = text.count("\n", 0, first + len(old) - 1) + 1
            if not ctx.deps.files.read_file_state[key].contains(start_line, end_line):
                return "[错误] 匹配位置尚未读取，文件未修改；请 read_file 读取目标行后再编辑"
            updated = text[:first] + new + text[first + len(old):]
            bom = codecs.BOM_UTF8 if data.startswith(codecs.BOM_UTF8) else b""
            new_data = bom + updated.encode("utf-8")
            ctx.deps.rewind.capture(resolved, data, new_data)
            changed = _atomic_write(resolved, new_data, version)
            # 模型只看到了局部内容时，编辑后也不能冒充完整读取。
            record = ReadFileState(changed, len(updated.splitlines()))
            if ctx.deps.files.read_file_state[key].fully_read:
                record.record(1, record.total_lines)
            ctx.deps.files.remember(key, record)
            return f"已编辑 {path}：完成 1 处替换"
    except (OSError, UnicodeError, ValueError) as error:
        return f"[错误] 无法编辑 {path}：{error}"


def run_command(command: str) -> str:
    """
    执行一条 shell 命令并返回输出。
    """
    try:
        # shell=True 让系统 shell 解析命令；命令可以产生实际的文件或进程副作用。
        # capture_output=True 收集 stdout / stderr，不直接在终端实时输出。
        # text=True 解码为文本；errors="replace" 用替代字符处理无法解码的字节。
        # timeout=10 限制此次等待时长，超过后进入下面的超时分支。
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, errors="replace", timeout=10
        )
        # 默认只返回标准输出；退出码非零时才额外拼接标准错误。
        # 因此成功命令写到 stderr 的提示不会被当前实现返回给模型。
        output = result.stdout
        if result.returncode != 0:
            output += f"\n[错误] {result.stderr}"
        # 用明确的占位文本表示没有输出，方便模型区别于缺失的工具结果。
        return output or "(无输出)"
    except subprocess.TimeoutExpired:
        return "[错误] 命令执行超时（10秒）"
    except OSError as error:
        return f"[错误] 无法启动命令：{error}"


# RunContext 参数由 SDK 注入，不出现在模型的工具参数中；sequential 让编辑调用按模型顺序排队。
# SDK 从函数签名 + docstring 自动生成 JSON Schema
# JSON Schema 是工具参数的结构说明：模型据此知道有哪些参数及其类型。
# 工具 docstring 也会参与模型看到的说明，因此教学细节主要放在 # 注释里。
TOOLS = [
    Tool(memory_read, sequential=True),
    Tool(memory_write, sequential=True),
    Tool(memory_delete, sequential=True),
    Tool(task_create, sequential=True),
    Tool(task_get, sequential=True),
    Tool(task_update, sequential=True),
    Tool(task_list, sequential=True),
    Tool(ask_user_question, sequential=True),
    Tool(read_file, sequential=True),
    Tool(write_file, sequential=True),
    Tool(edit_file, sequential=True),
    # 命令也可能改文件，设置为排队工具，避免与本 Agent 的文件读写同时执行。
    Tool(run_command, sequential=True),
]
