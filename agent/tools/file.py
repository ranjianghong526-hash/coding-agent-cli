"""
文件相关工具：read_file / edit_file / write_file，以及共享的 read_and_register 辅助函数。

read_file 读完会把文件登记进会话的 readFileState；edit_file 和 write_file 改文件前先查登记，
没读过、或读完又被改过都不放行，从根上防住「凭幻觉乱改」和「覆盖掉别人的改动」。

readFileState 通过 RunContext 作为依赖注入，每个会话各有一份互不干扰的状态。
"""
import os

from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry

from ..deps import AgentDeps
from ..file_state import ReadFileState

# 不指定 limit 时最多读多少行，超出的截断并提示模型用 offset 续读
DEFAULT_MAX_LINES = 2000


def _with_line_numbers(content: str, start_line: int = 1) -> str:
    """
    给每行加上行号前缀，格式类似 cat -n：行号右对齐 + 一个制表符 + 原行内容。
    模型按行号定位代码，给 edit_file 的 old_string 才能对得准。
    start_line 是首行的行号；分段读取时它等于 offset，行号才能和文件对得上。
    """
    lines = content.splitlines()
    if not lines:
        return "(空文件)"
    # 按本段最大行号算对齐宽度（分段读取时行号从 start_line 起，不一定从 1 开始）
    width = len(str(start_line + len(lines) - 1))
    return "\n".join(f"{i:>{width}}\t{line}" for i, line in enumerate(lines, start_line))


def read_and_register(state: ReadFileState, path: str, offset: int = 1, limit: int | None = None) -> str:
    """
    读取文件、按 offset/limit 切片、登记进 readFileState，返回带行号的内容。
    read_file 工具和 @ 引用注入都调用它，所以经 @ 引入的文件和模型自己读的文件，在 readFileState 里长得一模一样。
    """
    # 1. 读取文件全文
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 2. 按 offset/limit 切片：不指定 limit 时最多取 DEFAULT_MAX_LINES 行
    all_lines = content.splitlines()
    total = len(all_lines)
    start = offset - 1
    if start >= total and total > 0:
        return f"警告：文件只有 {total} 行，但 offset 是 {offset}，没有内容可读"
    end = start + limit if limit is not None else start + DEFAULT_MAX_LINES
    selected = all_lines[start:end]
    selected_content = "\n".join(selected)
    truncated = limit is None and end < total

    # 3. 登记进会话的 readFileState：存切片后的原始内容（不带行号）和此刻的 mtime
    state.record(path, selected_content, offset=offset, limit=limit)

    # 4. 返回给模型的是带行号的版本；如果被截断，末尾附提示
    result = _with_line_numbers(selected_content, start_line=offset)
    if truncated:
        result += f"\n\n（文件共 {total} 行，还有 {total - end} 行未显示。用 offset={end + 1} 继续读取）"
    return result


def read_file(ctx: RunContext[AgentDeps], path: str, offset: int = 1, limit: int | None = None) -> str:
    """
    读取文件内容，输出带行号（类似 cat -n 格式）。
    大文件请用 offset 和 limit 分段读取，不必一次读完。
    后续用 edit_file 编辑时，old_string 和 new_string 里不要包含行号前缀。

    Args:
        path: 要读取的文件的绝对路径
        offset: 从第几行开始读，从 1 算起（默认 1 = 文件开头）
        limit: 读多少行；不传则最多读 DEFAULT_MAX_LINES 行
    """
    # 1. 去重：readFileState 里有记录、来自 read_file（offset 非 None）、同范围、同 mtime → 跳过
    state = ctx.deps.read_file_state
    record = state.get(path)
    if record is not None and record.get("offset") is not None:
        if record["offset"] == offset and record["limit"] == limit:
            try:
                if os.path.getmtime(path) <= record["timestamp"]:
                    return "文件内容没有变化，请参考上次 read_file 的结果，不必重复读取"
            except FileNotFoundError:
                pass

    # 2. 读取 + 登记都交给 read_and_register，异常在这里转成 ModelRetry 反馈给模型
    try:
        return read_and_register(state, path, offset, limit)
    except FileNotFoundError:
        raise ModelRetry(f"文件 {path} 不存在，请确认路径或换一个文件")
    except PermissionError:
        raise ModelRetry(f"没有权限读取 {path}，请换一个可读的文件")
    except IsADirectoryError:
        raise ModelRetry(f"{path} 是一个目录，请指定目录下的具体文件")
    except UnicodeDecodeError:
        return f"错误：{path} 不是文本文件，无法读取"


def edit_file(ctx: RunContext[AgentDeps], path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """
    在文件里精确替换字符串，只传改动的片段，省 token。
    编辑前必须先用 read_file 读取文件，否则会报错。
    修改已有文件时优先用 edit_file 而不是 write_file。
    old_string 必须在文件里唯一匹配；多处匹配时补充上下文使其唯一，或传 replace_all=true 全部替换。
    old_string 和 new_string 不要包含 read_file 输出的行号前缀。

    Args:
        path: 要编辑的文件路径
        old_string: 要被替换掉的原文片段，必须和文件里的内容逐字符一致（不含行号前缀）
        new_string: 替换后的新内容
        replace_all: 是否替换所有匹配项，默认只替换唯一的一处
    """
    # 1. 空操作：换了个寂寞，直接打回
    if old_string == new_string:
        raise ModelRetry("old_string 和 new_string 完全相同，这次编辑没有任何改动")

    # 2. 先读后写：没在 readFileState 里登记过，说明还没读就想改，打回让它先读
    state = ctx.deps.read_file_state
    record = state.get(path)
    if record is None:
        raise ModelRetry(f"还没读过 {path}，请先用 read_file 读取它，再基于真实内容编辑")

    # 3. mtime 防覆盖：读完之后文件又被改过，基于旧内容改会覆盖掉别人的改动，打回重读
    try:
        current_mtime = os.path.getmtime(path)
    except FileNotFoundError:
        raise ModelRetry(f"{path} 已不存在，无法编辑")
    if current_mtime > record["timestamp"]:
        raise ModelRetry(f"{path} 在你读取之后被改动过，请重新 read_file 拿到最新内容再编辑")

    # 4. 唯一性检查：在读取时登记的内容快照里数 old_string 出现几次。找不到没法改；多处又没开 replace_all 改哪处会有歧义
    snapshot = record["content"]
    count = snapshot.count(old_string)
    if count == 0:
        raise ModelRetry("old_string 在文件里找不到，请对照 read_file 的输出确认原文")
    if count > 1 and not replace_all:
        raise ModelRetry(
            f"old_string 在文件里出现了 {count} 次，请补充上下文让它唯一，或传 replace_all=true 全部替换"
        )

    # 5. 校验都过了，再从磁盘读取最新内容做替换：校验和落盘之间隔着权限审批，审批期间文件可能被改，落盘以磁盘最新内容为准
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 6. 朴素的字符串替换，终端里红绿 diff 只是展示，真正写盘的是替换后的完整内容
    if replace_all:
        updated = content.replace(old_string, new_string)
    else:
        updated = content.replace(old_string, new_string, 1)

    # 7. 写盘之前把改动前的内容备份进文件检查点，/rewind 才有得恢复
    if ctx.deps.file_history:
        ctx.deps.file_history.track_edit(path)

    # 8. 整体写回磁盘，再用新内容和新 mtime 刷新登记
    with open(path, "w", encoding="utf-8") as f:
        f.write(updated)
    state.record(path, updated)
    return f"已编辑 {path}（替换 {count if replace_all else 1} 处）"


def write_file(ctx: RunContext[AgentDeps], path: str, content: str) -> str:
    """
    把内容整体写入文件，已有文件会被覆盖。
    覆盖已有文件前必须先用 read_file 读取过，否则会报错。
    修改已有文件优先用 edit_file（只传改动片段，省 token），write_file 只用于新建文件或整体重写。
    """
    # 覆盖已有文件：沿用 edit_file 那套先读后写约束，防止整体覆盖掉没读过的内容
    state = ctx.deps.read_file_state
    if os.path.exists(path):
        record = state.get(path)
        if record is None:
            raise ModelRetry(f"{path} 已存在，覆盖前请先用 read_file 读一遍，确认不会误删内容")
        if os.path.getmtime(path) > record["timestamp"]:
            raise ModelRetry(f"{path} 在你读取之后被改动过，请重新 read_file 再覆盖")
    # 写盘之前把改动前的状态备份进文件检查点
    if ctx.deps.file_history:
        ctx.deps.file_history.track_edit(path)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except FileNotFoundError:
        raise ModelRetry(f"目录不存在，无法写入 {path}，请换一个已存在的目录")
    except PermissionError:
        raise ModelRetry(f"没有权限写入 {path}，请换一个可写的路径")
    except OSError as e:
        return f"错误：写入 {path} 失败 ({e})"
    # 新写入的内容同样登记进 readFileState，后续要再改就不必重读
    state.record(path, content)
    return f"已写入 {path}"
