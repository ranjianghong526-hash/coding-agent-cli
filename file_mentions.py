"""@文件引用：输入补全、解析路径，以及在首次模型请求前准备真实文件的工具结果。"""
import os
import re
import subprocess
from pathlib import Path
from uuid import uuid4

from prompt_toolkit.completion import Completer, Completion
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart, UserPromptPart

from file_state import FileContext

# 只在输入开头或空白后识别 @，不会把 name@example.com 当成文件引用。
# 空格路径写成 @"docs/学习笔记.md"；未加引号的路径遇到空白结束。
MENTION = re.compile(r'''(?<!\S)@(?:"([^"\r\n]+)"|'([^'\r\n]+)'|([^\s"'@]+))''')
INCOMPLETE_MENTION = re.compile(r'''(?:^|\s)(@(?:"[^"\r\n]*|'[^'\r\n]*|[^\s"'@]*))$''')
EXCLUDED_DIRECTORIES = {".git", ".venv", "venv", "node_modules", ".sessions", ".codex", ".agents",
                        "__pycache__", ".pytest_cache", ".mypy_cache", "build", "dist"}


def _candidate(path: str) -> bool:
    """候选不展示密钥、聊天记录、依赖和缓存；用户明确输入的路径另由读取函数处理。"""
    parts = Path(path).parts
    name = parts[-1]
    return (not any(part in EXCLUDED_DIRECTORIES for part in parts[:-1])
            and not (name == ".env" or (name.startswith(".env.") and name != ".env.example"))
            and not name.endswith((".pyc", ".pyo")))


def list_project_files(root: Path) -> list[str]:
    """Git 项目遵守 .gitignore；没有 Git 时使用标准库遍历并排除已知依赖目录。"""
    root = root.resolve()
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            capture_output=True, timeout=3,
        )
        if result.returncode == 0:
            # -z 用 NUL 分隔文件名，保留中文和包含空格的路径，避免解析 Git 的引号转义。
            paths = result.stdout.decode("utf-8").split("\0")
            return sorted({path for path in paths if path and _candidate(path) and (root / path).is_file()})
    except (OSError, UnicodeError, subprocess.TimeoutExpired):
        pass
    paths = []
    for directory, children, names in os.walk(root):
        children[:] = [name for name in children if name not in EXCLUDED_DIRECTORIES
                       and not (Path(directory) / name).is_symlink()]
        for name in names:
            path = (Path(directory) / name).relative_to(root).as_posix()
            if _candidate(path) and (root / path).is_file():
                paths.append(path)
    return sorted(paths)


def _mention_text(path: str) -> str:
    # 补全后仍必须是解析器可读的引用；有空格、单引号或 @ 时自动加双引号。
    return '@"' + path + '"' if any(c.isspace() or c in "'@" for c in path) else "@" + path


class FileMentionCompleter(Completer):
    """只补全光标前当前的 @片段，保留前面的自然语言需求。"""
    def __init__(self, root: Path | None = None):
        self.root = root

    def get_completions(self, document, complete_event):
        match = INCOMPLETE_MENTION.search(document.text_before_cursor)
        if match is None or document.text.lstrip().startswith("/"):
            return
        token = match.group(1)
        prefix = token[1:].lstrip("\"'").replace("\\", "/").casefold()
        for path in list_project_files(self.root or Path.cwd()):
            if prefix in path.casefold():
                yield Completion(_mention_text(path), start_position=-len(token), display=path, display_meta="文件")


def extract_mentions(text: str) -> list[str]:
    """同一文件即便用不同相对路径引用，本轮也只读一次；保留第一次出现的顺序。"""
    paths, seen = [], set()
    for match in MENTION.finditer(text):
        quoted = match.group(1) or match.group(2)
        path = quoted if quoted is not None else match.group(3).rstrip(",，。;；!?！？")
        if not path:
            continue
        key = os.path.normcase(str(Path(path).expanduser().resolve()))
        if key not in seen:
            seen.add(key)
            paths.append(path)
    return paths


def prepare_file_messages(text: str, files: FileContext) -> list:
    """有引用时生成：用户原话 → read_file 调用 → 真实读取结果；无引用返回空列表。"""
    paths = extract_mentions(text)
    if not paths:
        return []
    # 延迟导入：单独使用补全/解析功能不需要初始化 Agent 或配置 API Key。
    from agent.tools import read_file_content

    calls, returns = [], []
    for path in paths:
        call_id = "mention-" + uuid4().hex
        # 强制展示当前内容，避免恢复会话或重复引用时只拿到“以前读过”的提示。
        args = {"path": path, "force": True}
        content = read_file_content(files, **args)
        calls.append(ToolCallPart("read_file", args, tool_call_id=call_id))
        returns.append(ToolReturnPart("read_file", content, tool_call_id=call_id))
    # 文件内容属于 tool-return，不能拼进 user-prompt 冒充用户授权。
    # 两边相同的 tool_call_id 用于将每个调用与它的结果配对。
    return [ModelRequest([UserPromptPart(text)]), ModelResponse(calls), ModelRequest(returns)]
