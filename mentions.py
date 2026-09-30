"""
@ 引用文件：把用户输入里的 @path 解析出来，逐个读成文件内容，伪装成一次 read_file 工具调用塞进对话历史，
让模型以为自己已经读过这个文件，先读后写的检查也就直接放行，可以立刻编辑。
"""
import os
import re
import subprocess
import uuid

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)

from agent.file_state import ReadFileState
from agent.tools import read_and_register
import images

# 匹配 @ 引用：要求 @ 前面是行首或空白，避免把 foo@bar.com 这种邮箱误判成引用
# @ 后面跟一段路径常见字符（字母、数字、下划线、点、斜杠、反斜杠、连字符、波浪号），遇到空格或中文标点就停，不会把后面的句子也吞进来
# 只支持纯路径 @path 的最简形态，不处理带引号路径、行号区间、@agent 等分支
AT_MENTION_RE = re.compile(r"(?:^|\s)@([\w./~\\:\-]+)")


def extract_at_mentions(text: str) -> list[str]:
    """
    从用户输入里抽出所有 @path，去重且保持出现顺序。
    """
    seen = []
    for match in AT_MENTION_RE.finditer(text):
        path = match.group(1)
        if path not in seen:
            seen.append(path)
    return seen


def build_mention_messages(paths: list[str], state: ReadFileState,
                           attachments: list | None = None, image_paths: list | None = None) -> list:
    """
    把每个被 @ 的文件读出来，为它伪造一对 read_file 的 tool_call + tool_return，返回要塞进历史的消息列表。
    read_and_register 顺手把文件登记进 readFileState，和真正的 read_file 调用一模一样，之后 edit_file 就不会被拦。
    """
    messages = []
    for path in paths:
        abs_path = os.path.abspath(os.path.expanduser(path))
        # 不存在的路径、目录都跳过，简化处理
        if not os.path.isfile(abs_path):
            continue
        if images.is_image(abs_path):
            if attachments is not None and image_paths is not None:
                try:
                    attachments.append(images.load_image(abs_path))
                    image_paths.append(path)
                except (OSError, ValueError):
                    continue
            continue
        try:
            numbered = read_and_register(state, abs_path)
        except (OSError, UnicodeDecodeError):
            continue

        # tool_call 和 tool_return 要用同一个 id 配对，模型才认得这是一次完整的调用
        call_id = "mention_" + uuid.uuid4().hex[:8]
        # 一条 assistant 消息，假装模型自己发起了 read_file 调用
        messages.append(
            ModelResponse(
                parts=[ToolCallPart(tool_name="read_file", args={"path": abs_path}, tool_call_id=call_id)]
            )
        )
        # 一条工具返回消息，内容就是带行号的文件正文，和 read_file 真跑一遍的返回完全一致
        messages.append(
            ModelRequest(
                parts=[ToolReturnPart(tool_name="read_file", content=numbered, tool_call_id=call_id)]
            )
        )
    return messages


def replace_image_mentions(text: str, image_paths: list[str], start_index: int = 1) -> str:
    """保留引用位置；重复引用同一个文件共用图片编号，与粘贴图片衔接。"""
    numbers = {path: start_index + i for i, path in enumerate(image_paths)}
    def replace(match):
        path = match.group(1)
        if path not in numbers:
            return match.group(0)
        prefix = match.group(0)[:match.start(1) - match.start() - 1]
        return prefix + f"[Image #{numbers[path]}]"
    return AT_MENTION_RE.sub(replace, text)


# 遍历目录时跳过的常见噪音目录：依赖、缓存、构建产物，列出来当文件候选没意义
# 遍历目录时没有 .gitignore 可以利用，用固定黑名单跳过这些常见的 deps/cache/build 目录
IGNORED_DIRS = {"__pycache__", "node_modules", "venv", "dist", "build", "target"}


def list_candidate_files() -> list[str]:
    """
    @ 补全的候选文件列表。优先用 git ls-files（快，又自动跳过 .gitignore 的文件）；不在 git 仓库就退回遍历目录。
    """
    try:
        result = subprocess.run(
            ["git", "ls-files"], capture_output=True, text=True, timeout=2
        )
        tracked = result.stdout.splitlines() if result.returncode == 0 else []
        # 拿到了已跟踪文件就直接用；空列表（非 git 仓库、或文件还没 add）则退回遍历目录
        if tracked:
            return tracked
    except (OSError, subprocess.SubprocessError):
        pass

    # 退回遍历当前目录，跳过隐藏目录（.git、.venv 之类）和黑名单里的噪音目录
    files = []
    for root, dirs, names in os.walk("."):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in IGNORED_DIRS]
        for name in names:
            files.append(os.path.relpath(os.path.join(root, name)))
    return files
