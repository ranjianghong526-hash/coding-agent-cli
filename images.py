"""图片输入：读取二进制图片、保持图文顺序，以及通过系统工具取得剪贴板图片。"""
import json
import os
import platform
import re
import shlex
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

from pydantic_ai import BinaryContent

IMAGE_EXTENSIONS = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".gif": "image/gif", ".webp": "image/webp"}
CLIPBOARD_DIR = Path.home() / ".my-claude-code" / "clipboard"
_IMAGE_PLACEHOLDER = re.compile(r"\[Image #(\d+)\]")
_CHECK_COMMANDS = {
    "Darwin": ["osascript", "-e", "clipboard info for «class PNGf»"],
    "Linux": ["xclip", "-selection", "clipboard", "-t", "TARGETS", "-o"],
    "Windows": ["powershell", "-NoProfile", "-STA", "-Command",
                "[bool](Get-Clipboard -Format Image)"],
}


def is_image(path: str) -> bool:
    """和教程一样按扩展名判断；它不是文件内容真实性校验。"""
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def load_image(path: str) -> BinaryContent:
    """保留原始字节和 MIME 类型，编码为 data URI 的工作交给 SDK。"""
    media_type = IMAGE_EXTENSIONS.get(Path(path).suffix.lower())
    if media_type is None:
        raise ValueError(f"不支持的图片格式：{Path(path).suffix}")
    data = Path(path).read_bytes()
    if not data:
        raise ValueError(f"图片 {path} 是空文件")
    return BinaryContent(data=data, media_type=media_type)


def build_user_content(text: str, attachments: list[BinaryContent]) -> list[str | BinaryContent]:
    """原地替换占位符，不把图片统一追加到文本末尾；删除占位符即不发送该图。"""
    content = []
    cursor = 0
    for match in _IMAGE_PLACEHOLDER.finditer(text):
        index = int(match.group(1)) - 1
        if not 0 <= index < len(attachments):
            raise ValueError(f"{match.group(0)} 没有对应图片，请重新粘贴或 @ 引用图片")
        if match.start() > cursor:
            content.append(text[cursor:match.start()])
        content.append(attachments[index])
        cursor = match.end()
    if cursor < len(text):
        content.append(text[cursor:])
    return content


def prompt_text(content) -> str:
    """旁路文本模型只看真实文字，不把图片字节的 repr 塞进授权、记忆或终端。"""
    if isinstance(content, str):
        return content
    return "".join(item for item in content if isinstance(item, str))


def content_summary(content) -> str:
    """终端与会话列表显示简短占位符，模型消息中的图片数据不受影响。"""
    if isinstance(content, BinaryContent):
        return f"[图片 {content.media_type}，{len(content.data) / 1024:.0f} KB]"
    if isinstance(content, (list, tuple)):
        return "".join(content_summary(item) if isinstance(item, BinaryContent) else str(item)
                       for item in content)
    return str(content)


def restore_prompt(content: list) -> tuple[str, list[BinaryContent]]:
    """从会话中的多模态消息还原草稿，回退时图片不依赖原文件仍然存在。"""
    parts, attachments = [], []
    for item in content:
        if isinstance(item, BinaryContent):
            attachments.append(item)
            parts.append(f"[Image #{len(attachments)}]")
        elif isinstance(item, str):
            parts.append(item)
    return "".join(parts), attachments


def _save_command(path: str) -> list[str]:
    system = platform.system()
    if system == "Windows":
        # 参数是我们生成的路径；仍转义引号，兼容含单引号的用户目录。
        quoted = path.replace("'", "''")
        return ["powershell", "-NoProfile", "-STA", "-Command",
                "$ErrorActionPreference='Stop'; Add-Type -AssemblyName System.Drawing; "
                f"$img=Get-Clipboard -Format Image; if ($img) {{ "
                f"$img.Save('{quoted}', [System.Drawing.Imaging.ImageFormat]::Png); $img.Dispose() }}"]
    if system == "Darwin":
        return ["osascript", "-e", "set png_data to (the clipboard as «class PNGf»)",
                "-e", f"set fp to open for access POSIX file {json.dumps(path, ensure_ascii=False)} with write permission",
                "-e", "write png_data to fp", "-e", "close access fp"]
    return ["sh", "-c", f"xclip -selection clipboard -t image/png -o > {shlex.quote(path)}"]


def read_clipboard_image() -> BinaryContent | None:
    """无图片、缺少系统工具或获取失败返回 None，由输入区明确提示。"""
    system = platform.system()
    check = _CHECK_COMMANDS.get(system)
    if check is None:
        return None
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    path = None
    try:
        probe = subprocess.run(check, capture_output=True, timeout=5, **options)
        if probe.returncode or not probe.stdout.strip():
            return None
        if system == "Windows" and b"True" not in probe.stdout:
            return None
        if system == "Linux" and b"image/png" not in probe.stdout:
            return None
        CLIPBOARD_DIR.mkdir(parents=True, exist_ok=True)
        path = CLIPBOARD_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex}.png"
        saved = subprocess.run(_save_command(str(path)), capture_output=True, timeout=10, **options)
        if saved.returncode:
            return None
        return load_image(str(path))
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    finally:
        # 失败时可能留下零字节文件，成功保存的图片保留供用户查看。
        if path and path.exists() and not path.stat().st_size:
            path.unlink()
