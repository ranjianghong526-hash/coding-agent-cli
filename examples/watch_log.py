"""跨平台示例：只输出日志中新追加的匹配行，供 monitor 工具接收事件。"""
import argparse
import time
from pathlib import Path


def watch(path: Path, patterns: list[str], interval: float = 0.2):
    """初次跳过已有内容，支持文件被截短/替换；逐行 flush，不依赖 tail/grep。"""
    handle = None
    identity = None
    initial = True
    partial = ""
    try:
        while True:
            try:
                stat = path.stat()
            except FileNotFoundError:
                if handle:
                    handle.close()
                    handle = None
                    partial = ""
                time.sleep(interval)
                continue
            current = (stat.st_dev, stat.st_ino)
            if handle is None or current != identity:
                if handle:
                    handle.close()
                handle = path.open(encoding="utf-8", errors="replace")
                identity = current
                partial = ""
                if initial:
                    handle.seek(0, 2)
                    initial = False
            elif stat.st_size < handle.tell():
                handle.seek(0)
                partial = ""
            chunk = handle.readline()
            if not chunk:
                time.sleep(interval)
                continue
            partial += chunk
            if not partial.endswith("\n"):
                continue
            line, partial = partial.rstrip("\r\n"), ""
            if any(pattern in line for pattern in patterns):
                print(line, flush=True)
    finally:
        if handle:
            handle.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--match", action="append", help="要匹配的文本，可重复指定；默认 ERROR")
    args = parser.parse_args()
    try:
        watch(args.path, args.match or ["ERROR"])
    except KeyboardInterrupt:
        pass
