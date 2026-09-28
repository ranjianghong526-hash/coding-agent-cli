"""
Coding Agent 用到的三个工具：读文件、写文件、跑 shell 命令。

模型决定工具名和参数，Pydantic AI 调用下面的 Python 函数，返回值再交回模型。
这些函数本身不调用大模型，也不负责保存对话历史。
"""
import subprocess


def read_file(path: str) -> str:
    """
    读取指定文件的内容。
    """
    try:
        # with 在读取成功或发生异常后都会关闭文件，避免文件句柄泄漏。
        # 相对路径按进程工作目录解析，而不是按本 tools.py 所在目录解析。
        with open(path, "r", encoding="utf-8") as f:
            # 一次性返回整个文件，没有按大小截断或按行分页。
            return f.read()
    except FileNotFoundError:
        # 已知环境错误转换成工具结果，模型据此调整路径或操作。
        return f"错误：文件 {path} 不存在"
    except (OSError, UnicodeError) as error:
        # 例如权限不足、路径是目录、文件不是 UTF-8；错误不能伪装成读取成功。
        return f"[错误] 无法读取 {path}：{error}"


def write_file(path: str, content: str) -> str:
    """
    将内容写入指定文件。
    """
    # w 模式会覆盖已有文件，也会创建新文件；不会自动创建父目录。
    # 因此模型应先读已有内容，再决定需要写回的完整文本。
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except (OSError, UnicodeError) as error:
        # 父目录不存在、没有写入权限等问题直接反馈，不自动更换路径或重复写入。
        return f"[错误] 无法写入 {path}：{error}"
    return f"已写入 {path}"


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


# Pydantic AI 支持 tools=[plain_function]，从函数签名 + docstring 自动生成 JSON Schema
# JSON Schema 是工具参数的结构说明：模型据此知道有哪些参数及其类型。
# 工具 docstring 也会参与模型看到的说明，因此教学细节主要放在 # 注释里。
TOOLS = [read_file, write_file, run_command]
