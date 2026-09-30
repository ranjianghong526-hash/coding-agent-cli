"""
工具子包：按职责拆成 file（读写文件）/ shell（执行命令）/ tasks（task 列表管理）/ ask_user（向用户提问）。

外部代码只需要：
- TOOLS: 注册给 Agent 的工具列表
- read_and_register: mentions.py 复用，把 @ 引用伪装成 read_file 调用塞进对话历史
"""
from pydantic_ai import Tool

# 显式触发 shell.py 顶层 permissions.register_self_check 副作用：必须在 import 时就把
# run_command 的高危特征自检挂上去，否则后续 default 模式下的危险命令直接放行
from . import shell as _shell  # noqa: F401
from .ask_user import ask_user_question
from .file import read_and_register, read_file, edit_file, write_file
from .shell import run_command
from .tasks import task_create, task_get, task_list, task_update

# edit_file 和 write_file 标记 sequential=True：同一轮里的多个改文件调用必须串行执行，
# 否则它们会基于同一份旧快照并发写盘、互相覆盖（这正是 readFileState + mtime 想防住的并发问题）
# task_update 也标记 sequential=True：同一轮里连续改同一条 task，并发会基于同一份旧快照覆盖（store 没加锁）
TOOLS = [
    read_file,
    Tool(edit_file, sequential=True),
    Tool(write_file, sequential=True),
    run_command,
    task_create,
    task_list,
    task_get,
    Tool(task_update, sequential=True),
    ask_user_question,
]

__all__ = ["TOOLS", "read_and_register"]
