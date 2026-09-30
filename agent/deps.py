"""
Agent 复合依赖：现在 agent 同时需要 ReadFileState（给文件工具用）和 TasksStore（给 task 工具用），把它们包成一个对象，统一通过 RunContext 注入。
"""
from dataclasses import dataclass

from file_history import FileHistory
from tasks_store import TasksStore
from background_jobs import JobRegistry

from .file_state import ReadFileState


@dataclass
class AgentDeps:
    read_file_state: ReadFileState
    # 后台记忆 agent 的 fork 只挂文件工具，不需要 tasksStore，允许为 None
    tasks_store: TasksStore | None
    # 文件检查点，写文件的工具用它备份改动前内容；默认 None，后台记忆 fork 写的记忆文件不进回退范围
    file_history: FileHistory | None = None
    # 后台记忆 fork 不执行命令，因此可以不挂注册表。
    job_registry: JobRegistry | None = None
