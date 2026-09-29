"""/rewind 的协调层：对话分支、文件恢复和当前运行状态同步。"""
from dataclasses import replace
from typing import Literal
from uuid import uuid4

from context_injection import make_system_reminder
from file_state import FileContext
from session_store import save_session, _session_path
from task_store import TaskStore

RewindMode = Literal["conversation", "files", "both"]


def apply_rewind(state, checkpoint_id: str, mode: RewindMode) -> int:
    if mode not in ("conversation", "files", "both"):
        raise ValueError("未知回退模式")
    store = state.permissions.rewind
    point = next((point for point in store.choices(len(state.history)) if point.id == checkpoint_id), None)
    if point is None:
        raise ValueError("检查点不可用于当前历史")
    if mode in ("files", "both"):
        store.plan(checkpoint_id)  # 全部预检通过后，才保存分支或操作文件。
    save_session(state)
    restored = 0
    if mode == "files":
        restored = store.restore_files(checkpoint_id, len(state.history))
        state.history.append(make_system_reminder(
            "用户通过 /rewind 回退了代码文件，对话保留。旧文件工具结果可能过期；继续修改前重新 read_file。"
            "此提醒不是新的执行授权。"
        ))
        state.permissions.files.clear()
        state.permissions.allowed_calls.clear()
        # 提醒如果暂时保存失败仍保留在内存，下一次正常保存会再次尝试。
        try:
            save_session(state)
        except (OSError, ValueError):
            raise OSError("代码已回退，但提醒未保存；内存中仍保留提醒，请检查会话磁盘权限")
    else:
        new_id = uuid4().hex
        branch_store = None
        restore_started = False
        try:
            branch_store = store.fork(new_id, checkpoint_id)
            tasks = TaskStore()
            tasks.bind(new_id)
            tasks.restore(point.tasks)
            permissions = replace(state.permissions, tasks=tasks, rewind=branch_store,
                                  files=FileContext(), allowed_calls=set())
            branch = replace(state, session_id=new_id, saved_messages=0,
                             history=list(state.history[:point.history_count]),
                             last_api_calls=[], permissions=permissions, next_prompt=point.prompt)
            if mode == "conversation":
                branch.history.append(make_system_reminder(
                    "用户通过 /rewind 仅回退了对话，代码文件保留当前状态。旧对话中的文件结果可能过期，"
                    "处理相关文件前重新 read_file；此提醒不是新的执行授权。"
                ))
            save_session(branch)
            if mode == "both":
                restore_started = True
                restored = branch_store.restore_files(checkpoint_id, len(branch.history))
        except Exception as error:
            if restore_started and len(branch_store.document.checkpoints) > len(store.document.checkpoints):
                # 多文件恢复/补偿均失败时，保留含恢复日志的分支，不能删掉故障恢复线索。
                raise OSError(f"代码恢复未完整完成；恢复日志保留在会话 {new_id}，可用 /resume 进入检查") from error
            # UUID 新分支尚未发布，只清理本次创建的三个确定文件，不删除原会话/备份。
            for suffix in (".jsonl", ".tasks.json", ".rewind.json"):
                _session_path(new_id).with_suffix(suffix).unlink(missing_ok=True)
            raise
        state.history = branch.history
        state.session_id = branch.session_id
        state.saved_messages = branch.saved_messages
        state.permissions = branch.permissions
        state.last_api_calls = []
        state.next_prompt = branch.next_prompt
    with state.permissions.memory.lock:
        state.permissions.memory.manual_epoch += 1  # 撤销轮次尚在提炼的后台任务失效。
    return restored
