"""
system-reminder 正文构造。真正的注入在 agent/hooks.py 里挂 hook。
"""
from tasks_store import TasksStore
from html import escape
from background_jobs import JobRegistry

from .file_state import ReadFileState


def build_job_reminder_text(registry: JobRegistry | None) -> str | None:
    """领取已完成且未通知的后台 job；领取后标记，活跃/空闲链路不会重复发送。"""
    jobs = registry.pop_unnotified() if registry else []
    if not jobs:
        return None
    blocks = []
    for job in jobs:
        guidance = "。最终报告见 result，可用 read_file 查看日志。" if job.kind == "agent" else "。请读取日志了解结果，再继续原任务。"
        fields = {"task-id": job.id, "task-type": job.kind,
                  "output-file": str(job.log_path), "status": job.status,
                  "summary": job.summary() + guidance}
        if job.result is not None:
            fields["result"] = job.result
        # 命令里可能有 <、>，转义后不能伪造通知标签。
        body = "\n".join(f"<{key}>{escape(value)}</{key}>" for key, value in fields.items())
        blocks.append(f"<task-notification>\n{body}\n</task-notification>")
    return "\n".join(blocks)


def _wrap(lines: list[str]) -> str:
    # 所有 reminder 正文都包在 <system-reminder>...</system-reminder> 标签里，集中一处避免分散重复
    return f"<system-reminder>\n{chr(10).join(lines)}\n</system-reminder>"


def build_monitor_event_text(registry: JobRegistry | None) -> str | None:
    blocks = []
    for job, events, dropped in registry.pop_events() if registry else []:
        omitted = (f"<dropped>另有 {dropped} 条事件因为输出过快或队列已满被省略，"
                   "需要时用更精确的过滤条件重新挂载</dropped>\n") if dropped else ""
        blocks.append(f"<monitor-event>\n<task-id>{job.id}</task-id>\n"
                      f"<description>{escape(job.description, quote=False)}</description>\n"
                      f"<events>\n{escape(chr(10).join(events), quote=False)}\n</events>\n"
                      f"{omitted}</monitor-event>")
    return "\n\n".join(blocks) or None


def build_job_notifications(registry: JobRegistry | None) -> str | None:
    parts = [build_job_reminder_text(registry), build_monitor_event_text(registry)]
    return "\n\n".join(part for part in parts if part) or None


def build_reminder_text(state: ReadFileState) -> str | None:
    """
    根据当前会话状态拼出提醒正文；没什么值得提醒的就返回 None。
    """
    stale = state.stale_paths()
    if not stale:
        return None
    lines = [
        "以下文件在你读取之后被外部修改过，你 context 里的内容可能已过时，编辑前请重新用 read_file 读取：",
    ]
    lines += [f"- {path}" for path in stale]
    return _wrap(lines)


def build_task_reminder_text(store: TasksStore) -> str:
    """
    拼出 task 提醒的正文：一段温和的提醒 + 当前 task 列表。列表直接从 store 读，所以这条提醒同时承担了"把磁盘状态反向推回 prompt"的作用，模型不必主动 task_list。

    开头那句话同时被 hooks.py 用来扫历史识别"上一条已经是 reminder"，保持稳定不要改。
    """
    tasks = store.list()
    lines = [
        "task 工具最近没有被使用。如果你正在处理的工作适合用 task 跟踪进度，建议用 task_create 新建 task，用 task_update 维护状态（开工时切 in_progress，做完切 completed）；如果列表里有过时的 task，也可以顺手清理掉。只在与当前工作相关时再用这些工具。这只是一句友好的提醒——和当前工作无关的话忽略即可。",
    ]
    if tasks:
        lines.append("")
        lines.append("现存的 task 列表：")
        lines.append("")
        for t in tasks:
            lines.append(f"#{t.id}. [{t.status}] {t.subject}")
    return _wrap(lines)
