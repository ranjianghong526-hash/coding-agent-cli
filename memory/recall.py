"""
记忆召回：用户消息进入 Agent 循环之前，先单独发一次 LLM 请求从记忆清单里挑出相关记忆，包成 system-reminder 消息塞进对话历史。
"""
import json
import os

from openai import AsyncOpenAI
from pydantic_ai.messages import ModelRequest, UserPromptPart

import session
from ui.render import print_step

from . import store

# 和 classifier 一样是主对话之外单独发起的请求，不经过 agent 框架
_client = AsyncOpenAI(
    api_key=os.environ.get("API_KEY"),
    base_url="https://api.deepseek.com",
)

RECALL_MODEL = "deepseek-flash"

# 每轮最多召回几条记忆
MAX_RECALL = 5
# 输入太短（如「继续」「好的」）多半是承接上文，不值得为它发一次召回请求
MIN_INPUT_CHARS = 4

SYSTEM_PROMPT = """你在为一个 coding agent 挑选记忆，帮它处理用户的最新输入。用户消息会给你这条输入，和一份可用记忆的清单（每行一个记忆文件，带类型、日期和描述）。

返回对处理这条输入明确有用的记忆文件名列表（最多 5 个）。只选那些凭文件名和描述就能确定有帮助的记忆。
- 拿不准某条记忆有没有用，就不要把它放进列表。要挑剔。
- 清单里没有明确有用的记忆，返回空列表就好。

只输出 JSON：{"selected_memories": ["文件名", ...]}
"""


async def _select(user_input: str, headers: list) -> list[str]:
    """
    单独请求一次 LLM，挑出相关的文件名。fail-open 原则：召回只是锦上添花，出错就当这轮没有相关记忆。
    """
    manifest = store.format_manifest(headers)
    try:
        response = await _client.chat.completions.create(
            model=RECALL_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"最新输入：{user_input}\n\n记忆清单：\n{manifest}"},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        selected = json.loads(response.choices[0].message.content).get("selected_memories") or []
    except Exception:
        return []
    # 模型可能编造文件名，过滤成清单里真实存在的
    valid = {h.filename for h in headers}
    return [name for name in selected if isinstance(name, str) and name in valid][:MAX_RECALL]


def _build_reminder(header) -> str:
    """
    把一个记忆文件包成 system-reminder 正文：相对保存时间（超过一天换成新鲜度警告）+ 文件路径 + 内容。
    """
    content = header.path.read_text(encoding="utf-8")
    content = store.truncate_content(
        content, store.MAX_MEMORY_LINES, store.MAX_MEMORY_BYTES,
        f"这条记忆被截断了，用 read_file 查看完整文件：{header.path}",
    )
    days = store.age_days(header.mtime)
    if days <= 1:
        head = f"记忆（保存于{store.age_text(days)}）：{header.path}："
    else:
        # 旧记忆的开头换成新鲜度警告，提醒模型先验证再使用
        head = f"{store.staleness_text(days)}\n\n记忆：{header.path}："
    return f"<system-reminder>\n{head}\n\n{content}\n</system-reminder>"


async def inject_memories(user_input: str, state) -> None:
    """
    每条用户消息触发的召回入口：挑出记忆、包成消息塞进历史并持久化；本会话注入过的记忆不再重复注入。
    """
    if len(user_input.strip()) < MIN_INPUT_CHARS:
        return
    headers = [
        h for h in store.scan_memory_files()
        if h.filename not in state.surfaced_memories
    ]
    if not headers:
        return
    selected = await _select(user_input, headers)
    if not selected:
        return

    by_name = {h.filename: h for h in headers}
    injected = []
    for filename in selected:
        header = by_name[filename]
        try:
            text = _build_reminder(header)
        except OSError:
            continue
        injected.append(ModelRequest(parts=[UserPromptPart(content=text)]))
        state.surfaced_memories.add(filename)
        # 终端回显召回了哪条记忆，让用户看到召回确实发生了
        print_step("[dim]◇ memory[/]", f"[dim]召回记忆：{filename}（{store.age_text(store.age_days(header.mtime))}）[/]")
    if not injected:
        return

    # 和 @ 引用一样塞进历史并持久化，/resume 恢复会话时召回过的记忆也能一起还原
    state.history += injected
    session.append_messages(state.session_id, injected)
