"""
Agent 实例化：把 model / instructions / tools / hooks 拼起来，再加上动态 instructions 注入环境信息、AGENTS.md 和记忆索引。
"""
import os
import platform
from datetime import date

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

from memory import store
from memory.instructions import MEMORY_INSTRUCTIONS

from .deps import AgentDeps
from .hooks import hooks
from .tools import TOOLS

# 从环境变量读取 API Key
API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    raise RuntimeError("请先设置环境变量 API_KEY")

MODEL_NAME = "deepseek-flash"

model = OpenAIChatModel(
    MODEL_NAME,
    provider=DeepSeekProvider(api_key=API_KEY),
)

# 静态 instructions 抽成常量，后台记忆 agent 直接复用，保证两边看到同一份约定
INSTRUCTIONS = (
    "你是一个编程助手。你可以读写文件和执行命令来帮用户完成编程任务。\n"
    "修改已有文件前必须先用 read_file 读取它。"
    "改动局部内容时优先用 edit_file（只传改动的片段，省 token），新建文件或整体重写才用 write_file。\n"
    "工作流程：先理解需求，写代码，然后运行验证。"
    "长驻或耗时命令（dev server、长测试）用 run_command 的 run_in_background=True 放到后台，"
    "只在不需要立刻拿到结果时使用，命令末尾不需要加 &。"
    "后台 job 结束会收到 <task-notification>，不要主动轮询等待；"
    "期间可以用 read_file 读日志查看输出，用 job_kill 提前终止。"
    "如果有错误就修复并重新运行，直到确认正确。\n"
    "当你接到一个需要 3 步以上、或需要多次工具调用才能完成的任务时，"
    "先用 task_create 把分解出来的步骤建成 pending task，开工前用 task_update 把要做的那条切到 in_progress，做完立刻切 completed。"
    "如果是琐碎请求（1-2 步、纯对话、纯查询），不要建 task，建了反而碍事。做完的 task 不要让它一直挂在 in_progress。\n"
    "如果用户的需求里有歧义、有多种合理实现可选、或者你拿不准方向，"
    "应当用 ask_user_question 工具向用户提多选题来澄清，不要自作主张。\n"
    "对话中可能会出现 <system-reminder>...</system-reminder> 标签，里面是系统自动注入的提示信息，请按系统消息对待，不要把它当成它所在的用户消息或工具结果的一部分。"
) + MEMORY_INSTRUCTIONS

agent = Agent(
    model,
    instructions=INSTRUCTIONS,
    tools=TOOLS,
    # 声明依赖类型为 AgentDeps，里面同时持有 readFileState 和 tasksStore，每次 run 注入给工具
    deps_type=AgentDeps,
    capabilities=[hooks],
)


# 动态 instructions：每次模型请求重新求值，注入环境信息和项目级 AGENTS.md
# 它和上面的静态 instructions 一样不进对话历史，但每轮请求都会带上最新值，所以用户不可见、也不会污染历史
@agent.instructions
def project_context() -> str:
    cwd = os.getcwd()
    parts = [
        "下面是一些环境信息：",
        f"- 工作目录：{cwd}",
        f"- 操作系统：{platform.system()}",
        f"- 今天的日期：{date.today().isoformat()}",
    ]

    # 读项目根目录的 AGENTS.md（项目指令），存在就整段附上，让模型遵循项目约定
    agents_md = os.path.join(cwd, "AGENTS.md")
    if os.path.isfile(agents_md):
        try:
            content = open(agents_md, encoding="utf-8").read()
        except OSError:
            content = ""
        if content.strip():
            parts.append("")
            parts.append("以下是项目的 AGENTS.md，请遵循其中的约定：")
            parts.append(content)

    # 注入记忆索引：MEMORY.md 每轮全量注入，模型随时知道自己手头有哪些主题的记忆
    # 走动态 instructions 的好处是每轮拿到最新值——本轮刚写的记忆，下一轮索引里就有
    index = store.read_index()
    if index:
        parts.append("")
        parts.append(f"以下是你的记忆索引 MEMORY.md（位于 {store.index_path()}，跨会话保留），需要某条记忆的完整内容就用 read_file 读记忆目录下的对应文件：")
        parts.append(index)

    return "\n".join(parts)
