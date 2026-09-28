"""
Agent 组装层：把模型连接、工作要求、工具和调用记录功能拼成一个实例。

本模块在 import 时执行配置，因此 .env 和 API_KEY 必须在创建模型前准备好。
它只配置 Agent；真正开始处理用户需求的是 main.py 中驱动 agent.iter() 的 run_agent()。
"""
import os
from pathlib import Path

# python-dotenv 将 .env 文件里的键值加载进进程环境变量。
from dotenv import load_dotenv
# Agent 是协调模型和工具的框架对象；模型适配器负责组织模型接口请求。
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

# 点号表示从当前 agent 包内导入，而不是寻找顶层同名模块。
from .hooks import hooks
from .tools import TOOLS

# 固定读取项目根目录的 .env，保留已设置的系统环境变量。
# __file__ 是当前文件路径；两次 parent 从 agent/core.py 回到项目根目录。
# 使用绝对路径，避免从不同工作目录启动时误读其他目录的配置。
# load_dotenv 默认不覆盖已有变量：系统 API_KEY 的优先级高于文件中的值。
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    # 文件不存在、变量未设置或值为空都会在这里提前失败，避免发出无效请求。
    raise RuntimeError("请在项目根目录的 .env 中填写 API_KEY，或设置环境变量 API_KEY")

# 模型名集中定义，同时供模型实例和 /status 的显示使用。
MODEL_NAME = "deepseek-flash"

# OpenAIChatModel 是接口格式适配器，实际服务提供方由 DeepSeekProvider 指定。
# 这里创建配置对象，不等于已经发送了用户需求。
model = OpenAIChatModel(
    MODEL_NAME,
    provider=DeepSeekProvider(api_key=API_KEY),
)

# 同一个 Agent 实例可以执行多轮任务；对话历史由 main.py 显式传入。
agent = Agent(
    model,
    # 这是发给模型的工作要求，不是 Python 层面的强制验证机制。
    # 相邻字符串会由 Python 自动拼接成一段完整文本。
    instructions=(
        "你是一个编程助手。你可以读写文件和执行命令来帮用户完成编程任务。\n"
        "工作流程：先理解需求，写代码，然后运行验证。"
        "如果有错误就修复并重新运行，直到确认正确。"
    ),
    # 注册后，框架允许模型选择工具并把参数映射为 Python 函数调用。
    tools=TOOLS,
    # hooks 在每次模型请求前后记账，不负责执行文件或命令操作。
    capabilities=[hooks],
)
