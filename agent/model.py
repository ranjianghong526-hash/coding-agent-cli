"""主 Agent 和 sub agent 共享模型配置；独立模块避免工具注册时循环导入 core。"""
import os

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    raise RuntimeError("请先设置环境变量 API_KEY")
MODEL_NAME = "deepseek-flash"
model = OpenAIChatModel(MODEL_NAME, provider=DeepSeekProvider(api_key=API_KEY))
