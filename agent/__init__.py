"""Coding Agent 完整封装包。

外部代码只关心：
- agent: 配置好工具和 hooks 的 Agent 实例
- MODEL_NAME: 当前使用的模型名（/status 命令要用）
- api_call_log: 主循环跑 Agent 时收集的 API 调用元数据（/api-detail 命令要用）
- ApiCall: 一次 API 调用的数据结构

子模块（tools / hooks / core）是实现细节，不需要直接 import。
"""
# __init__.py 定义包对外提供哪些对象；主程序不必了解子模块的组装细节。
# 注意这个导入会执行 core.py 的顶层代码，包括 .env 加载和 API_KEY 校验。
from .core import agent, MODEL_NAME
from .hooks import api_call_log, ApiCall

# __all__ 约定公开接口，并控制 from agent import * 的导入范围。
# 它不是访问权限控制，调用方仍可以显式导入 agent.tools 等子模块。
__all__ = ["agent", "MODEL_NAME", "api_call_log", "ApiCall"]
