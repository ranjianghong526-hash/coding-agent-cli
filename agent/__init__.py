"""Coding Agent 完整封装包。

外部代码只关心：
- agent: 配置好工具和 hooks 的 Agent 实例
- MODEL_NAME: 当前使用的模型名（/status 命令要用）
- api_call_log: 主循环跑 Agent 时收集的 API 调用元数据（/api-detail 命令要用）
- ApiCall: 一次 API 调用的数据结构

子模块（tools / hooks / core）是实现细节，不需要直接 import。
"""
from .core import agent, MODEL_NAME
from .hooks import api_call_log, ApiCall

__all__ = ["agent", "MODEL_NAME", "api_call_log", "ApiCall"]
