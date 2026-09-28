"""
终端 UI 包：render 提供基础展示组件，commands 维护会话、处理命令并展示消息。

本文件标识 ui 是一个普通 Python 包，没有初始化业务或重导出其他对象。
调用方直接从 ui.render 或 ui.commands 导入所需函数，方便看清依赖来自哪里。
当前依赖方向为 commands -> render，render 不反过来导入 commands。
"""
