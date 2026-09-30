"""
长期记忆子包：prompt 指令（instructions）、文件存储（store）、按需召回（recall）、后台提炼 + dream 合并（background）。

直接从子模块导入（memory.store / memory.recall / ...）。这个 __init__ 故意不做任何重导出：
store 是连 permissions、agent.core 也依赖的最底层模块，在这里重导出 background 可能经由它们形成循环导入。
"""
