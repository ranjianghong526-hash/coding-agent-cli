"""
终端 UI 层：渲染原语（render）、斜杠命令（commands），以及输入区（input_ui，部分版本才有）。

直接从子模块导入（ui.render / ui.commands / ...）。这个 __init__ 故意不做任何重导出：
render 是连非 UI 模块（如 permissions、agent.hooks）也依赖的最底层原语，在这里重导出 commands 可能经由它们形成循环导入。
"""
