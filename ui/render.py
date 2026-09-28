"""
共享的终端展示基础组件，只依赖 Rich，当前由 ui.commands 使用。

这里处理颜色、缩进和边框，不处理模型、工具或会话状态。
把简单的展示函数独立出来，使上层 commands 可以复用它们且不会形成反向依赖。
"""
# Console 是终端输出对象；Padding 控制留白；Panel 提供边框；Text 拼接不同样式。
from rich.console import Console
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text


# 共享的 Console 实例，跨平台自动处理 ANSI 颜色（Windows 老终端也能正常显示）。
# highlight=False 关闭 Rich 自带的数字/字符串自动高亮，只让我们手动加的 markup 生效。
console = Console(highlight=False)


def print_step(label: str, content: str = "") -> None:
    """
    打印一个中间过程 block：标签独占一行，内容用 Padding 左缩进 2 格
    （这样自动折行的续行也能保持缩进），末尾留一个空行。
    """
    console.print(label)
    if content:
        # 四元组依次为上、右、下、左留白；左侧 2 格也会应用到自动折行的续行。
        console.print(Padding(content, (0, 0, 0, 2)))
    # 块之间保留空行，方便区分一次工具调用和紧随其后的工具返回。
    console.print()


def print_welcome_banner(title: str) -> None:
    """
    打印启动欢迎横幅：标题装在带色彩的 rich.Panel 里，附 /help 提示。
    """
    # Text.assemble 接收“文本、样式”元组，将多个样式片段组成一个对象。
    # 这里的 /help 仅是提示文字，实际命令处理在 commands.py 中。
    body = Text.assemble(
        ("✻ ", "bright_magenta"),
        (f"欢迎使用 {title}\n\n", "bold"),
        ("  /help ", "cyan"),
        ("查看可用命令", "dim"),
    )
    # Panel 的 padding 二元组表示垂直与水平留白，width 指定面板宽度。
    console.print(Panel(body, border_style="bright_blue", padding=(0, 1), width=48))
    console.print()
