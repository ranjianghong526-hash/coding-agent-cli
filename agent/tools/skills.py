"""技能正文加载工具；引用的资料与脚本仍使用原来的文件、命令工具。"""
from skills import read_skill


def load_skill(name: str) -> str:
    """当可用 skill 的描述匹配当前任务后，读取它的完整指令。

    返回根目录与正文；相对路径的资料/脚本按这个根目录定位，确实需要时
    再用 read_file 或 run_command，不要一次性读取全部资源。

    Args:
        name: 可用 skills 清单里的准确名称
    """
    return read_skill(name)
