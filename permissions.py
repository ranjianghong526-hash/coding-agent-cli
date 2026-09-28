"""工具执行前的人工审批：权限规则和临时授权只在本次程序运行中生效。"""
import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from prompt_toolkit import PromptSession
from pydantic_ai.exceptions import SkipToolExecution
from pydantic_ai.messages import ModelMessage

from classifier import classify

from ui.render import console

PermissionMode = Literal["default", "acceptEdits", "auto", "bypass"]
MODES: tuple[PermissionMode, ...] = ("default", "acceptEdits", "auto", "bypass")


@dataclass
class PermissionState:
    """与对话历史分开保存；恢复历史不等于恢复过去的执行授权。"""
    mode: PermissionMode = "default"
    # 授权范围是工具名和完整参数，修改任意参数后需要重新确认。
    allowed_calls: set[tuple[str, str]] = field(default_factory=set)
    # 模型可能一次返回多个工具调用，锁保证终端只有一个审批问题。
    approval_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def cycle_mode(self) -> None:
        """Shift+Tab 按固定顺序切换模式，不影响正在编辑的需求。"""
        self.mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]


def requires_approval(mode: PermissionMode, tool_name: str) -> bool:
    """只读工具明确列入白名单，未知工具默认需要审批。"""
    if mode not in MODES:
        raise ValueError("未知权限模式")
    if mode == "bypass" or tool_name == "read_file":
        return False
    return not (mode == "acceptEdits" and tool_name == "write_file")


async def ask_permission(tool_name: str, args: dict[str, Any]) -> tuple[bool, bool, str]:
    """返回允许、是否记住和拒绝说明；空输入或输入中断都按拒绝处理。"""
    console.print(f"工具执行需要确认：{tool_name}", style="yellow", markup=False)
    # 完整展示实际参数，不截断命令或文件内容，也不把参数解释为 Rich 标签。
    console.print(json.dumps(args, ensure_ascii=False, indent=2), markup=False)
    console.print("y / 1：允许一次；a / 2：本次运行允许相同工具及参数；n / 3：拒绝。")
    console.print("拒绝时可附说明，例如：n 请先读取文件，不要覆盖。")
    session = PromptSession()
    while True:
        try:
            answer = (await session.prompt_async("审批 [默认拒绝] ❯ ")).strip()
        except (EOFError, KeyboardInterrupt):
            return False, False, "用户取消了审批"
        choice, _, reason = answer.partition(" ")
        if choice.lower() in ("y", "yes", "1", "a", "2") and not reason:
            return True, choice.lower() in ("a", "2"), ""
        if not answer or choice.lower() in ("n", "no", "3"):
            return False, False, reason.strip() or "用户拒绝了本次操作"
        console.print("请输入 y、a 或 n，也可以在 n 后填写拒绝说明。")


async def check_permission(
    state: PermissionState, tool_name: str, args: dict[str, Any],
    messages: Sequence[ModelMessage] = (),
) -> None:
    """允许则返回；拒绝则跳过真实工具，并把明确的拒绝结果交给模型。"""
    async with state.approval_lock:
        if not requires_approval(state.mode, tool_name):
            return
        key = (tool_name, json.dumps(args, ensure_ascii=False, sort_keys=True))
        # 在锁内再检查授权，前一个并发审批刚记住的许可可以被后一个使用。
        if key in state.allowed_calls:
            return
        if state.mode == "auto":
            console.print(f"✻ auto 正在审查 {tool_name}…", style="dim", markup=False)
            verdict = await classify(messages, tool_name, args)
            console.print(f"auto 审查：{verdict['reason']}", markup=False)
            if verdict["should_block"] is False:
                # 自动裁决只对当前调用有效；不记入 allowed_calls，下一次重新看上下文。
                return
            # 拦截表示停止自动允许，仍由已有人工审批给出最终决定。
        allowed, remember, reason = await ask_permission(tool_name, args)
        if not allowed:
            console.print(f"已拒绝 {tool_name}，工具未执行。", style="yellow", markup=False)
            # 不使用 ModelRetry：用户拒绝不是参数错误，不应消耗工具修正预算。
            raise SkipToolExecution(
                f"[权限拒绝] 工具 {tool_name} 未执行。{reason}。"
                "请遵循用户说明，不要换工具或命令绕过拒绝；需要时向用户说明。"
            )
        if remember:
            state.allowed_calls.add(key)
