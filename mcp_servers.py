"""MCP 注册表：沿用参考项目的 RECORDS / startup / shutdown / active_toolsets。"""
import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path

from fastmcp.client.transports import StdioTransport
from pydantic_ai.mcp import MCPToolset, load_mcp_toolsets
from pydantic_ai.toolsets import AbstractToolset

USER_CONFIG = Path.home() / ".my-claude-code" / "mcp.json"
PROJECT_CONFIG = Path(".mcp.json")
LOG_DIR = Path.home() / ".my-claude-code" / "mcp-logs"


@dataclass
class ServerRecord:
    server: MCPToolset
    transport: str = ""
    status: str = "pending"
    error: str = ""
    tool_names: list[str] = field(default_factory=list)
    # 当前 SDK 用 PrefixedToolset 包装名称，原 server 留着管理连接、查询工具。
    toolset: AbstractToolset | None = field(default=None, repr=False)


RECORDS: list[ServerRecord] = []
_stack = AsyncExitStack()


def load_servers() -> None:
    """SDK 解析配置和环境变量；先用户、后项目，按同名 id 整条覆盖。"""
    merged = {}
    for config_path in (USER_CONFIG, PROJECT_CONFIG):
        if config_path.is_file():
            for wrapped in load_mcp_toolsets(config_path):
                original = wrapped.wrapped
                transport = original.client.transport
                if isinstance(transport, StdioTransport):
                    # 新 SDK 的原生 log_file 对应旧 QuietStdioServer 的 stderr 重定向。
                    transport.log_file = LOG_DIR / f"{original.id}.log"
                    transport.keep_alive = False
                    if ("/" in transport.command or "\\" in transport.command) and not Path(transport.command).is_absolute():
                        transport.command = str((Path(transport.cwd or Path.cwd()) / transport.command).resolve())
                # 重新构造以使 30s 初始化/读取超时都进入实际客户端，uvx/npx 冷启动不再被 5s 截断。
                server = MCPToolset(transport, id=original.id, include_instructions=True,
                                    init_timeout=30, read_timeout=30)
                merged[server.id] = server
    RECORDS.clear()
    for server in merged.values():
        RECORDS.append(ServerRecord(server=server, transport=_transport(server),
                                    toolset=server.prefixed(f"mcp__{server.id}_")))


def _transport(server: MCPToolset) -> str:
    transport = server.client.transport
    if isinstance(transport, StdioTransport):
        return "stdio: " + " ".join([transport.command, *transport.args])
    return f"http: {transport.url}"


async def startup() -> str:
    """并发连接，单个失败记入记录；返回与参考项目相同的摘要。"""
    load_servers()
    tasks = [asyncio.create_task(_connect(record)) for record in RECORDS]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        # 取消时先收完所有连接任务，再统一断开，避免资源登记发生在关闭之后。
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await shutdown()
        raise
    connected = [r for r in RECORDS if r.status == "connected"]
    failed = [r for r in RECORDS if r.status == "failed"]
    parts = []
    if connected:
        parts.append("已连接 MCP server：" + "、".join(
            f"{r.server.id}（{len(r.tool_names)} 个工具）" for r in connected))
    if failed:
        parts.append("连接失败：" + "、".join(r.server.id for r in failed) + "（详情见 /mcp）")
    return "；".join(parts)


async def _connect(record: ServerRecord) -> None:
    try:
        if isinstance(record.server.client.transport, StdioTransport):
            LOG_DIR.mkdir(parents=True, exist_ok=True)
        await _stack.enter_async_context(record.server)
        tools = await record.server.list_tools()
        record.tool_names = [f"mcp__{record.server.id}__{tool.name}" for tool in tools]
        record.status = "connected"
    except Exception as error:
        # 和参考项目一样，展开 ExceptionGroup，显示底层真正的错误。
        while getattr(error, "exceptions", None):
            error = error.exceptions[0]
        record.status = "failed"
        record.error = f"{type(error).__name__}: {error}" if str(error) else type(error).__name__


async def shutdown() -> None:
    """统一释放；子进程已经退出时的清理异常不影响 CLI 退出。"""
    try:
        await _stack.aclose()
    except Exception:
        pass


def active_toolsets() -> list[AbstractToolset]:
    return [record.toolset for record in RECORDS if record.status == "connected"]
