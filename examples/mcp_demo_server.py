"""无外部账号的 MCP 演示服务器：启动后提供 add 工具，用于体验真实通信。

SDK 根据函数签名生成 inputSchema；stdio 的 stdout 用于协议，日志只能写 stderr。
"""
from mcp.server.mcpserver import MCPServer

server = MCPServer("coding-agent-demo", instructions="这是算术演示服务器。add 用于整数加法，不操作项目文件。")


@server.tool()
def add(a: int, b: int) -> int:
    """计算两个整数的和。"""
    return a + b


if __name__ == "__main__":
    server.run(transport="stdio")
