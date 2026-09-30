# My Claude Code CLI

一个 Python 编程 Agent CLI，整体实现已对齐本地 `my-claude-code-mcp-完整项目`。支持文件读写与精确编辑、命令执行、工具审批、自动审查、任务管理、长期记忆、会话恢复、检查点回退、上下文压缩和 MCP。

详细对照见 [参考项目对齐说明](参考项目对齐说明.md)，源码阅读见 [项目阅读路线](项目阅读路线.md)。

## 启动

```powershell
cd "C:\Users\A1455\Desktop\实习\coding-agent-cli"
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

当前虚拟环境没有 pip 时，可以使用：

```powershell
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
```

在项目根目录 `.env` 中设置 `API_KEY`。模型与参考项目相同：DeepSeek 的 `deepseek-flash`。已有 `.env` 保留；系统环境变量优先。不要提交真实密钥。

## 交互

- 输入框常驻屏幕底部，输出显示在上方；有任务时显示任务面板。
- 请求过程中 Esc/Ctrl+C 中断当前任务；空闲时 Ctrl+C 退出，Esc 清空输入。
- Shift+Tab 切换 `default → acceptEdits → auto → bypass`。
- Ctrl+B 将正在等待的前台命令转后台，保留原进程；底部显示运行中的后台 job 数量。
- 输入 `@main.py` 会补全路径，提交后自动注入文件内容和读取记录。
- 权限审批改为方向键选择「允许」「允许且本会话不再询问该工具」「拒绝」，Enter 确认。按工具名记住整场会话的授权，/new、/resume 清空白名单。
- 需求不明确时，模型可调用 ask_user_question，支持单选、多选和自定义文本。

| 命令 | 功能 |
|---|---|
| /help | 可用命令 |
| /new | 新会话 |
| /resume | 选择历史会话并恢复 |
| /rewind | 回退对话、文件或两者 |
| /compact [补充指令] | 压缩历史为摘要，并恢复最近文件 |
| /status | 会话、模型、权限、上下文水位和 token |
| /mcp | MCP 连接状态与工具 |
| /jobs | 当前会话的命令、状态和日志路径 |
| /memory | 查看记忆文件与索引 |
| /dream | 立即整理合并记忆 |
| /api-detail | 最近一轮模型请求摘要 |
| /exit | 退出 |

## 工具和权限

本地工具：read_file、edit_file、write_file、run_command、job_kill、task_create、task_list、task_get、task_update、ask_user_question。

已有文件修改前必须读取。read_file 支持 offset/limit 和行号；edit_file 精确匹配原文，不要把行号写入 old_string。登记的 mtime 过期时要求重读。写文件和任务更新通过 sequential=True 串行执行。

default 自动放行读取、任务和提问，编辑/命令审批；acceptEdits 自动放行编辑；auto 同样自动放行编辑，其余待审批操作交给旁路分类器；bypass 全部放行。高危 shell 自检在非 bypass 模式下优先于会话白名单。MCP 名称不命中本地只读白名单，仍走同一权限 hook。

## MCP

项目 `.mcp.json` 与用户 `~/.my-claude-code/mcp.json` 合并，同名服务器项目优先。启动并发连接，初始化与读取超时 30 秒；连接全程复用，退出统一关闭。stdio stderr 追加到 `~/.my-claude-code/mcp-logs/<服务器名>.log`。

本地演示配置已准备。克隆仓库后，没有现有配置时可以复制：

```powershell
Copy-Item .mcp.json.example .mcp.json
```

```json
{
  "mcpServers": {
    "demo": {
      "command": ".venv/Scripts/python.exe",
      "args": ["examples/mcp_demo_server.py"]
    },
    "context7": {
      "url": "https://mcp.context7.com/mcp"
    }
  }
}
```

配置支持 `${VAR}` 和 `${VAR:-默认值}`。修改后重启生效。运行 `/mcp`，再要求「用 demo 的 add 计算 2+3」即可体验。Context7 是远程示例，需要可用网络。

工具以 `mcp__服务器__工具` 注册给模型，调用时 SDK 还原原名。服务器说明自动注入模型指令。配置中的服务器启动命令会在启动时运行，工具审批约束后续工具调用，不是子进程沙箱。外部 MCP/命令修改不在本地文件回退跟踪范围。

当前 SDK 使用 MCPToolset/load_mcp_toolsets，参考源码的旧接口已做适配。真实 stdio 已测试，远程 HTTP 账号服务尚未验证。

## 后台命令

按照「让 Agent 在后台运行命令」文章实现。对 Agent 说：

```text
请在后台运行 .venv\Scripts\python.exe -u -c "import time; print('START'); time.sleep(15); print('DONE')"，完成后读取日志并告诉我结果。
```

模型调用 `run_command(command=..., run_in_background=True)`，正常经过权限审批，然后立即获得 job ID 和日志路径。命令持续运行，stdout/stderr 合并写入日志。前台命令仍有 10 秒超时，等待期间可按 Ctrl+B 转后台。Python 子进程默认使用 UTF-8 输出；其他程序的日志编码由该程序决定。

`/jobs` 查看当前会话所有命令；让模型通过 `read_file` 看日志，或 `job_kill(job_id=...)` 停止自己启动的命令。job_kill 不接受任意 PID。

命令退出时 watcher 更新状态。模型忙时，before_model_request hook 注入 `<task-notification>`；模型空闲时，本地每秒扫描并自动开启一轮处理通知。扫描本身不请求模型。通知包含 ID、状态、日志路径和摘要，完整输出需读取日志。通知使用独立来源标记，不作为 auto 分类器的用户授权，不触发用户输入的检查点、记忆召回和记忆提炼流程。

`pop_unnotified()` 领取时标记 notified=True，两条链路不会重复发送同一个通知。常驻服务没有结束就不会发完成通知，可随时查看日志或停止。`/new`、`/resume`、退出都会终止旧会话进程树；日志保留，恢复会话只恢复聊天，不重启旧命令。Windows 用 taskkill /T /F，POSIX 使用独立进程组。

## 数据

```text
~/.my-claude-code/
├── mcp.json
├── mcp-logs/
├── jobs/<session_id>/<job_id>.log # 命令输出；注册表和进程状态只在内存中
├── tasks/<session_id>/           # 每条 task 一个 JSON
└── projects/<项目路径编码>/
    ├── <session_id>.jsonl        # 每行一条 SDK 消息
    ├── file-history/<session_id>/
    ├── compact-history/         # 完整对话存档，/resume 不扫描子目录
    └── memory/                  # MEMORY.md + 带 frontmatter 的 Markdown
```

长期记忆通过普通文件工具保存，模型更新正文和 MEMORY.md 索引；旁路模型挑选相关记忆注入 system-reminder，本会话去重；后台 fork 只允许写记忆目录，完成后可定期 dream 合并。主模型不再使用旧版 memory_write 等专用工具。

/compact 使用同一会话 ID，先存档，再用摘要和最近文件重写有效历史，丢弃旧检查点；连续 3 次自动压缩失败后停止自动尝试。

首次启动自动导入旧 `.sessions/` 和 `.memory/`，源文件保留，已有目标不覆盖。旧压缩记录只恢复最后一段有效历史，完整原始记录保留为存档。无法转换的数据会显示文件名和错误类型，原数据仍在。

## 测试

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -q
```

覆盖文件读取/编辑/过期校验、任务持久化、权限、序列化、回退、@ 引用、提醒阈值、压缩、记忆召回与后台闸门、重试、提问 picker、常驻输入键盘提交、中断、真实 MCP、并发连接、退出清理和旧数据迁移。模型均为模拟，不消耗真实模型 API。

新增 test_background_jobs.py，验证真实命令日志、退出码、前台超时/取消、后台通知去重、SDK 工具 schema 与权限、忙时提醒、空闲自动续跑、真实 Ctrl+B、会话切换与 Windows 子进程树清理。当前整套 34 项测试通过。

改动前源码和旧接口测试在 `.reference-migration-backup/before-reference-alignment/`。目录被 Git 忽略；当前测试针对对齐后的接口。
