# My Claude Code CLI

一个 Python 编程 Agent CLI，支持文件读写与精确编辑、命令执行、工具审批、自动审查、任务管理、长期记忆、会话恢复、检查点回退、上下文压缩、后台子 Agent 和 MCP。

源码阅读见 [项目阅读路线](项目阅读路线.md)。

## 启动

以下为 Windows PowerShell 命令，每行分别执行。先在项目文件夹的上一级目录打开终端；如果已经位于项目根目录，跳过 `cd`。首次使用时创建虚拟环境，已有 `.venv` 则跳过创建步骤。

```powershell
cd .\coding-agent-cli
python -m venv .venv
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
| /agents | 内置和自定义子 Agent 类型 |
| /memory | 查看记忆文件与索引 |
| /dream | 立即整理合并记忆 |
| /api-detail | 最近一轮模型请求摘要 |
| /exit | 退出 |

## 工具和权限

本地工具：read_file、edit_file、write_file、run_command、job_kill、run_agent、monitor、task_create、task_list、task_get、task_update、ask_user_question。

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

## 子 Agent

实现依据为「后台运行 sub agent」文章。可要求「派一个 explore 子 Agent 调查项目结构」或「派两个子 Agent 分别调查模块划分和 TODO」。主模型通过 run_agent(description, prompt, agent_type) 委派；工具立即返回 a 开头的 job ID，子 Agent 在独立上下文工作，最终报告通过 task-notification 的 result 字段送回主模型，再由主模型转述。

内置 explore 使用 read_file/run_command，指令要求只读调查；general 可额外使用 edit_file/write_file。子 Agent 不带主会话历史，不带记忆索引、任务面板或 MCP 工具，不能再调用 run_agent、ask_user_question、task_*、job_kill。与主 Agent 使用同一模型配置，最多 40 次模型请求。

项目级 `.my-claude-code/agents/*.md` 定义自定义类型，启动时加载。name/description 必填，tools 是逗号分隔的白名单，省略时给四个允许的工具，正文作为独立指令。已提供 reviewer.md 示例；改完定义重启后用 `/agents` 查看。explore/reviewer 的只读约束来自工具裁剪和指令，run_command 本身仍是 shell 工具，并不是文件系统沙箱。

子 Agent 中间消息写入 `jobs/<session_id>/<agent_id>.log`，它启动的命令日志在 `jobs/<session_id>/<agent_id>/<shell_id>.log`。主模型默认只收到报告，按需可读日志。底部区分 shell/agent 数量，`/jobs` 和 job_kill 同样适用；结束或被取消时清理它的命令进程。

权限下沉到子 Agent 的每次工具调用：规则放行直接执行，auto 用分类器，人工审批进入队列。子 Agent await Future 暂停，主界面空闲时才弹出注明来源的审批框。选择 always 共享会话工具白名单，拒绝会作为工具结果让子 Agent在报告中说明。审批期间不会接受新一轮输入或投递完成通知。auto 审查参考真实父会话授权，不把主模型生成的委派 prompt 当成人类授权。

文件读取状态独立，file_history 继承主会话，因此文件工具的修改可回退；shell 命令修改文件仍不在回退追踪范围内。并发 general Agent 应分配不同文件，避免交叉编辑。

子 Agent 的执行和审批链路可从 [subagents.py](subagents.py) 与 [run_agent 工具](agent/tools/agents.py) 阅读。

## 持续事件监控

后台命令等结束才通知一次，monitor 在运行期间持续将非空输出行推送给主模型，适合“每次出现 ERROR 都告诉我”。monitor 创建真实进程，沿用命令权限审批和高危自检，不属于只读白名单，也不开放给子 Agent。

Windows 示例：先在项目目录创建 app.log，然后向 Agent 输入：

```text
用 monitor 运行 .venv\Scripts\python.exe -u examples/watch_log.py app.log --match ERROR，持续监听新出现的 ERROR，出现时告诉我。timeout 设置为 300 秒。
```

在另一个位于同一项目目录的 PowerShell 终端追加日志：

```powershell
Add-Content -Path app.log -Value "ERROR database connection refused" -Encoding utf8
```

初次监听跳过现有内容，只输出新行；示例脚本支持日志被截短或替换。命令自行决定过滤哪些内容，monitor 将它实际输出的非空行作为事件。Linux/macOS 可使用 `tail -n 0 -F app.log | grep --line-buffered ERROR`。管道每级应及时 flush，stderr 也要过滤时使用 `2>&1`，过滤条件要考虑失败信号。

默认超时 300 秒，persistent=True 不设超时；job_kill、会话切换和退出仍会清理。状态栏显示 monitor 数量，/jobs 显示 m 开头的 ID。事件先落盘，再入队，忙时在下一次请求注入，空闲时自动唤醒主模型；取出即清空，避免重复。事件不是用户回复，也不作为 auto 审查的用户授权。

每个 monitor 固定 10 秒窗口最多接收 20 条，队列最多 200 条，单条事件截断到 500 字符；超额事件计数，通知携带 dropped。累计丢弃 100 条会自动停止。超时、退出、手动停止和过长行都通过事件通道说明，不再重复发送 task-notification。限流是每个 monitor 的保护，不能保证整个会话的上下文或日志磁盘总量有严格上限。

实现链路：monitor 工具 → JobRegistry.spawn_monitor → stdout PIPE → _pump_monitor → pending_events → build_job_notifications → hook / watch_jobs → 主模型回答。

## 图片输入

按「接通图片输入」教程实现剪贴板、@ 图片、read_file 三条入口，支持 png/jpg/jpeg/gif/webp，沿用当前模型配置；服务端模型需要支持图像输入。

- 复制图片后，Windows 输入区按 **Alt+V**；macOS/Linux 按 **Ctrl+V**。出现 `[Image #1]` 后输入问题，回车提交。普通文本粘贴仍用终端原来的快捷键。
- 输入 `@assets/login.png 解释这个页面`，图片在引用位置直接作为附件；文本文件仍通过模拟 read_file 调用注入历史。路径与说明用空格分隔；@ 解析不支持含空格的路径。
- 告诉 Agent 图片路径，让它调用 read_file；图片按图像返回，offset/limit 只用于文本。文件工具可以读取含空格的路径。

`[Image #1] 这是什么？ [Image #2] 这个呢？` 转换为 `[图片1, 文本1, 图片2, 文本2]`，保留顺序。剪贴板和 @ 图片共用编号；删除占位符即不发送该粘贴图片，Esc 清空草稿及附件。提交后新草稿从 #1 编号；上下键只回填文本，重发旧截图需重新粘贴或 @ 引用。/rewind 会从所选轮次的多模态历史回填图片与文字。

剪贴板图片保存到 `~/.my-claude-code/clipboard/`，每次独立 PNG。Windows 使用 PowerShell（STA），macOS 用 osascript，Linux 当前使用 X11/xclip，需要预先安装。缺少工具或没有图片时明确提示，可改用 @ 引用。系统取图移到工作线程，避免阻塞输入界面。

三条入口最终都是 BinaryContent 原始字节与 MIME 类型，SDK 编码为 data URI，发送 Chat Completions 的 image_url 内容块。图片也随 JSONL 历史保存，/resume 无需原图片仍然存在。终端和 /api-detail 只显示类型/大小，会话列表提取用户文字。auto 分类器只取用户文字，不把图片内容当作授权。图片读取不登记到文本编辑状态，因此不会放开以文本覆盖二进制图片的检查。

详细解释见 [图片输入讲解](图片输入讲解.md)。test_images.py 验证图文顺序、混合引用、工具返回、真实 SDK 请求映射（模拟 HTTP）、历史恢复、回退、剪贴板分支与模拟终端按键，不调用真实模型或改变真实剪贴板。

## 数据

```text
~/.my-claude-code/
├── mcp.json
├── mcp-logs/
├── clipboard/                  # 每次粘贴的 PNG；图片数据也随会话保存
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

test_background_jobs.py 验证命令日志、退出码、超时/取消、通知去重、SDK schema 与权限、忙时提醒、空闲续跑、真实 Ctrl+B、会话切换与 Windows 进程树清理。test_subagents.py 增加隔离、报告回传、并行、失败、审批冒泡、拒绝、白名单、回退、自定义类型、请求上限和取消清理验证。

test_monitor.py 验证运行中事件、多批通知、日志、退出/超时、限流/队列/熔断、过长行、权限、事件转义、会话标题和忙时/空闲投递。模型均模拟，命令与 Windows 进程树清理使用真实临时进程。

改动前源码和旧接口测试在 `.reference-migration-backup/before-reference-alignment/`。目录被 Git 忽略；当前测试针对对齐后的接口。
