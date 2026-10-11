# 安装与接入

公开命令使用 `repoengram`。本版保留内部 `skillforge` 模块、MCP 名称和 `.skillforge` 数据目录，兼容已有产物；旧 `skillforge` CLI 别名仍可使用。

## 安装 RepoEngram

本版通过 GitHub 安装，尚未发布到 PyPI。运行环境要求 Python 3.11+ 和 Git；知识提炼当前使用已安装、已登录的 Codex CLI。使用 Claude Code 执行任务也仍需这个提炼后端。

建议放在独立虚拟环境中：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install "git+https://github.com/Winterxin/RepoEngram.git@main"
repoengram --help
```

激活环境后切换到需要增强的项目目录。该项目应为具有已提交 HEAD 的 Python Git 仓库；项目本身的测试依赖需另行安装。首次准备会调用模型，已验证的相同快照可以复用知识。

## 客户端接入与运行后端

RepoEngram 通过项目 Skills 与 MCP 服务提供工程知识。在 Codex 本地项目会话中，可委托 Agent 完成部署与初始化；客户端需要具备本地命令执行权限，并能访问目标仓库及安装环境。

知识准备与任务执行采用独立的接入路径：

- **语义提炼后端**：知识初始化与刷新通过已认证的 Codex CLI 执行。运行环境需提供可调用的 `codex` 命令；客户端安装本身不保证该命令可用。
- **知识消费与任务执行**：客户端通过项目 Skill 和 MCP 检索已发布知识、执行修改并提交验证。用户可在现有项目会话中完成交互。

Codex 桌面客户端、CLI 与 IDE 扩展共用 MCP 配置。仓库级 `.codex/config.toml` 仅在受信任项目中加载。完成配置后，在目标项目新建会话并确认 `skillforge` MCP 服务可用；若未加载，检查项目信任状态、服务配置与 Python 路径。

配置机制参见 [OpenAI 官方 MCP 文档](https://developers.openai.com/codex/mcp/)与 [Skills 文档](https://developers.openai.com/codex/skills/)。当前端到端验收基于 Codex CLI；桌面客户端依照共用配置与标准接口接入，尚未单独完成同等端到端验收。

## Codex：准备并连接

在目标项目目录运行：

```powershell
$prepared = repoengram prepare . --task "为当前模块添加新能力，同时维护相关测试和调用方" --target codex --json | ConvertFrom-Json
repoengram connect . --baseline $prepared.evolution_baseline_path
```

可增加 `--current-module your_package.your_module` 指明实际模块。使用准确模块名，不要把示例名当作项目配置。

`connect` 将项目 MCP 配置写入 `.codex/config.toml`，将发现入口写入 `.agents/skills/skillforge-discovery/SKILL.md`，并发布已审核的知识快照。连接本身不调用模型。保留安装环境；配置引用这个环境中的 Python。

然后在该项目新开 Codex 会话，确认 `skillforge` MCP 已加载。Agent 的正常顺序是：查询工作流 → 完整读取适用 Skill → 修改和测试 → 为有实际测试依据的行为声明检查覆盖 → 正式验证。

需要正式验证时，先配置 `.skillforge/autonomous/checks.json`。例如，先创建 `.runtime` 目录，并填写目标项目实际可运行的测试命令：

```json
{
  "checks": [
    {
      "check_id": "project-tests",
      "kind": "TEST",
      "covers": [],
      "argv": ["python", "-m", "pytest", "-q", "--basetemp", ".runtime/pytest-verify", "tests"],
      "timeout_seconds": 90
    }
  ]
}
```

这是一份配置示例。命令的 Python 应指向已安装目标项目依赖的解释器；时间和测试目录按项目实际情况设置。先离线运行同一命令，确认它能执行。空 `covers` 不代表覆盖任何 Skill 行为；Agent 必须通过 `bind_verify_check` 声明真实有据的当前义务覆盖，再调用 `verify_patch`。不能用零义务或 Agent 自述代替正式验证。

## Claude Code：导出项目 Skills

同样安装 RepoEngram，在目标项目运行：

```bash
repoengram prepare . --task "为当前模块添加新能力，同时维护相关测试和调用方" --target claude_code --json
```

选中的 Skills 写入 `.claude/skills/<generated-name>/`，可供 Claude Code 按标准 Skills 机制读取。Claude 导出与目录同步已由本地回归覆盖；目前完整真实任务闭环的实跑证据来自 Codex，不能把二者的验收程度视为相同。

`repoengram connect` 当前自动配置 Codex，不会自动注册 Claude MCP。其他支持 stdio MCP 的客户端可自行注册已发布的知识服务，入口为：

```bash
python -m skillforge.integration.mcp_server --repo <项目绝对路径>
```

服务要求目标项目已有经过 `connect --baseline` 发布的有效快照，客户端要使用安装了 RepoEngram 的 Python。Skills 标准和 MCP 提供可迁移的接口；不同客户端的注册、权限和 Agent 是否读取知识仍由各自适配决定。当前不声称所有 harness 开箱即用。

## 验证与更新

准备结果还返回 Verify receipt 和演进 baseline。CLI 可使用实际 receipt 和检查文件验证当前修改：

```bash
repoengram verify . --receipt <verify_receipt_path> --checks <checks_json_path> --json
```

仓库身份、HEAD、源码或配置变化后，应使用正常 prepare/演进流程刷新知识。保持当前项目的基线与知识一致；不要把别的项目产物复制过来当作缓存。

## 接入格式参考

- [Codex Skills](https://developers.openai.com/codex/skills/)
- [Codex MCP](https://developers.openai.com/codex/mcp/)
- [Claude Code Skills](https://code.claude.com/docs/en/skills)
- [Claude Code MCP](https://code.claude.com/docs/en/mcp)
