# GaoAgent

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![CLI](https://img.shields.io/badge/Interface-CLI-green)](#)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](./LICENSE)

一个面向终端的 Python 智能体工具，支持任务执行、MCP 服务接入、Skills 管理、RAG 知识库检索与 API 配置管理。

## 目录

- [功能亮点](#功能亮点)
- [1. 需要安装的依赖](#1-需要安装的依赖)
- [2. 如何安装](#2-如何安装)
- [3. 如何使用](#3-如何使用)
- [4. 详细操作说明（基于 `gaoagent/cli.py`）](#4-详细操作说明基于-gaoagentclipy)
- [FAQ](#faq)
- [开发与贡献](#开发与贡献)
- [许可证](#许可证)

## 功能亮点

- 统一 CLI 入口，支持子命令和快捷任务两种调用方式
- 支持 MCP 服务配置、启停、连通性测试与工具清单查看
- 支持 Skills 安装/卸载与作用域管理（项目/全局）
- 支持 RAG 知识库创建、更新、删除、语义检索
- 支持多 API 配置与默认模型切换

## 1. 需要安装的依赖

### 运行环境

- Python `>= 3.12`
- Windows / macOS / Linux

### Python 包依赖（项目已声明）

- `click>=8.1`
- `modelcontextprotocol`
- `chromadb==1.5.8`

说明：依赖声明位于 `pyproject.toml`，通过 `pip install` 安装项目时自动拉取。

## 2. 如何安装

```bash
git clone <你的仓库地址>
cd GaoAgent
pip install -e .
```

### 安装验证

```bash
gaoagent --version
gaoagent --help
```

## 3. 如何使用

### 命令入口模式

GaoAgent 提供两种入口：

1. 子命令模式：`gaoagent <command> [options]`
2. 快捷任务模式：`gaoagent --task "任务描述" --mode react|plan|retry`

### 快速开始

```bash
# 1) 查看帮助
gaoagent -h

# 2) 执行配置采集（交互式）
gaoagent config

# 3) 初始化当前项目
gaoagent init

# 4) 新增 API 配置
gaoagent api add

# 5) 执行任务
gaoagent task "帮我写一个发布脚本"

# 6) 快捷任务调用（无需显式 task 子命令）
gaoagent --task "分析这个仓库结构" --mode react
```

## 4. 详细操作说明（基于 `gaoagent/cli.py`）

以下命令说明按当前代码实现整理。

### 4.1 根命令与全局参数

- `gaoagent --version`：显示版本
- `gaoagent -h` / `gaoagent --help`：显示帮助
- `gaoagent --task <文本> --mode <plan|react|retry>`：快捷执行任务

示例：

```bash
gaoagent --task "实现文件上传接口" --mode react
```

注意：`task` 命令注释标明当前主要实现为 `react` 模式，`plan/retry` 为预留模式。

### 4.2 核心命令

#### `gaoagent init`

在当前目录初始化项目配置。

```bash
gaoagent init
```

#### `gaoagent config`

执行交互式配置采集/修复流程，覆盖 API、MCP、Skills、RAG 初始化流程。

```bash
gaoagent config
```

#### `gaoagent chat`

聊天入口（当前以参数透传为主）。

参数：

- `--new`：开启新会话
- `--prompt <文本>`：直接发送输入
- `--api <api_name>`：指定 API
- `--model <model_name>`：指定模型
- `--context-size <int>` / `--contextSize <int>`：指定上下文长度

示例：

```bash
gaoagent chat
gaoagent chat --new
gaoagent chat --prompt "你好"
gaoagent chat --api openai --model gpt-4.1 --context-size 20
```

#### `gaoagent task [question] --mode <plan|react|retry> [--id <session_id>]`

创建并运行任务。

- `question` 可省略，省略时进入交互输入
- `--mode` 默认 `react`
- `--id` 传入时，可以导入或保存以该 id 命名的历史对话记录（`.gaoagent/history/<session_id>.json`）

示例：

```bash
gaoagent task "生成接口文档"
gaoagent task --mode react "帮我重构这个模块"
gaoagent task --id session_1 "继续刚才的工作"
gaoagent task
```

### 4.3 MCP 命令组：`gaoagent mcp`

用于 MCP 服务配置管理。

#### `gaoagent mcp list`

列出 MCP 服务。

```bash
gaoagent mcp list
```

#### `gaoagent mcp add`

交互式新增 MCP 配置（输入 JSON 对象）。

```bash
gaoagent mcp add
```

#### `gaoagent mcp remove [name]`

删除 MCP 服务，`name` 可省略（交互选择）。

```bash
gaoagent mcp remove
gaoagent mcp remove my-mcp
```

#### `gaoagent mcp enable [name]` / `gaoagent mcp disable [name]`

启用或禁用 MCP 服务。

```bash
gaoagent mcp enable my-mcp
gaoagent mcp disable my-mcp
```

#### `gaoagent mcp test [name]`

测试 MCP 连通性并输出工具摘要。

```bash
gaoagent mcp test
gaoagent mcp test my-mcp
```

### 4.4 Skills 命令组：`gaoagent skills`

用于 Skills 安装和卸载。

#### `gaoagent skills list`

列出当前作用域可见 Skills。

```bash
gaoagent skills list
```

#### `gaoagent skills add`

交互式安装 Skill。

```bash
gaoagent skills add
```

#### `gaoagent skills remove`

交互式卸载 Skill。

```bash
gaoagent skills remove
```

### 4.5 RAG 命令组：`gaoagent rag`

用于知识库生命周期管理和检索。

#### `gaoagent rag list`

列出知识库。

```bash
gaoagent rag list
```

#### `gaoagent rag add [name] [chunker_py_file]`

新增知识库，可选自定义切片器。

```bash
gaoagent rag add
gaoagent rag add mykb
gaoagent rag add mykb ./chunker.py
```

#### `gaoagent rag update [name] [chunker_py_file]`

增量更新知识库。

```bash
gaoagent rag update
gaoagent rag update mykb
gaoagent rag update mykb ./chunker.py
```

#### `gaoagent rag remove [name]`

移除知识库。

```bash
gaoagent rag remove
gaoagent rag remove mykb
```

#### `gaoagent rag search <kb_name> <query> [--top-k N]`

执行知识库语义检索。

```bash
gaoagent rag search mykb "如何部署"
gaoagent rag search mykb "向量检索" --top-k 8
```

### 4.6 API 命令组：`gaoagent api`

用于模型 API 配置管理。

#### `gaoagent api list`

列出 API 配置。

```bash
gaoagent api list
```

#### `gaoagent api add`

交互式新增 API 配置。

```bash
gaoagent api add
```

#### `gaoagent api remove <name>`

删除指定 API 配置。

```bash
gaoagent api remove openai
```

#### `gaoagent api edit <name>`

编辑指定 API 配置。

```bash
gaoagent api edit openai
```

#### `gaoagent api default <name>`

设置默认 API。

```bash
gaoagent api default openai
```

### 4.7 配置文件位置

- 全局配置目录：`~/.gaoagent/`
- 全局 API 配置：`~/.gaoagent/gao_client_api_config.json`
- 全局 MCP 配置：`~/.gaoagent/gao_client_mcp_setting.json`
- 项目配置目录：`<project_root>/.gaoagent/`

## FAQ

### Q1：第一次使用建议怎么走？

推荐顺序：

1. `gaoagent init`
2. `gaoagent config`
3. `gaoagent api add`
4. `gaoagent task "你的任务描述"`

### Q2：为什么有时会操作“项目配置”，有时是“全局配置”？

当你在已初始化项目目录内运行命令时，部分命令会优先使用项目作用域配置；不在项目目录时则使用全局配置。

### Q3：RAG 入库前要准备什么？

先创建知识库，再把待入库文件放入知识库目录，最后执行 `rag add` 或 `rag update`。

## 开发与贡献

欢迎提交 Issue / PR。

## 许可证

本项目采用 `MIT License`，详见 [LICENSE](./LICENSE)。
