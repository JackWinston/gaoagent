from pathlib import Path
from typing import Any

import datetime
import json
import shutil
import time
import click

from gaoagent.mcp.MCPClientCompat import (
    MCPStdioClientSync,
    build_mcp_tools_cache_payload,
    write_mcp_tools_cache,
)
from gaoagent.core.runner.Utils import scan_skills_metadata
from gaoagent.rag.RagApiConfig import RagApiConfigStore
from gaoagent.rag.RagChromaIndexer import RagChromaIndexer, RagChromaIndexerConfig
from gaoagent.rag.RagStorePath import resolve_chroma_store_dir, resolve_index_meta_file


class CoreConfigDefault:
    """
    核心配置入口。

    在程序初始化时调用，用于引导用户完成默认配置的创建与管理。
    """

    def config(self) -> None:
        """
        配置流程（交互式）：

        1. 检查是否存在 ~/.gaoagent；不存在则创建。
        2. 检查是否已存在默认配置文件；存在则展示当前配置。
        3. 若不存在默认配置文件，引导用户完成配置创建：
           a. 添加 API 配置并命名（url、key、model、每个model的context_windows）。
           b. 添加 MCP 配置并命名（写入 gao_client_mcp_setting.json）。
           c. 添加 Skills 配置并命名（创建 skills/，将用户添加的 skill.md 放入其中）。
           d. 添加 RAG 配置并命名（创建 rag/；每个知识库独立子目录；调用三方库切片并写入向量库）。
           e. 每完成一步,都写入最终 config.json。
        """
        config_dir = self._ensure_config_dir()
        api_config_file = config_dir / "gao_client_api_config.json"
        mcp_config_file = config_dir / "gao_client_mcp_setting.json"

        apis: dict[str, Any] = {}
        existing_api_payload = self._read_json(api_config_file)
        if isinstance(existing_api_payload, dict) and isinstance(existing_api_payload.get("apis"), dict):
            apis = existing_api_payload["apis"]

        api_names: set[str] = set(apis.keys())
        new_api_count = 0
        while True:
            api_config = self._import_api_config()
            if api_config is None:
                break

            if api_config["name"] in api_names:
                click.echo("API 配置名称重复，请重新输入")
                continue

            api_names.add(api_config["name"])
            apis[api_config["name"]] = {
                "base_url": api_config["base_url"],
                "api_key": api_config["api_key"],
                "models": api_config["models"],
            }
            self._write_api_config(apis)
            new_api_count += 1
            click.echo(
                f"API 配置已采集：name={api_config['name']}, base_url={api_config['base_url']}, models={list(api_config['models'].keys())}"
            )

            if not click.confirm("继续添加一组 API 配置？", default=False):
                break

        click.echo(f"API 配置采集完成，本次新增 {new_api_count} 组")

        mcp_configs: dict[str, Any] = {}
        existing_mcp_payload = self._read_json(mcp_config_file)
        if isinstance(existing_mcp_payload, dict):
            if isinstance(existing_mcp_payload.get("mcpServers"), dict):
                mcp_configs = existing_mcp_payload["mcpServers"]

        new_mcp_count = 0
        while True:
            mcp_config = self._import_mcp_config()
            if mcp_config is None:
                break

            (mcp_name, mcp_body) = next(iter(mcp_config.items()))
            if mcp_name in mcp_configs:
                click.echo(f"MCP 配置已存在，将覆盖：{mcp_name}")
            mcp_configs[mcp_name] = mcp_body
            self._write_mcp_config(mcp_configs)
            new_mcp_count += 1
            click.echo(f"MCP 配置已采集：{mcp_name}")

            if not click.confirm("继续添加一组 MCP 配置？", default=False):
                break

        click.echo(f"MCP 配置采集完成，本次新增 {new_mcp_count} 组")

        if mcp_configs:
            try:
                # 在配置阶段预拉取 MCP 工具并落缓存：
                # - 让 ReActRunner 启动时无需每次都连 MCP server 做 tools/list；
                # - 工具 schema 可直接注入到 LLM 的 tools 定义中，提高参数生成质量。
                cache_payload = build_mcp_tools_cache_payload(
                    mcp_configs,
                    connect_and_list_tools=lambda name, body: MCPStdioClientSync.from_config(
                        server_name=name,
                        config=body,
                    ).list_tools(),
                    generated_at=datetime.datetime.now(datetime.UTC).isoformat(),
                )
                write_mcp_tools_cache(cache_payload)
                tool_count = len((cache_payload.get("exported_map") or {}).keys())
                click.echo(f"MCP 工具缓存已更新，共 {tool_count} 个工具")
            except Exception as e:
                click.echo(f"MCP 工具缓存更新失败：{e}")

        isInitSkills = self._import_skills_config()

        if isInitSkills:
            skills_dir = Path.home() / ".gaoagent" / "skills"
            (skills, invalid_skills) = self._load_skills_metadata(skills_dir)
            click.echo(f"Skills 配置采集完成，共 {len(skills)} 个")
            for skill in skills:
                click.echo(f"- {skill['name']}: {skill['description']}")
            if invalid_skills:
                click.echo(f"以下 SKILL.md 格式不正确，共 {len(invalid_skills)} 个")
                for item in invalid_skills:
                    click.echo(f"- {item['path']}: {item['reason']}")

        self._import_rag_config()

    def _ensure_config_dir(self) -> Path:
        """
        确保用户级配置目录存在。

        配置目录固定为 `~/.gaoagent`（即 `Path.home() / ".gaoagent"`）。
        """
        config_dir = Path.home() / ".gaoagent"
        if config_dir.exists() and not config_dir.is_dir():
            raise RuntimeError(f"配置路径已存在但不是目录：{config_dir}")

        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir

    def _read_json(self, file_path: Path) -> Any | None:
        """_read_json 方法。
        
        用途:
        - 读取指定 JSON 文件的内容。
        
        参数:
        - file_path: 输入参数，指定要读取的 JSON 文件路径。
        
        返回:
        - Any: 返回 JSON 文件的内容。
        """
        while True:
            if not file_path.exists():
                return None
            try:
                return json.loads(file_path.read_text(encoding="utf-8"))
            except Exception as e:
                click.echo(f"读取失败：{file_path}，{e}")
                if click.confirm("忽略该文件并继续？", default=True):
                    return None

    def _write_json(self, file_path: Path, payload: Any) -> None:
        """_write_json 方法。
        
        用途:
        - 写入指定 JSON 文件的内容。
        
        参数:
        - file_path: 输入参数，指定要写入的 JSON 文件路径。
        - payload: 输入参数，指定要写入的 JSON 内容。
        
        """
        tmp_file = file_path.with_name(f"{file_path.name}.tmp")
        tmp_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        tmp_file.replace(file_path)

    def _write_api_config(self, apis: dict[str, Any]) -> None:
        """
        覆盖写入 `~/.gaoagent/gao_client_api_config.json`。
        """
        config_dir = self._ensure_config_dir()
        config_file = config_dir / "gao_client_api_config.json"
        self._write_json(config_file, {"apis": apis})

    def _write_mcp_config(self, mcp_configs: dict[str, Any]) -> None:
        """
        覆盖写入 `~/.gaoagent/gao_client_mcp_setting.json`。
        """
        config_dir = self._ensure_config_dir()
        config_file = config_dir / "gao_client_mcp_setting.json"
        self._write_json(config_file, {"mcpServers": mcp_configs})

    def _import_api_config(self) -> dict[str, Any] | None:
        """
        引导用户输入 API 相关配置，并返回可序列化的 dict；若用户选择跳过则返回 None。

        字段：
        1. name: 该组配置的名字
        2. base_url: API 的 base url
        3. api_key: API key（仅用于认证；不会在终端回显）
        4. models: 模型集合（key 为模型名）
           - context_window: 上下文窗口大小
           - capabilities: 默认能力（vision/tools/reasoning）
           - aliases: 别名列表（用于命令行快捷选择）
        """
        if click.confirm("是否跳过 API 配置？", default=False):
            return None

        name = self._prompt_non_empty_str("请输入 API 配置名称")

        base_url = self._prompt_non_empty_str("请输入 API Base URL")

        api_key = self._prompt_non_empty_str("请输入 API Key", hide_input=True)

        models: dict[str, Any] = {}
        while True:
            model_id = self._prompt_non_empty_str("请输入模型名")
            if model_id in models:
                click.echo("模型名重复，请重新输入")
                continue

            context_window = self._prompt_positive_int(
                "请输入该模型 context window", default=8192, show_default=True
            )

            vision = click.confirm("该模型是否支持图片（vision）？", default=False)
            tools = click.confirm("该模型是否支持工具调用（tools）？", default=True)
            reasoning = click.confirm("该模型是否支持推理（reasoning）？", default=False)

            aliases_raw = click.prompt("请输入别名（逗号分隔，可留空）", type=str, default="", show_default=False).strip()
            aliases = [a.strip() for a in aliases_raw.split(",") if a.strip()] if aliases_raw else []

            models[model_id] = {
                "id": model_id,
                "context_window": context_window,
                "capabilities": {"vision": vision, "tools": tools, "reasoning": reasoning},
                "aliases": aliases,
            }

            if not click.confirm("继续添加模型？", default=False):
                break

        return {"name": name, "base_url": base_url, "api_key": api_key, "models": models}

    def _import_mcp_config(self) -> dict[str, Any] | None:
        """
        引导用户输入 MCP 相关配置，并返回可序列化的 dict；若用户选择跳过则返回 None。

        格式示例：
        {
          "bing-search": {
            "disabled": true,
            "timeout": 60,
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "bing-cn-mcp"],
            "env": {"HTTP_PROXY": "http://127.0.0.1:7890"}
          }
        }

        也支持：
        - SSE: {"type":"sse","url":"https://...","headers":{...}}
        - Streamable HTTP: {"type":"streamable_http","url":"https://...","headers":{...}}
        """
        if click.confirm("是否跳过 MCP 配置？", default=False):
            return None

        while True:
            raw = click.prompt("请输入 MCP JSON对象", type=str).strip()
            try:
                value = json.loads(raw)
            except Exception:
                click.echo("格式错误：请输入合法的 JSON 对象")
                continue

            if not isinstance(value, dict):
                click.echo("格式错误：MCP 配置必须是 JSON 对象")
                continue

            try:
                self._validate_mcp_config(value)
            except Exception as e:
                click.echo(f"格式错误：{e}")
                continue

            return value

    def _import_skills_config(self) -> bool :
        """
        引导用户输入技能相关配置。
        """

        if click.confirm("是否跳过 Skills 配置？", default=False):
            return False        

        skills_dir = Path.home() / ".gaoagent" / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)

        click.echo(f"请将Skills对应的md文件复制到 {skills_dir}")
        return click.confirm("是否已经完成?", default=False)

    def _import_rag_config(self) -> bool :
        """
        引导用户输入 RAG 相关配置。
        """
        if click.confirm("是否跳过 RAG 配置？", default=False):
            return False
        return self.create_rag_knowledge_base()

    def create_rag_knowledge_base(
        self,
        *,
        rag_root_dir: Path | None = None,
        kb_name: str | None = None,
        chunker_py_file: str | None = None,
    ) -> bool:
        """
        交互式创建一个知识库目录（可复用）。

        流程：
        1. 询问并获取知识库名称；
        2. 在 `~/.gaoagent/rag/<知识库名>` 创建目录；
        3. 提示用户复制文件到该目录；
        4. Embedding 与向量库写入暂不实现；
        5. 输出创建成功信息。
        """
        rag_dir = rag_root_dir if rag_root_dir is not None else (self._ensure_config_dir() / "rag")
        rag_dir.mkdir(parents=True, exist_ok=True)

        if kb_name is None and (not click.confirm("是否创建知识库？", default=True)):
            return False

        selected_name = (kb_name or "").strip()
        while True:
            if not selected_name:
                selected_name = self._prompt_non_empty_str("请输入知识库名称")
            kb_dir = rag_dir / selected_name
            if kb_dir.exists():
                if not kb_dir.is_dir():
                    click.echo(f"同名路径已存在且不是目录，请更换名称：{kb_dir}")
                    if kb_name is not None:
                        return False
                    selected_name = ""
                    continue
                backup_count = self._backup_existing_rag_artifacts(kb_dir)
                if backup_count > 0:
                    click.echo(f"检测到已有数据库/索引，已完成备份（{backup_count} 项）：{kb_dir}")
                else:
                    click.echo(f"知识库目录已存在，将直接复用：{kb_dir}")
                break
            kb_dir.mkdir(parents=True, exist_ok=False)
            break

        click.echo(f"请将需要入库的文件复制到目录：{kb_dir}")
        copied = click.confirm("是否已经完成文件复制？", default=False)
        if not copied:
            click.echo("已取消创建知识库（未执行入库）")
            return False
        self._prompt_import_rag_api_after_copy(kb_name=selected_name, rag_dir=rag_dir)

        (ok, reason) = self._build_rag_vector_store(
            kb_name=selected_name,
            kb_dir=kb_dir,
            chunker_py_file=chunker_py_file,
        )
        if not ok:
            # 入库失败时仅清理本次生成的索引产物，保留用户源文件。
            self._remove_rag_artifacts(kb_dir)
            if reason:
                click.echo(f"知识库创建失败：{reason}")
            else:
                click.echo("知识库创建失败")
            return False

        click.echo(f"知识库创建成功：{selected_name}")
        return True

    def _backup_existing_rag_artifacts(self, kb_dir: Path) -> int:
        """
        备份已有知识库目录中的向量库与索引文件。

        仅备份：
        - `.gaoagent/rag/.chrome_store/` 下的 Chroma 存储目录
        - 同目录下的 index_meta.json
        """
        store_dir = resolve_chroma_store_dir(kb_dir=kb_dir, kb_name=kb_dir.name)
        meta_file = resolve_index_meta_file(kb_dir=kb_dir, kb_name=kb_dir.name)
        targets = [store_dir, meta_file]
        existing = [p for p in targets if p.exists()]
        if not existing:
            return 0

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_root = kb_dir / "_backup" / ts
        backup_root.mkdir(parents=True, exist_ok=False)
        count = 0
        for src in existing:
            dst = backup_root / src.name
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            count += 1
        return count

    def _remove_rag_artifacts(self, kb_dir: Path) -> None:
        """_remove_rag_artifacts 方法。
        
        用途:
        - 删除知识库目录中的向量库与索引文件。
        
        参数:
        - kb_dir: 输入参数，指定知识库目录路径。
        """
        store_dir = resolve_chroma_store_dir(kb_dir=kb_dir, kb_name=kb_dir.name)
        meta_file = resolve_index_meta_file(kb_dir=kb_dir, kb_name=kb_dir.name)
        targets = [store_dir, meta_file]
        for p in targets:
            if not p.exists():
                continue
            last_err: Exception | None = None
            for _ in range(6):
                try:
                    if p.is_dir():
                        shutil.rmtree(p)
                    else:
                        p.unlink(missing_ok=True)
                    last_err = None
                    break
                except PermissionError as e:
                    # Windows 上 sqlite 文件可能被短暂占用，稍后重试。
                    last_err = e
                    time.sleep(0.3)
                except Exception as e:
                    last_err = e
                    break
            if last_err is not None:
                click.echo(f"清理索引产物失败（将跳过该项）：{p}，{last_err}")

    def _build_rag_vector_store(
        self,
        *,
        kb_name: str,
        kb_dir: Path,
        chunker_py_file: str | None = None,
    ) -> tuple[bool, str]:
        """
        构建知识库 Embedding 并写入向量库。

        返回：
        - bool: 是否成功；
        - str: 失败原因（成功时可为空字符串）。
        """
        config_store = RagApiConfigStore(kb_name=kb_name, kb_dir=kb_dir)
        indexer_config: RagChromaIndexerConfig = config_store.resolve_indexer_config(
            local_embedding_model="all-MiniLM-L6-v2",
            chunk_size=1200,
            chunk_overlap=200,
            batch_size=64,
        )
        custom_chunker = (chunker_py_file or "").strip()
        if custom_chunker:
            indexer_config.chunker_py_file = custom_chunker
        indexer = RagChromaIndexer(indexer_config)
        return indexer.ingest_knowledge_base(kb_name=kb_name, kb_dir=kb_dir)

    def _prompt_import_rag_api_after_copy(self, *, kb_name: str, rag_dir: Path) -> None:
        """_prompt_import_rag_api_after_copy 方法。
        
        用途:
        - 提示用户导入 RAG 远程 Embedding API 配置。
        
        参数:
        - kb_name: 输入参数，指定知识库名称。
        - rag_dir: 输入参数，指定 RAG 目录路径。

        """
        if not click.confirm("是否现在导入 RAG 远程 Embedding API 配置？", default=False):
            return
        kb_dir = rag_dir / kb_name
        store = RagApiConfigStore(kb_name=kb_name, kb_dir=kb_dir)
        try:
            config_file = store.config_file()
        except Exception as e:
            click.echo(f"导入 RAG API 配置失败：{e}")
            return

        payload = store.load()
        remote_api = payload.get("remote_api")
        if isinstance(remote_api, dict) and remote_api:
            should_overwrite = click.confirm(
                f"知识库 '{kb_name}' 已存在远程配置，是否覆盖？",
                default=False,
            )
            if not should_overwrite:
                click.echo(f"已跳过导入：{kb_name}")
                return

        base_url = self._prompt_non_empty_str("请输入远程 Base URL（OpenAI 兼容）").rstrip("/")
        api_key = self._prompt_non_empty_str("请输入远程 API Key", hide_input=True)
        embedding_model = self._prompt_non_empty_str("请输入远程 Embedding Model")
        timeout_sec = self._prompt_positive_int("请输入请求超时秒数", default=120, show_default=True)
        payload["remote_api"] = {
            "base_url": base_url,
            "api_key": api_key,
            "embedding_model": embedding_model,
            "timeout_sec": timeout_sec,
        }

        try:
            store.save(payload)
        except Exception as e:
            click.echo(f"导入 RAG API 配置失败：{e}")
            return
        click.echo(f"已导入知识库远程配置：{kb_name}")
        click.echo(f"配置文件：{config_file}")
        
    def _load_skills_metadata(self, skills_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        """_load_skills_metadata 方法。
        
        用途:
        - 加载技能目录下的技能元数据文件。
        
        参数:
        - skills_dir: 输入参数，指定技能目录路径。
        
        返回:
        - tuple[list[dict[str, str]], list[dict[str, str]]]: 返回技能元数据列表（已禁用 与未禁用）。
        """
        return scan_skills_metadata(skills_dir)

    
    def _prompt_non_empty_str(self, text: str, *, hide_input: bool = False) -> str:
        """
        获取非空字符串输入；为空则提示并重新输入。
        """
        while True:
            value = click.prompt(text, type=str, hide_input=hide_input).strip()
            if value:
                return value
            click.echo("输入不能为空，请重新输入")

    def _prompt_positive_int(self, text: str, *, default: int, show_default: bool = True) -> int:
        """
        获取正整数输入；非正整数则提示并重新输入。
        """
        while True:
            value = click.prompt(text, type=int, default=default, show_default=show_default)
            if value > 0:
                return value
            click.echo("请输入正整数")


    def _validate_mcp_config(self, config: dict[str, Any]) -> None:
        """
        校验 MCP 配置格式；不符合时抛出 ValueError。
        """
        if not isinstance(config, dict) or len(config) != 1:
            raise ValueError("MCP 配置必须是仅包含 1 个键的对象（键为 MCP 名称）")

        (name, body) = next(iter(config.items()))
        if not isinstance(name, str) or not name.strip():
            raise ValueError("MCP 名称必须是非空字符串")
        if not isinstance(body, dict):
            raise ValueError("MCP 配置内容必须是对象")

        required_keys = {"disabled", "timeout", "type"}
        missing = required_keys - set(body.keys())
        if missing:
            raise ValueError(f"MCP 配置缺少字段：{sorted(missing)}")

        if not isinstance(body["disabled"], bool):
            raise ValueError("MCP.disabled 必须为 bool")
        if not isinstance(body["timeout"], int) or body["timeout"] <= 0:
            raise ValueError("MCP.timeout 必须为正整数")
        mcp_type = body["type"]
        if mcp_type not in ("stdio", "sse", "streamable_http"):
            raise ValueError("MCP.type 仅支持 stdio/sse/streamable_http")
        if mcp_type == "stdio":
            if not isinstance(body.get("command"), str) or not str(body.get("command")).strip():
                raise ValueError("stdio MCP.command 必须为非空字符串")
            args = body.get("args")
            if not isinstance(args, list) or any((not isinstance(x, str)) for x in args):
                raise ValueError("stdio MCP.args 必须为字符串数组")
            env = body.get("env")
            if env is not None:
                if not isinstance(env, dict) or any((not isinstance(k, str) or not isinstance(v, str)) for k, v in env.items()):
                    raise ValueError("stdio MCP.env 必须是 string->string 对象")
            return
        if not isinstance(body.get("url"), str) or not str(body.get("url")).strip():
            raise ValueError(f"{mcp_type} MCP.url 必须为非空字符串")
        headers = body.get("headers")
        if headers is not None:
            if not isinstance(headers, dict) or any((not isinstance(k, str) or not isinstance(v, str)) for k, v in headers.items()):
                raise ValueError(f"{mcp_type} MCP.headers 必须是 string->string 对象")
