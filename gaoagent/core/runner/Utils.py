from __future__ import annotations

"""
Runner 层通用工具模块。

架构定位:
- 为 `gaoagent.core.runner` 提供稳定、无副作用（或弱副作用）的基础能力，
  包括时间与文本处理、异常规范化、项目配置定位、MCP/Skill/RAG 配置读取、
  以及 LLM 响应协议归一化。

设计意图:
- 将跨 Handler/Runner 复用的“环境探测 + 配置访问 + 协议适配”逻辑集中维护，
  避免在调用方散落重复实现，确保行为一致、便于演进。
"""

import json
import time
import traceback
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from gaoagent.core.runner.HttpClient import HttpResponse
    from gaoagent.core.runner.BaseRunner import RequestBaseInfo, StepResult


_PROJECTS_REGISTRY_FILENAME = "inited_projects.txt"


def now_ms() -> int:
    """now_ms 函数。

    用途:
    - 获取当前时间戳（毫秒级）。

    返回:
    - int: 返回当前时间戳（毫秒级）。
    """
    return int(time.time() * 1000)


def truncate_text(s: str, limit: int) -> str:
    """truncate_text 函数。

    用途:
    - 截断文本，确保不超过指定长度。

    参数:
    - s: 输入参数，用于指定要截断的文本。
    - limit: 输入参数，用于指定截断后的文本长度上限。

    返回:
    - str: 返回截断后的文本。
    """
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 1)] + "…"


def safe_json_dumps(value: Any) -> str:
    """safe_json_dumps 函数。

    用途:
    - 安全地将任意Python对象转换为JSON字符串。

    参数:
    - value: 输入参数，用于指定要转换的Python对象。

    返回:
    - str: 返回转换后的JSON字符串。
    """
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return repr(value)


def summarize(value: Any, limit: int = 400) -> str:
    """summarize 函数。

    用途:
    - 对任意值进行总结，确保不超过指定长度。

    参数:
    - value: 输入参数，用于指定要总结的值。
    - limit: 输入参数，用于指定总结后的文本长度上限。

    返回:
    - str: 返回总结后的文本。
    """
    if value is None:
        return "null"
    if isinstance(value, (str, int, float, bool)):
        return truncate_text(str(value), limit)
    return truncate_text(safe_json_dumps(value), limit)


def redact(value: Any) -> Any:
    """
    递归脱敏敏感字段，适用于日志与调试输出。

    脱敏策略:
    - 当输入为 `dict` 时，若 key 命中敏感词（如 `api_key/token/password`），
      对应 value 统一替换为 `"***"`。
    - `list` 会递归处理其元素。
    - 其他类型原样返回。

    参数:
    - value: 任意待处理对象。

    返回:
    - Any: 结构保持不变的脱敏结果。
    """
    sensitive_keys = {
        "api_key",
        "apikey",
        "key",
        "token",
        "secret",
        "password",
        "authorization",
    }
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if isinstance(k, str) and k.lower() in sensitive_keys:
                out[k] = "***"
            else:
                out[k] = redact(v)
        return out
    if isinstance(value, list):
        return [redact(x) for x in value]
    return value


def normalize_exception(e: BaseException) -> dict[str, Any]:
    """normalize_exception 函数。

    用途:
    - 对异常进行标准化处理，提取异常类型、消息和栈轨迹。

    参数:
    - e: 输入参数，用于控制该函数的处理行为。

    返回:
    - dict[str, Any]: 返回标准化后的异常信息，包含异常类型、异常消息和异常栈轨迹。
    """
    return {
        "type": type(e).__name__,
        "message": str(e),
        "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
    }


def global_config_dir() -> Path:
    """global_config_dir 函数。

    用途:
    - 获取全局配置目录的路径。

    返回:
    - Path: 返回全局配置目录的路径。
    """
    return Path.home() / ".gaoagent"


def load_project_registry_paths() -> list[Path]:
    """load_project_registry_paths 函数。

    用途:
    - 加载项目注册文件中的项目路径。

    返回:
    - list[Path]: 返回项目注册文件中记录的项目路径列表。
    """
    registry_file = global_config_dir() / _PROJECTS_REGISTRY_FILENAME
    if not registry_file.exists() or not registry_file.is_file():
        return []
    try:
        lines = registry_file.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []

    roots: list[Path] = []
    seen: set[str] = set()
    for line in lines:
        raw = line.strip()
        if not raw:
            continue
        try:
            root = Path(raw).expanduser().resolve()
        except Exception:
            continue
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        roots.append(root)
    return roots


def try_project_root_dir() -> Path | None:
    """
     返回当前执行上下文对应的项目根目录（目录下需存在 `.gaoagent`）。

    解析顺序:
    - 优先使用当前工作目录；
    - 若当前目录不是项目根，则在注册表中查找能覆盖当前路径的已初始化项目，
      并选择“最深匹配”目录（最长路径前缀）。

    失败行为:
    - 向终端输出初始化提示并抛出 `RuntimeError`。
    """
    cwd = Path.cwd().resolve()
    config_dir = cwd / ".gaoagent"
    if config_dir.exists() and config_dir.is_dir():
        return cwd

    candidates: list[Path] = []
    for root in load_project_registry_paths():
        config = root / ".gaoagent"
        if not (
            root.exists() and root.is_dir() and config.exists() and config.is_dir()
        ):
            continue
        if root == cwd or root in cwd.parents:
            candidates.append(root)
    if candidates:
        candidates.sort(key=lambda p: len(p.parts), reverse=True)
        return candidates[0]
    return None


def project_config_dir() -> Path | None:
    """
    返回当前项目的配置目录路径（`<project_root>/.gaoagent`）。

    该函数是 Runner/Handler 访问项目级配置文件的统一入口。
    """
    root = try_project_root_dir()
    if root is None:
        return None
    return root / ".gaoagent"


def _find_config_file(name: str) -> Path | None:
    """
    生成项目配置目录下的目标文件路径。

    参数:
    - name: 配置文件名或子路径（相对 `.gaoagent`）。

    返回:
    - Path: 目标配置文件的绝对路径对象（不保证文件存在）。
    """
    config_dir = project_config_dir()
    if config_dir is None:
        return None
    return config_dir / name



def load_request_base_info() -> RequestBaseInfo | None:
    """
    加载默认 LLM 请求基础信息。

    配置来源:
    - `.gaoagent/gao_client_api_config.json`

    容错策略:
    - `default_api` 缺失或非法时，回退到第一个可用 API；
    - `default_model` 缺失或非法时，回退到该 API 的第一个可用模型；
    - 配置结构不合法或缺失关键节点时返回 `None`。
    """
    from gaoagent.core.runner.BaseRunner import RequestBaseInfo

    path = _find_config_file("gao_client_api_config.json")
    if path is None:
        return None
    if not path.exists():
        return None
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if not isinstance(config, dict):
        return None

    apis = config.get("apis", {})

    # ===================== 1. 处理 default_api 缺失 =====================
    default_api = config.get("default_api")
    if not default_api or default_api not in apis:
        # 兜底：取第一个 API
        if not apis:
            return None
        default_api = next(iter(apis.keys()))

    api_config = apis[default_api]
    api_key = api_config.get("api_key", "")
    base_url = api_config.get("base_url", "")
    models = api_config.get("models", {})

    # ===================== 2. 处理 default_model 缺失 =====================
    default_model = config.get("default_model")
    if not default_model or default_model not in models:
        # 兜底：取第一个模型
        if not models:
            return None
        default_model = next(iter(models.keys()))

    model_info = models[default_model]
    context_window = model_info.get("context_window", 4096)

    return RequestBaseInfo(
        baseurl=base_url,
        api_key=api_key,
        modules=default_model,
        context_window=context_window,
    )


def load_mcp_servers_raw() -> dict[str, Any]:
    """
    读取原始 MCP servers 配置（不裁剪字段）。

    用途：
    - ReActRunner 在真正调用 MCP 工具时，需要拿到完整 `command/args/timeout`。
 
    """
    path = _find_config_file("gao_client_mcp_setting.json")
    if path is None:
        return {}
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    servers = payload.get("mcpServers")
    return servers if isinstance(servers, dict) else {}


def load_mcp_tools_cache() -> dict[str, Any] | None:
    """
    读取 MCP 工具缓存。

    缓存由配置阶段生成，核心结构是：
    - `servers`: 各 server 的工具列表
    - `exported_map`: 导出工具名 -> (server/tool/schema) 映射
    """
    path = _find_config_file("gao_client_mcp_tools_cache.json")
    if path is None:
        return None
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def write_mcp_tools_cache_for_current_scope(payload: dict[str, Any]) -> None:
    """
    将 MCP 工具缓存写回当前生效配置作用域。

    规则：
    - 缓存统一写入项目级 `.gaoagent/` 目录。
    """
    cfg_dir = project_config_dir()
    if cfg_dir is None:
        return
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cfg_dir / "gao_client_mcp_tools_cache.json"
    tmp_path = cache_path.with_name(f"{cache_path.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(cache_path)


def load_history(session_id: str) -> list[dict[str, Any]] | None:
    """
    加载历史记录。
    
    返回:
    - 历史记录列表，如果不存在则返回 None。
    """
    cfg_dir = project_config_dir()
    if cfg_dir is None:
        return None
    history_dir = cfg_dir / "history"
    history_file = history_dir / f"{session_id}.json"
    if not history_file.exists() or not history_file.is_file():
        return None
    try:
        payload = json.loads(history_file.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
    except Exception:
        pass
    return None


def save_history(session_id: str, history: list[dict[str, Any]]) -> None:
    """
    保存历史记录。
    """
    cfg_dir = project_config_dir()
    if cfg_dir is None:
        return
    history_dir = cfg_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    history_file = history_dir / f"{session_id}.json"
    tmp_path = history_file.with_name(f"{history_file.name}.tmp")
    tmp_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(history_file)


def load_skills() -> dict[str, Any] | None:
    """
    加载技能元数据索引（用于提示词注入与能力展示）。

    返回结构:
    - `available`: 是否检测到技能目录；
    - `items`: 技能列表（仅保留 name/description/path）。
    """
    skills_cfg = _find_config_file("skills")
    if skills_cfg is None:
        return None
    skills_dir = skills_cfg.resolve()
    if not skills_dir.exists() or not skills_dir.is_dir():
        return None

    (skills, _) = scan_skills_metadata(skills_dir)
    items: list[dict[str, Any]] = []
    for item in skills:
        items.append(
            {
                "name": item.get("name"),
                "description": item.get("description"),
                "path": item.get("path"),
            }
        )
    return {"available": True, "items": items}


def parse_skill_frontmatter(file_path: Path) -> dict[str, Any] | None:
    """解析 Skill.md frontmatter，返回 name/description。"""
    (meta, _) = parse_skill_frontmatter_with_reason(file_path)
    return meta


def parse_skill_frontmatter_with_reason(
    file_path: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    """解析 Skill.md frontmatter，失败时返回原因。"""
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        return (None, f"读取失败：{e}")

    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return (None, "缺少 YAML frontmatter 起始标记 ---")

    frontmatter: list[str] = []
    i = 1
    while i < len(lines):
        if lines[i].strip() == "---":
            break
        frontmatter.append(lines[i])
        i += 1
    if i >= len(lines) or lines[i].strip() != "---":
        return (None, "缺少 YAML frontmatter 结束标记 ---")

    name = description = None
    j = 0
    while j < len(frontmatter):
        line = frontmatter[j].rstrip("\n")
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            j += 1
            continue

        if stripped.startswith("name:"):
            name = stripped.split(":", 1)[1].strip().strip("\"'")
            j += 1
            continue

        if stripped.startswith("description:"):
            desc_val = stripped.split(":", 1)[1].strip()
            if desc_val in (">", ">-", "|", "|-"):
                j += 1
                block: list[str] = []
                while j < len(frontmatter):
                    nxt = frontmatter[j]
                    if nxt.strip():
                        indent = len(nxt) - len(nxt.lstrip(" \t"))
                        if indent == 0:
                            break
                    block.append(nxt.lstrip())
                    j += 1
                description = " ".join(" ".join(block).split()).strip()
                continue
            else:
                description = desc_val.strip("\"'")
                j += 1
                continue
        j += 1

    if not name or not description:
        missing: list[str] = []
        if not name:
            missing.append("name")
        if not description:
            missing.append("description")
        return (None, f"缺少字段 {', '.join(missing)}")
    return ({"name": name, "description": description}, None)


def scan_skills_metadata(
    skills_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """
    扫描 skills 目录并做统一解析（尽力加载，不做规范性校验）：
    - 若 frontmatter 可解析，使用其 name/description；
    - 若不可解析，回退为目录名 + 空 description。
    """
    skills: list[dict[str, Any]] = []
    invalid_skills: list[dict[str, str]] = []
    if not skills_dir.exists() or not skills_dir.is_dir():
        return (skills, invalid_skills)

    for file_path in skills_dir.rglob("*"):
        if not file_path.is_file() or file_path.name.lower() != "skill.md":
            continue
        (meta, reason) = parse_skill_frontmatter_with_reason(file_path)
        if meta is None:
            # 尽力加载：解析失败时不拦截，仍提供可用的最小元数据。
            meta = {"name": file_path.parent.name, "description": ""}
            invalid_skills.append(
                {"path": str(file_path), "reason": reason or "解析失败"}
            )
        else:
            skills.append(
                {
                    "name": str(meta["name"]),
                    "description": str(meta["description"]),
                    "path": str(file_path),
                    "src_dir": file_path.parent,
                }
            )

    skills.sort(key=lambda x: str(x.get("name") or ""))
    return (skills, invalid_skills)


def load_rag() -> dict[str, Any] | None:
    """
    加载项目级 RAG 索引目录概览。

    返回结构:
    - `available`: 是否检测到 `rag` 目录；
    - `indexes`: 当前可见索引子目录名列表（已排序）。
    """
    rag_cfg = _find_config_file("rag")
    if rag_cfg is None:
        return None
    rag_dir = rag_cfg.resolve()
    if not rag_dir.exists() or not rag_dir.is_dir():
        return None

    indexes = sorted([p.name for p in rag_dir.iterdir() if p.is_dir()])
    return {"available": True, "indexes": indexes}


def parse_llm_response(response: HttpResponse) -> StepResult:
    """
    将底层 HTTP 客户端返回的 LLM 响应规范化为 StepResult。

    该函数是 Runner 的“协议收口层”，职责是把不同形态的上游响应
    （HTTP 失败、空响应、tool_calls、thought/final 文本协议等）
    统一映射为 StepResult，供主循环直接消费。

    解析约定：
    - 网络/HTTP 失败：返回 decision="final"，content 为可读错误摘要。
    - 响应体非 JSON 对象：返回 decision="retry"，提示上层重试。
    - 协议 type="tool_calls"：返回 decision="tool_calls"。
    - 协议 type="thought"：返回 decision="thought"。
    - 协议 type="final"：返回 decision="final"。
    - 协议 type="retry"：返回 decision="retry"。
    - 协议 type in {"question","observation"}：向后兼容，映射为 thought。
    - 其他未知 type：返回 decision="retry" 说明协议错误（通常可重试）。

    参数:
    - response: HttpClient 返回的 HttpResponse。

    返回:
    - StepResult: 统一后的单步决策结果。
    """
    from gaoagent.core.runner.HttpClient import HttpResponse
    from gaoagent.core.runner.BaseRunner import StepResult
    from gaoagent.core.runner.Console import Console

    if not isinstance(response, HttpResponse):
        Console.fatal(f"模型返回类型不对，收到的是：{type(response).__name__}。")
        return StepResult(
            decision="final",
            content=f"LLM 响应类型错误：{type(response).__name__}",
            raw={"response": repr(response)},
        )

    # HTTP 层失败（非 2xx 或网络异常）直接终止当前步，并返回可诊断信息。
    if not response.ok:
        status_text = (
            f"status={response.status}"
            if response.status is not None
            else "status=null"
        )
        reason_text = response.reason or "unknown"
        body = response.text or (
            safe_json_dumps(response.json) if response.json is not None else ""
        )
        content = f"LLM 请求失败：{status_text}, reason={reason_text}"
        if body:
            # 错误体可能很长，避免污染终端与日志。
            content = f"{content}\n{truncate_text(body, 800)}"
        Console.debug(
            safe_json_dumps(
                {
                    "event": "llm_response_not_ok",
                    "status": response.status,
                    "reason": response.reason,
                    "body_preview": summarize(body, 260),
                }
            )
        )
        Console.fatal(f"模型请求失败了：{status_text}，原因：{reason_text}")

        return StepResult(
            decision="final",
            content=content,
            raw={
                "http": {
                    "ok": response.ok,
                    "status": response.status,
                    "reason": response.reason,
                }
            },
        )

    # 优先使用已解析 JSON；若缺失则尝试从 text 二次反序列化。
    payload: dict[str, Any] | None = (
        response.json if isinstance(response.json, dict) else None
    )
    if payload is None and isinstance(response.text, str) and response.text.strip():
        try:
            parsed = json.loads(response.text)
        except Exception:
            parsed = None
        payload = parsed if isinstance(parsed, dict) else None

    # 上游返回空文本或非对象 JSON，视为可重试的临时协议异常。
    if payload is None:
        text = response.text or ""
        return StepResult(
            decision="retry",
            content=(
                truncate_text(text, 800) if text else "LLM 返回为空或不是 JSON 对象"
            ),
            raw={"http": {"ok": True, "status": response.status}, "text": text},
        )

    from gaoagent.core.runner.FunctionCallProtocol import (
        map_chat_completion_to_protocol,
    )

    # 统一协议映射：把 ChatCompletions payload 转成内部 protocol dict。
    protocol = map_chat_completion_to_protocol(payload)
    action_type = protocol.get("type") if isinstance(protocol, dict) else None

    if action_type == "tool_calls":
        # 新协议：支持多 tool call；每个调用保留 name/arguments/tool_call_id。
        calls: list[dict[str, Any]] = []
        protocol_calls = protocol.get("calls")
        if isinstance(protocol_calls, list):
            for item in protocol_calls:
                if not isinstance(item, dict):
                    continue
                call: dict[str, Any] = {
                    "name": item.get("name"),
                    "arguments": item.get("arguments", {}),
                }
                if isinstance(item.get("tool_call_id"), str):
                    call["tool_call_id"] = item.get("tool_call_id")
                calls.append(call)

        return StepResult(
            decision="tool_calls",
            tool_calls=calls,
            raw={"payload": payload, "protocol": protocol},
        )

    if action_type == "thought":
        # thought 允许 content 或 output，两者都缺失时回退为空字符串。
        content = protocol.get("content")
        if content is None:
            content = protocol.get("output")
        if content is None:
            content = ""
        return StepResult(
            decision="thought",
            content=str(content),
            raw={"payload": payload, "protocol": protocol},
        )

    if action_type == "final":
        # final 直接透传，交由上层作为本轮最终回答处理。
        content = protocol.get("content", "")
        return StepResult(
            decision="final",
            content=str(content),
            raw={"payload": payload, "protocol": protocol},
        )

    if action_type == "retry":
        content = protocol.get("content", "")
        return StepResult(
            decision="retry",
            content=str(content),
            raw={"payload": payload, "protocol": protocol},
        )

    if action_type in ("question", "observation"):
        # 历史兼容：旧类型在当前执行器中按 thought 处理，不直接中断流程。
        content = protocol.get("content", "")
        return StepResult(
            decision="thought",
            content=str(content),
            raw={"payload": payload, "protocol": protocol},
        )

    # 未识别协议类型：按可重试错误处理，避免直接终止任务。
    return StepResult(
        decision="retry",
        content=f"LLM 协议错误：未知 type={repr(action_type)}",
        raw={"payload": payload, "protocol": protocol},
    )
