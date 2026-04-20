from __future__ import annotations

"""
任务执行器（Runner）骨架。

该文件提供 3 种典型 Agent 执行模式的可运行骨架（尚未接入 LLM）：
1) Plan: 先生成结构化计划（plan），再按计划逐步执行。
2) ReAct: 思考-行动-观察循环（由 policy 决策下一步）。
3) Retry: 在失败时反思并生成修复策略，在 max_retries 约束下重试。

核心设计点：
- BaseRunner 统一 step 循环、终止条件、工具调用入口与失败归一化；
- ToolRegistry 提供可插拔的 tool 注册/分发；
- AuditLogger 将每一步的审计信息写入 jsonl（便于离线排查与复盘）。
"""

import json
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import click

Mode = Literal["plan", "react", "retry"]


def _now_ms() -> int:
    """
    获取当前 Unix 时间戳（毫秒）。
    """
    return int(time.time() * 1000)


def _truncate_text(s: str, limit: int) -> str:
    """
    将字符串截断到指定长度，用于日志/审计摘要，避免单条记录过大。
    """
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 1)] + "…"


def _safe_json_dumps(value: Any) -> str:
    """
    尽可能将对象序列化为 JSON 字符串；失败时回退到 repr。
    """
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return repr(value)


def _summarize(value: Any, limit: int = 400) -> str:
    """
    将任意对象转换为可审计的短摘要字符串（默认 400 字符以内）。

    目的：
    - 记录足够的信息用于排查；
    - 避免把完整 payload（可能很大或包含敏感信息）直接写入日志。
    """
    if value is None:
        return "null"
    if isinstance(value, (str, int, float, bool)):
        return _truncate_text(str(value), limit)
    return _truncate_text(_safe_json_dumps(value), limit)


def _redact(value: Any) -> Any:
    """
    对可能包含密钥的字段做脱敏处理。

    注意：这是“尽力而为”的轻量脱敏，仅覆盖常见键名。
    """
    sensitive_keys = {"api_key", "apikey", "key", "token", "secret", "password", "authorization"}
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if isinstance(k, str) and k.lower() in sensitive_keys:
                out[k] = "***"
            else:
                out[k] = _redact(v)
        return out
    if isinstance(value, list):
        return [_redact(x) for x in value]
    return value


def _normalize_exception(e: BaseException) -> dict[str, Any]:
    """
    将异常对象归一化为可序列化 dict，便于审计与跨层传递。
    """
    return {
        "type": type(e).__name__,
        "message": str(e),
        "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
    }


@dataclass(frozen=True)
class ToolCall:
    """
    一次工具调用的抽象表示。

    - name: 工具名（与 ToolRegistry.register 的 name 一致）
    - arguments: 工具入参（必须可 JSON 序列化，审计时会做摘要与脱敏）
    """
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunnerConfig:
    """
    Runner 的通用运行参数。

    - max_steps: 单次 run 允许的最大 step 数
    - max_retries: RetryRunner 允许的最大重试次数
    - audit_path: 审计日志路径（默认 ~/.gaoagent/audit.jsonl）
    - console: 是否输出到终端（click.echo）
    """
    max_steps: int = 32
    max_retries: int = 2
    audit_path: Path | None = None
    console: bool = True


@dataclass
class RunnerContext:
    """
    Runner 执行上下文（跨 step 持久）。

    - question: 用户问题/任务描述
    - mode: 当前 runner 模式
    - plan: PlanRunner 生成的结构化计划（若已生成）
    - memory: 共享状态字典（用于在 runner/step/attempt 之间传递信息）
    - last_observation: 上一次工具执行输出（ReAct/Plan 常用）
    - last_error: 上一次失败信息（Retry 常用）
    - step: 当前 step 计数（从 1 开始）
    - attempt: 当前 retry attempt（从 0 开始）
    """
    question: str
    mode: Mode
    plan: list[dict[str, Any]] | None = None
    memory: dict[str, Any] = field(default_factory=dict)
    last_observation: Any | None = None
    last_error: dict[str, Any] | None = None
    step: int = 0
    attempt: int = 0


@dataclass(frozen=True)
class RunnerResult:
    """
    Runner.run 的结果。

    - ok: 是否成功
    - final: 成功时的最终输出（可为空）
    - error: 失败归一化后的错误信息
    """
    ok: bool
    final: str | None = None
    error: dict[str, Any] | None = None


@dataclass(frozen=True)
class Decision:
    """
    Runner 在某个 step 的“决策”结果。

    kind:
    - "tool": 调用一个工具（tool_call 必填）
    - "final": 结束并返回最终输出（final 必填）
    - "internal": 仅更新内部状态或记录信息，不调用工具（internal 可选）
    - "stop": 立即停止（不视为失败）
    """
    kind: Literal["tool", "final", "internal", "stop"]
    tool_call: ToolCall | None = None
    final: str | None = None
    internal: dict[str, Any] | None = None


ToolHandler = Callable[[RunnerContext, dict[str, Any]], Any]


class ToolRegistry:
    """
    工具注册表与分发器。

    ToolHandler 签名：handler(ctx, arguments) -> Any
    - ctx: RunnerContext（可读写，用于工具更新状态）
    - arguments: 该次工具调用的参数
    - 返回值会被写入 ctx.last_observation，并被审计摘要记录
    """
    def __init__(self) -> None:
        self._tools: dict[str, ToolHandler] = {}

    def register(self, name: str, handler: ToolHandler) -> None:
        """
        注册工具。

        - name: 工具名（字符串，唯一键）
        - handler: 工具处理函数
        """
        if not name or not isinstance(name, str):
            raise ValueError("tool name must be non-empty str")
        self._tools[name] = handler

    def call(self, ctx: RunnerContext, call: ToolCall) -> Any:
        """
        调用工具；找不到工具时抛出 KeyError。
        """
        handler = self._tools.get(call.name)
        if handler is None:
            raise KeyError(f"tool not found: {call.name}")
        return handler(ctx, call.arguments)

    def list_names(self) -> list[str]:
        """
        返回已注册工具名列表（用于构建 function-call 工具声明）。
        """
        return sorted(self._tools.keys())


class AuditLogger:
    """
    审计日志写入器（jsonl）。

    默认写入 ~/.gaoagent/audit.jsonl；在受限环境（如沙箱或权限不足）下，
    会回退写入到当前工作目录的 ./.gaoagent/audit.jsonl，以保证 runner 仍可运行。
    """
    def __init__(self, audit_path: Path) -> None:
        self._path = audit_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fallback_path: Path | None = None

    @property
    def path(self) -> Path:
        """
        主审计文件路径（不包含回退路径）。
        """
        return self._path

    def write(self, record: dict[str, Any]) -> None:
        """
        追加写入一条审计记录（jsonl 单行）。

        record 约定字段（由 BaseRunner._write_step 生成）：
        - mode/step/attempt/tool/input_summary/output_summary/elapsed_ms/status/error
        """
        payload = dict(record)
        payload["ts_ms"] = payload.get("ts_ms") or _now_ms()
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        try:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            return
        except OSError:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
                return
            except OSError:
                pass

        if self._fallback_path is None:
            fallback_dir = Path.cwd() / ".gaoagent"
            fallback_dir.mkdir(parents=True, exist_ok=True)
            self._fallback_path = fallback_dir / "audit.jsonl"

        try:
            with self._fallback_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            return


def _default_audit_path() -> Path:
    """
    默认审计路径：~/.gaoagent/audit.jsonl
    """
    config_dir = Path.home() / ".gaoagent"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "audit.jsonl"


def _default_tool_registry() -> ToolRegistry:
    """
    默认工具集合（仅用于骨架可运行）。

    目前只提供 echo 工具：
    - echo(arguments) -> {"echo": arguments}
    """
    tools = ToolRegistry()

    def _echo(_ctx: RunnerContext, args: dict[str, Any]) -> dict[str, Any]:
        return {"echo": args}

    tools.register("echo", _echo)
    return tools


class BaseRunner:
    """
    Runner 抽象基类。

    BaseRunner 负责：
    - step 循环；
    - 调用子类 decide(ctx) 获取下一步 Decision；
    - 执行 tool 调用并更新 ctx.last_observation；
    - 对异常进行失败归一化并写入审计；
    - 统一输出每 step 的审计（以及可选的 console 输出）。
    """
    def __init__(
        self,
        *,
        mode: Mode,
        tools: ToolRegistry | None = None,
        audit: AuditLogger | None = None,
        config: RunnerConfig | None = None,
    ) -> None:
        self._mode: Mode = mode
        self._tools = tools or _default_tool_registry()
        cfg = config or RunnerConfig()
        audit_path = cfg.audit_path or _default_audit_path()
        self._audit = audit or AuditLogger(audit_path)
        self._cfg = cfg

    def decide(self, ctx: RunnerContext) -> Decision:
        """
        子类需要实现的决策函数。

        约定：
        - 返回 Decision(kind="tool") 表示要调用一个工具；
        - 返回 Decision(kind="final") 表示输出最终结果并结束；
        - 返回 Decision(kind="internal") 表示仅内部状态变更；
        - 返回 Decision(kind="stop") 表示提前停止（不视为失败）。
        """
        raise NotImplementedError()

    def run(self, question: str, shared_memory: dict[str, Any] | None = None) -> RunnerResult:
        """
        执行任务主循环。

        - question: 用户输入/任务描述
        - shared_memory: 外部注入的共享 dict（用于跨 runner/attempt 复用状态）
        """
        if question is None or not str(question).strip():
            return RunnerResult(ok=False, error={"type": "ValueError", "message": "question is empty"})

        ctx = RunnerContext(question=str(question).strip(), mode=self._mode, memory=shared_memory or {})
        for step in range(1, self._cfg.max_steps + 1):
            ctx.step = step
            t0 = time.perf_counter()
            decision: Decision | None = None
            tool_name = ""
            input_summary = ""
            output_summary = ""
            err: dict[str, Any] | None = None
            status: Literal["ok", "error", "stop"] = "ok"
            try:
                decision = self.decide(ctx)
                if decision.kind == "stop":
                    status = "stop"
                    tool_name = "stop"
                    output_summary = "stopped"
                    self._write_step(ctx, tool_name, input_summary, output_summary, status, err, t0)
                    return RunnerResult(ok=True, final=None)

                if decision.kind == "final":
                    status = "stop"
                    tool_name = "final"
                    input_summary = _summarize({"question": ctx.question})
                    output_summary = _summarize(decision.final)
                    self._write_step(ctx, tool_name, input_summary, output_summary, status, err, t0)
                    return RunnerResult(ok=True, final=decision.final)

                if decision.kind == "internal":
                    tool_name = (decision.internal or {}).get("name") or "internal"
                    input_summary = _summarize(_redact((decision.internal or {}).get("input")))
                    output_summary = _summarize((decision.internal or {}).get("output"))
                    self._write_step(ctx, tool_name, input_summary, output_summary, status, err, t0)
                    continue

                if decision.kind != "tool" or decision.tool_call is None:
                    status = "error"
                    tool_name = "invalid_decision"
                    err = {"type": "RuntimeError", "message": "invalid decision"}
                    self._write_step(ctx, tool_name, input_summary, output_summary, status, err, t0)
                    return RunnerResult(ok=False, error=err)

                tool_name = decision.tool_call.name
                input_summary = _summarize(_redact(decision.tool_call.arguments))
                raw_out = self._tools.call(ctx, decision.tool_call)
                ctx.last_observation = raw_out
                output_summary = _summarize(raw_out)
                self._write_step(ctx, tool_name, input_summary, output_summary, status, err, t0)
            except Exception as e:
                status = "error"
                err = _normalize_exception(e)
                ctx.last_error = err
                if not tool_name:
                    tool_name = "error"
                self._write_step(ctx, tool_name, input_summary, output_summary, status, err, t0)
                return RunnerResult(ok=False, error=err)

        err = {"type": "RuntimeError", "message": f"max_steps reached: {self._cfg.max_steps}"}
        ctx.last_error = err
        self._write_step(ctx, "max_steps", "", "", "error", err, time.perf_counter())
        return RunnerResult(ok=False, error=err)

    def _write_step(
        self,
        ctx: RunnerContext,
        tool: str,
        input_summary: str,
        output_summary: str,
        status: Literal["ok", "error", "stop"],
        err: dict[str, Any] | None,
        t0: float,
    ) -> None:
        """
        写入单步审计记录，并在 console 模式下输出一行简要状态。
        """
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        record = {
            "mode": ctx.mode,
            "step": ctx.step,
            "attempt": ctx.attempt,
            "tool": tool,
            "input_summary": input_summary,
            "output_summary": output_summary,
            "elapsed_ms": elapsed_ms,
            "status": status,
            "error": err,
        }
        self._audit.write(record)
        if self._cfg.console:
            click.echo(f"[{ctx.mode}] step={ctx.step} tool={tool} status={status} ms={elapsed_ms}")
            if status == "error" and err:
                click.echo(_truncate_text(err.get("message", ""), 400))


Planner = Callable[[RunnerContext], list[dict[str, Any]]]


class PlanRunner(BaseRunner):
    """
    计划驱动 Runner。

    工作方式：
    1) 第一次 decide 时生成结构化计划 plan（list[dict]），并存入 ctx.memory["plan"]；
    2) 后续每一步按 ctx.memory["plan_index"] 逐条执行。

    计划项约定：
    - {"type": "tool", "name": "...", "arguments": {...}}
    - {"type": "final", "content": "..."}  # 直接结束并输出
    """
    def __init__(
        self,
        *,
        tools: ToolRegistry | None = None,
        audit: AuditLogger | None = None,
        config: RunnerConfig | None = None,
        planner: Planner | None = None,
    ) -> None:
        super().__init__(mode="plan", tools=tools, audit=audit, config=config)
        self._planner = planner or self._default_planner

    def decide(self, ctx: RunnerContext) -> Decision:
        if ctx.plan is None:
            plan = self._planner(ctx)
            ctx.plan = plan
            ctx.memory["plan"] = plan
            ctx.memory.setdefault("plan_index", 0)
            return Decision(
                kind="internal",
                internal={"name": "planner", "input": {"question": ctx.question}, "output": {"plan_len": len(plan)}},
            )

        idx = int(ctx.memory.get("plan_index", 0))
        if idx >= len(ctx.plan):
            return Decision(kind="final", final="计划执行完成")

        item = ctx.plan[idx]
        ctx.memory["plan_index"] = idx + 1
        if not isinstance(item, dict):
            return Decision(kind="final", final=f"计划项格式错误：{repr(item)}")

        t = item.get("type")
        if t == "final":
            return Decision(kind="final", final=str(item.get("content") or ""))
        if t == "tool":
            name = str(item.get("name") or "")
            args = item.get("arguments") or {}
            if not isinstance(args, dict):
                args = {"value": args}
            return Decision(kind="tool", tool_call=ToolCall(name=name, arguments=args))

        return Decision(kind="internal", internal={"name": "plan_item", "input": item, "output": {"skipped": True}})

    def _default_planner(self, ctx: RunnerContext) -> list[dict[str, Any]]:
        """
        默认 planner：仅返回一条提示信息。

        接入 LLM 后，可替换为：
        planner(ctx) -> [{"type":"tool",...}, ...]
        """
        return [
            {
                "type": "final",
                "content": "PlanRunner 未配置 planner（LLM/规则）。请注入 planner(ctx)->plan 后再运行。",
            }
        ]


Policy = Callable[[RunnerContext], Decision]


class ReActRunner(BaseRunner):
    """
    ReAct（Reason + Act）模式 Runner。

    该 Runner 将“下一步要做什么”完全交给 policy 决策：
    - policy(ctx) -> Decision(kind="tool"/"final"/"internal"/"stop")

    常见实现会在 ctx.memory 中维护：
    - message history / scratchpad
    - tool 执行历史
    - last_observation 等
    """
    def __init__(
        self,
        *,
        tools: ToolRegistry | None = None,
        audit: AuditLogger | None = None,
        config: RunnerConfig | None = None,
        policy: Policy | None = None,
    ) -> None:
        super().__init__(mode="react", tools=tools, audit=audit, config=config)
        self._policy = policy or self._default_policy

    def decide(self, ctx: RunnerContext) -> Decision:
        return self._policy(ctx)

    def _default_policy(self, ctx: RunnerContext) -> Decision:
        """
        默认 policy（基于 function-call 协议）。

        规范定义：
        1) 传给 LLM 的工具声明（tools）:
           {
             "type": "function",
             "function": {
               "name": "<tool_name>",
               "description": "<tool_description>",
               "parameters": {"type":"object","additionalProperties": true}
             }
           }

        2) 期望 LLM 返回（response）:
           - 工具调用：
             {
               "type": "function_call",
               "name": "<tool_name>",
               "arguments": { ... }   # 必须是对象
             }
           - 直接结束：
             {
               "type": "final",
               "content": "<final_text>"
             }
           - 内部思考（可选，不执行工具）：
             {
               "type": "internal",
               "name": "<internal_name>",
               "input": {...},
               "output": {...}
             }

        3) 转换规则：
           - function_call -> Decision(kind="tool", tool_call=...)
           - final         -> Decision(kind="final", final=...)
           - internal      -> Decision(kind="internal", internal=...)
           - 非法结构      -> Decision(kind="final", final="协议解析失败...")
        """
        messages = self._build_react_messages(ctx)
        tools = self._build_function_specs()
        llm_raw = self._call_llm_function_call(ctx=ctx, messages=messages, tools=tools)
        if llm_raw is None:
            return Decision(
                kind="final",
                final="ReActRunner 的 LLM 调用尚未实现。请实现 _call_llm_function_call 后再运行。",
            )
        return self._parse_llm_response_to_decision(ctx, llm_raw)

    def _build_react_messages(self, ctx: RunnerContext) -> list[dict[str, Any]]:
        """
        构建给 LLM 的最小消息体。

        说明：
        - 这里先给出最小可用上下文，后续接入真实会话时可扩展为完整 history。
        - 为了便于排查，附带 step / last_observation / last_error。
        """
        system_text = (
            "你是一个 ReAct Agent。你必须遵循 function-call 响应协议："
            "仅返回 JSON 对象，type 只能是 function_call/final/internal。"
        )
        user_payload = {
            "question": ctx.question,
            "step": ctx.step,
            "last_observation": ctx.last_observation,
            "last_error": ctx.last_error,
        }
        return [
            {"role": "system", "content": system_text},
            {"role": "user", "content": _safe_json_dumps(user_payload)},
        ]

    def _build_function_specs(self) -> list[dict[str, Any]]:
        """
        将 ToolRegistry 中已注册工具映射为 function-call 工具声明。

        由于当前项目尚无每个工具的参数 schema 元数据，这里采用宽松 schema：
        - parameters = {"type":"object","additionalProperties": true}
        """
        specs: list[dict[str, Any]] = []
        for name in self._tools.list_names():
            specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": f"Tool `{name}` registered in ToolRegistry",
                        "parameters": {"type": "object", "additionalProperties": True},
                    },
                }
            )
        return specs

    def _call_llm_function_call(
        self,
        *,
        ctx: RunnerContext,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """
        调用 LLM（真实实现，基于 OpenAI 兼容 chat/completions）。

        配置来源：
        - ~/.gaoagent/gao_client_api_config.json
        - 结构：{"apis": { "<api_name>": {"base_url","api_key","models"} } }
        - 默认使用第一组 API 与第一组 model；可通过 ctx.memory["api_name"]/ctx.memory["model"] 覆盖。
        """
        config_path = Path.home() / ".gaoagent" / "gao_client_api_config.json"
        if not config_path.exists():
            return {"type": "final", "content": f"未找到 API 配置文件：{config_path}"}

        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as e:
            return {"type": "final", "content": f"读取 API 配置失败：{e}"}

        apis = payload.get("apis") if isinstance(payload, dict) else None
        if not isinstance(apis, dict) or not apis:
            return {"type": "final", "content": "API 配置无效：缺少 apis 或 apis 为空"}

        api_name = str(ctx.memory.get("api_name") or "").strip()
        selected_api: dict[str, Any] | None = None
        selected_api_name = ""
        if api_name:
            candidate = apis.get(api_name)
            if isinstance(candidate, dict):
                selected_api = candidate
                selected_api_name = api_name
            else:
                return {"type": "final", "content": f"未找到指定 API 配置：{api_name}"}
        else:
            selected_api_name, selected_api = next(iter(apis.items()))
            if not isinstance(selected_api, dict):
                return {"type": "final", "content": f"API 配置格式无效：{selected_api_name}"}

        base_url = str(selected_api.get("base_url") or "").strip()
        api_key = str(selected_api.get("api_key") or "").strip()
        models = selected_api.get("models")
        if not base_url:
            return {"type": "final", "content": f"API 配置缺少 base_url：{selected_api_name}"}
        if not api_key:
            return {"type": "final", "content": f"API 配置缺少 api_key：{selected_api_name}"}
        if not isinstance(models, dict) or not models:
            return {"type": "final", "content": f"API 配置缺少 models：{selected_api_name}"}

        model = str(ctx.memory.get("model") or "").strip()
        if model:
            if model not in models:
                return {"type": "final", "content": f"未找到指定模型：{model}"}
        else:
            model = next(iter(models.keys()))

        url = self._build_chat_completions_url(base_url)
        req_payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.2,
        }
        req_bytes = json.dumps(req_payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url=url,
            data=req_bytes,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                resp_text = response.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")
            except Exception:
                body = ""
            return {
                "type": "final",
                "content": f"LLM HTTPError: status={e.code}, reason={e.reason}, body={_truncate_text(body, 500)}",
            }
        except Exception as e:
            return {"type": "final", "content": f"LLM 请求失败：{type(e).__name__}: {e}"}

        try:
            resp_json = json.loads(resp_text)
        except Exception:
            return {"type": "final", "content": f"LLM 返回非 JSON：{_truncate_text(resp_text, 500)}"}

        return self._map_chat_completion_to_protocol(resp_json)

    def _build_chat_completions_url(self, base_url: str) -> str:
        """
        规范化 base_url，生成 chat/completions 地址。
        """
        clean = base_url.strip().rstrip("/")
        if clean.endswith("/chat/completions"):
            return clean
        if clean.endswith("/v1"):
            return f"{clean}/chat/completions"
        return f"{clean}/v1/chat/completions"

    def _map_chat_completion_to_protocol(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        将 OpenAI 兼容 chat/completions 响应映射到本地 function-call 协议。
        """
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return {"type": "final", "content": f"LLM 响应缺少 choices：{_summarize(payload)}"}

        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first, dict) else None
        if not isinstance(message, dict):
            return {"type": "final", "content": f"LLM 响应缺少 message：{_summarize(first)}"}

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            first_call = tool_calls[0] if isinstance(tool_calls[0], dict) else {}
            fn = first_call.get("function") if isinstance(first_call, dict) else None
            if isinstance(fn, dict):
                name = fn.get("name")
                args = self._parse_tool_arguments(fn.get("arguments"))
                return {"type": "function_call", "name": name, "arguments": args}

        content = message.get("content")
        if isinstance(content, str):
            return {"type": "final", "content": content}
        if isinstance(content, list):
            # 兼容部分多模态返回结构：[{"type":"text","text":"..."}]
            texts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        texts.append(text)
            if texts:
                return {"type": "final", "content": "\n".join(texts)}

        return {"type": "final", "content": "LLM 未返回可执行 tool_call，也未返回文本结果"}

    def _parse_tool_arguments(self, raw: Any) -> dict[str, Any]:
        """
        解析 tool 参数：
        - dict: 直接返回
        - str: 按 JSON 解析；失败时退化为 {"_raw": "..."}
        - 其他: 返回空对象
        """
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
                return {"value": parsed}
            except Exception:
                return {"_raw": text}
        return {}

    def _parse_llm_response_to_decision(self, ctx: RunnerContext, payload: dict[str, Any]) -> Decision:
        """
        将 LLM 响应（function-call 协议）转换为内部 Decision。
        """
        if not isinstance(payload, dict):
            return Decision(kind="final", final=f"LLM 响应必须是对象，实际是：{type(payload).__name__}")

        action_type = payload.get("type")
        if action_type == "function_call":
            name = payload.get("name")
            args = payload.get("arguments", {})
            if not isinstance(name, str) or not name.strip():
                return Decision(kind="final", final="协议错误：function_call.name 必须是非空字符串")
            if name not in self._tools.list_names():
                return Decision(kind="final", final=f"协议错误：未知工具 {name}")
            if not isinstance(args, dict):
                return Decision(kind="final", final="协议错误：function_call.arguments 必须是对象")
            return Decision(kind="tool", tool_call=ToolCall(name=name, arguments=args))

        if action_type == "final":
            content = payload.get("content", "")
            return Decision(kind="final", final=str(content))

        if action_type == "internal":
            inner_name = payload.get("name") or "internal"
            return Decision(
                kind="internal",
                internal={
                    "name": str(inner_name),
                    "input": payload.get("input"),
                    "output": payload.get("output"),
                },
            )

        # 协议兜底：给出可读错误，避免抛异常导致循环硬失败
        return Decision(
            kind="final",
            final=(
                "协议错误：LLM 响应 type 必须是 function_call/final/internal。"
                f" 当前为 {repr(action_type)}，step={ctx.step}"
            ),
        )


Reflector = Callable[[RunnerContext, dict[str, Any]], dict[str, Any]]


class RetryRunner:
    """
    失败重试 Runner。

    RetryRunner 并不直接继承 BaseRunner，而是“包一层”内部 runner：
    - inner_factory: (audit, config, tools) -> BaseRunner
    - 每次 attempt 运行 inner.run(...)
    - 如果失败，则调用 reflector(ctx, err) 生成反思/修复策略，并写入 shared_memory["retry_reflections"]
    - 重试直到 success 或 max_retries 用尽
    """
    def __init__(
        self,
        *,
        inner_factory: Callable[[AuditLogger, RunnerConfig, ToolRegistry], BaseRunner] | None = None,
        tools: ToolRegistry | None = None,
        audit: AuditLogger | None = None,
        config: RunnerConfig | None = None,
        reflector: Reflector | None = None,
    ) -> None:
        self._cfg = config or RunnerConfig()
        self._tools = tools or _default_tool_registry()
        audit_path = self._cfg.audit_path or _default_audit_path()
        self._audit = audit or AuditLogger(audit_path)
        self._inner_factory = inner_factory or (lambda a, c, t: ReActRunner(audit=a, config=c, tools=t))
        self._reflector = reflector or self._default_reflector

    def run(self, question: str, shared_memory: dict[str, Any] | None = None) -> RunnerResult:
        """
        在 max_retries 约束下重试执行。
        """
        memory = shared_memory or {}
        for attempt in range(0, self._cfg.max_retries + 1):
            t0 = time.perf_counter()
            ctx = RunnerContext(question=str(question).strip(), mode="retry", memory=memory, attempt=attempt, step=attempt + 1)
            try:
                self._audit.write(
                    {
                        "mode": "retry",
                        "step": ctx.step,
                        "attempt": ctx.attempt,
                        "tool": "attempt",
                        "input_summary": _summarize({"question": ctx.question}),
                        "output_summary": "",
                        "elapsed_ms": 0,
                        "status": "ok",
                        "error": None,
                    }
                )
                inner = self._inner_factory(self._audit, self._cfg, self._tools)
                result = inner.run(question, shared_memory=memory)
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                if self._cfg.console:
                    click.echo(f"[retry] attempt={attempt} ok={result.ok} ms={elapsed_ms}")
                if result.ok:
                    return result

                err = result.error or {"type": "RuntimeError", "message": "unknown error"}
                reflection = self._reflector(ctx, err)
                memory.setdefault("retry_reflections", []).append(reflection)
                self._audit.write(
                    {
                        "mode": "retry",
                        "step": ctx.step,
                        "attempt": ctx.attempt,
                        "tool": "reflect",
                        "input_summary": _summarize(_redact(err)),
                        "output_summary": _summarize(reflection),
                        "elapsed_ms": 0,
                        "status": "ok",
                        "error": None,
                    }
                )
            except Exception as e:
                err = _normalize_exception(e)
                self._audit.write(
                    {
                        "mode": "retry",
                        "step": ctx.step,
                        "attempt": ctx.attempt,
                        "tool": "error",
                        "input_summary": "",
                        "output_summary": "",
                        "elapsed_ms": 0,
                        "status": "error",
                        "error": err,
                    }
                )
                return RunnerResult(ok=False, error=err)

        return RunnerResult(ok=False, error={"type": "RuntimeError", "message": f"max_retries reached: {self._cfg.max_retries}"})

    def _default_reflector(self, ctx: RunnerContext, err: dict[str, Any]) -> dict[str, Any]:
        """
        默认反思器：生成最小 no-op 策略。
        """
        return {"attempt": ctx.attempt, "error_type": err.get("type"), "strategy": "no-op"}


class TaskRunner:
    """
    CLI/入口层的 Runner 适配器。

    根据 mode 选择具体 runner 执行：
    - plan  -> PlanRunner
    - react -> ReActRunner
    - retry -> RetryRunner（内部默认用 ReActRunner）
    """
    def __init__(
        self,
        *,
        tools: ToolRegistry | None = None,
        config: RunnerConfig | None = None,
    ) -> None:
        self._cfg = config or RunnerConfig()
        self._tools = tools or _default_tool_registry()

    def run(self, question: str, mode: str) -> None:
        """
        执行任务并将结果输出到终端。

        说明：
        - 该方法是“命令式输出”，而非返回结构化结果，便于 CLI 使用；
        - 若要在程序内二次封装，建议直接调用各 Runner.run 获得 RunnerResult。
        """
        m = (mode or "react").strip().lower()
        if m not in ("plan", "react", "retry"):
            m = "react"

        shared_memory: dict[str, Any] = {}
        if m == "plan":
            result = PlanRunner(config=self._cfg, tools=self._tools).run(question, shared_memory=shared_memory)
        elif m == "retry":
            result = RetryRunner(config=self._cfg, tools=self._tools).run(question, shared_memory=shared_memory)
        else:
            result = ReActRunner(config=self._cfg, tools=self._tools).run(question, shared_memory=shared_memory)

        if result.ok:
            if result.final:
                click.echo(result.final)
            return

        err = result.error or {"type": "RuntimeError", "message": "unknown error"}
        click.echo(f"任务失败：{err.get('type')}: {err.get('message')}")
