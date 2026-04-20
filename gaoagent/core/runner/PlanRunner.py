from __future__ import annotations

import json
from typing import Any, Callable

from gaoagent.core.runner.ApiConfig import default_api_config_path, load_api_config, select_api_and_model
from gaoagent.core.runner.AuditLogger import AuditLogger
from gaoagent.core.runner.BaseRunner import BaseRunner, Decision, RunnerConfig, RunnerContext
from gaoagent.core.runner.FunctionCallProtocol import http_error_to_final
from gaoagent.core.runner.HttpClient import OpenAICompatibleHttpClient
from gaoagent.core.runner.PromptBuilder import build_messages
from gaoagent.core.runner.Tooling import ToolCall, ToolRegistry
from gaoagent.core.runner.Utils import summarize, truncate_text

Planner = Callable[[RunnerContext], list[dict[str, Any]]]


class PlanRunner(BaseRunner):
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
        def extract_message_text(payload: dict[str, Any]) -> str:
            choices = payload.get("choices")
            if not isinstance(choices, list) or not choices:
                return ""
            first = choices[0] if isinstance(choices[0], dict) else {}
            message = first.get("message") if isinstance(first, dict) else None
            if not isinstance(message, dict):
                return ""

            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts: list[str] = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text")
                        if isinstance(text, str):
                            texts.append(text)
                return "\n".join(texts)
            return ""

        def strip_code_fence(text: str) -> str:
            s = (text or "").strip()
            if s.startswith("```"):
                lines = s.splitlines()
                if len(lines) >= 2 and lines[0].startswith("```"):
                    i = 1
                    while i < len(lines) and lines[i].strip() == "":
                        i += 1
                    if i < len(lines) and lines[-1].strip() == "```":
                        return "\n".join(lines[i:-1]).strip()
            return s

        tool_names = self._tools.list_names()
        messages = build_messages(ctx, tool_names=tool_names, mode="plan")

        try:
            config_payload = load_api_config(default_api_config_path())
            selection = select_api_and_model(config_payload, ctx.memory)
        except FileNotFoundError as e:
            return [{"type": "final", "content": f"未找到 API 配置文件：{e}"}]
        except KeyError as e:
            return [{"type": "final", "content": str(e)}]
        except Exception as e:
            return [{"type": "final", "content": f"读取/选择 API 配置失败：{e}"}]

        netlog_path = None
        if isinstance(ctx.memory, dict):
            netlog_path = ctx.memory.get("netlog_path")
        client = OpenAICompatibleHttpClient(
            base_url=selection.base_url,
            api_key=selection.api_key,
            timeout_s=60,
            network_log_path=netlog_path,
        )
        url = client.build_chat_completions_url()
        req_payload: dict[str, Any] = {"model": selection.model, "messages": messages, "temperature": 0.2}
        resp = client.post_json(url, req_payload)
        if not resp.ok:
            body = resp.text or ""
            if resp.status is not None:
                final = http_error_to_final(resp.status, resp.reason or "", body)
                return [{"type": "final", "content": str(final.get("content") or "")}]
            return [{"type": "final", "content": f"LLM 请求失败：{resp.reason}"}]

        if resp.json is None:
            return [{"type": "final", "content": f"LLM 返回非 JSON：{truncate_text(resp.text or '', 500)}"}]

        raw_text = extract_message_text(resp.json)
        if not raw_text.strip():
            return [{"type": "final", "content": f"LLM 未返回可解析内容：{summarize(resp.json)}"}]

        raw_text = strip_code_fence(raw_text)
        try:
            parsed = json.loads(raw_text)
        except Exception:
            return [{"type": "final", "content": f"LLM 规划结果不是合法 JSON：{truncate_text(raw_text, 800)}"}]

        if isinstance(parsed, dict) and isinstance(parsed.get("plan"), list):
            parsed = parsed.get("plan")
        elif isinstance(parsed, dict):
            parsed = [parsed]

        if not isinstance(parsed, list):
            return [{"type": "final", "content": f"LLM 规划结果必须是数组：{summarize(parsed)}"}]

        plan: list[dict[str, Any]] = []
        tool_set = set(tool_names)
        for item in parsed:
            if not isinstance(item, dict):
                return [{"type": "final", "content": f"计划项必须是对象：{summarize(item)}"}]
            t = item.get("type")
            if t == "final":
                plan.append({"type": "final", "content": str(item.get("content") or "")})
                continue
            if t == "tool":
                name = item.get("name")
                if not isinstance(name, str) or not name.strip():
                    return [{"type": "final", "content": f"计划项 tool.name 必须是非空字符串：{summarize(item)}"}]
                if name not in tool_set:
                    return [{"type": "final", "content": f"计划项包含未知工具：{name}"}]
                args = item.get("arguments", {})
                if not isinstance(args, dict):
                    return [{"type": "final", "content": f"计划项 tool.arguments 必须是对象：{summarize(item)}"}]
                plan.append({"type": "tool", "name": name, "arguments": args})
                continue
            plan.append({"type": "internal", "name": "plan_item", "input": item})

        if not plan:
            return [{"type": "final", "content": "LLM 规划结果为空"}]
        return plan
