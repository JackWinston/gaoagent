import asyncio
from pathlib import Path
from typing import Any, Optional

import click
import uvicorn
import httpx
from gaoagent.core.runner.console import Console
from gaoagent.core.handler_utils import read_json_file, write_json, resolve_scope_and_config_path

try:
    from a2a.server.apps import A2AStarletteApplication
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import InMemoryTaskStore, BaseTaskExecutor
    from a2a.types import AgentCard, AgentCapabilities, Skill, Task, Artifact, Part, Message
    from a2a.client import A2AClient
    from gaoagent.agent.gao_task_executor import GaoTaskExecutor
    HAS_A2A = True
except ImportError:
    HAS_A2A = False

class AgentHandlers:
    _A2A_CONFIG_FILENAME = "gao_client_a2a_setting.json"

    def __init__(self, agent=None):
        self.agent = agent

    def _resolve_scope_and_config_path(self) -> tuple[str, Path]:
        return resolve_scope_and_config_path(self._A2A_CONFIG_FILENAME)

    def _load_agents(self, file_path: Path) -> dict[str, Any]:
        payload = read_json_file(file_path)
        if not isinstance(payload, dict):
            return {}
        return payload.get("agents", {})

    def _write_agents(self, file_path: Path, agents: dict[str, Any]) -> None:
        write_json(file_path, {"agents": agents})

    def list_agents(self) -> None:
        """列出所有远程可以控制的智能体"""
        scope, config_file = self._resolve_scope_and_config_path()
        agents = self._load_agents(config_file)
        if not agents:
            Console.warn(f"未检测到{scope} A2A 智能体。请先使用 `gaoagent agent add` 添加。")
            return
        
        Console.info(f"{scope} A2A 智能体列表：")
        for idx, (name, body) in enumerate(sorted(agents.items()), 1):
            url = body.get("url", "未知地址")
            Console.info(f"{idx}. {name} [url: {url}]")

    def add_agent(self) -> None:
        """添加远程可以控制的智能体"""
        name = Console.prompt("请输入 A2A Agent 名称", type=str).strip()
        url = Console.prompt("请输入 A2A Agent Card URL (例如 http://partner.com/a2a)", type=str).strip()
        
        Console.info(f"正在校验 A2A Agent URL: {url} ...")
        try:
            response = httpx.get(url, timeout=10.0)
            response.raise_for_status()
            card_data = response.json()
            if "name" not in card_data:
                Console.warn("URL 返回的不是有效的 Agent Card 数据格式（缺少 name 字段）。")
                return
        except Exception as e:
            Console.warn(f"校验 A2A Agent 失败: {e}")
            return

        scope, config_file = self._resolve_scope_and_config_path()
        agents = self._load_agents(config_file)
        agents[name] = {"url": url, "card": card_data}
        self._write_agents(config_file, agents)
        Console.info(f"已添加 A2A Agent：{name} -> {url} (保存至 {scope} 配置)")

    def remove_agent(self, name: str | None = None) -> None:
        """移除远程可以控制的智能体"""
        scope, config_file = self._resolve_scope_and_config_path()
        agents = self._load_agents(config_file)
        if not agents:
            Console.info(f"未检测到{scope} A2A 配置：{config_file}")
            return
        
        if not name:
            names = list(agents.keys())
            if not names:
                Console.info("暂无可以移除的 Agent。")
                return
            name = Console.prompt("请输入要移除的 Agent 名称", type=click.Choice(names))
        
        if name in agents:
            del agents[name]
            self._write_agents(config_file, agents)
            Console.info(f"已移除 A2A Agent：{name}")
        else:
            Console.warn(f"未找到名为 {name} 的 Agent。")

    def register_agent(self, port: int = 8000) -> None:
        """创建当前智能体的远程服务 (A2A Server)"""
        if not HAS_A2A:
            Console.fatal("未安装 a2a-sdk，请先运行 `pip install a2a-sdk`")
            return

        Console.info(f"正在启动 GaoAgent A2A 服务，端口：{port}...")
        card = AgentCard(
            name="GaoAgent A2A Server",
            description="基于 A2A 协议暴露的 GaoAgent 服务",
            url=f"http://localhost:{port}/a2a",
            capabilities=AgentCapabilities(streaming=True),
            skills=[
                Skill(
                    id="gaoagent_task",
                    name="GaoAgent 通用任务",
                    description="接收自然语言任务指令并执行",
                    input_modes=["text", "file"],
                    output_modes=["text", "json"]
                )
            ]
        )

        task_store = InMemoryTaskStore()
        handler = DefaultRequestHandler(
            agent_card=card,
            task_store=task_store,
            executor=GaoTaskExecutor()
        )

        app = A2AStarletteApplication(
            agent_card=card,
            http_handler=handler
        ).build()

        uvicorn.run(app, host="0.0.0.0", port=port)

    # ==== 以下为客户端调用入口示例 (供 Agent 内部逻辑调用) ====
    async def call_remote_agent(self, agent_name: str, query: str):
        """调用配置好的远程 A2A 智能体"""
        if not HAS_A2A:
            Console.fatal("未安装 a2a-sdk，请先运行 `pip install a2a-sdk`")
            return
        
        scope, config_file = self._resolve_scope_and_config_path()
        agents = self._load_agents(config_file)
        if agent_name not in agents:
            Console.fatal(f"未找到 Agent {agent_name}")
            return
        
        agent_url = agents[agent_name]["url"]
        
        async with httpx.AsyncClient() as http_client:
            client = A2AClient(http_client=http_client, agent_card_url=agent_url)
            message = Message(
                role="user",
                parts=[Part(type="text", text=query)]
            )
            
            Console.info(f"正在创建任务到 {agent_name}...")
            task = await client.create_task(message=message)
            Console.info(f"任务已创建: {task.id}，等待结果...")
            
            async for event in client.subscribe_task(task.id):
                if event.type == "artifact_update":
                    for part in event.artifact.parts:
                        if part.type == "text":
                            Console.info(f"[实时输出] {part.text}")
                elif event.type == "task_complete":
                    Console.info("任务完成")
                    break
                elif event.type == "task_failed":
                    Console.info(f"任务失败: {event.error}")
                    break
