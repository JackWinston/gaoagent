import asyncio
from a2a.server.tasks import BaseTaskExecutor
from a2a.types import Task, Artifact, Part
from gaoagent.core.runner.console import Console
from gaoagent.core.task_runner import TaskRunner
import traceback

class GaoTaskExecutor(BaseTaskExecutor):
    """
    真正的 A2A 任务执行器。
    
    作用：
    - 接收来自 A2A 协议的任务请求
    - 提取文本内容作为 question
    - 委托给 GaoAgent 的底层核心调度器 (TaskRunner) 去执行
    - 捕获最终执行结果并通过 A2A Artifact 返回给调用方
    """
    async def execute(self, task: Task):
        # 1. 从用户消息中提取输入
        input_text = ""
        for part in task.message.parts:
            if part.type == "text":
                input_text += part.text
        
        if not input_text.strip():
            yield Artifact(
                task_id=task.id,
                parts=[Part(type="text", text="Error: 任务输入文本为空")],
                is_final=True
            )
            return

        Console.info(f"[A2A Server] 接收到任务请求: {task.id}, 内容: {input_text}")
        
        # 2. 先响应一个收到任务的状态
        yield Artifact(
            task_id=task.id,
            parts=[Part(type="text", text="[A2A Server] 已接收到任务，正在启动 GaoAgent 调度器处理...")],
        )

        # 3. 将 A2A 请求桥接到实际的内部 TaskRunner
        # 注意: TaskRunner 内部是同步逻辑，且可能有长时间的 LLM/工具调用，因此放在线程中执行避免阻塞 asyncio
        try:
            # 这里的 TaskRunner().run() 默认直接打到控制台，需要通过封装获取返回值。
            # 为了适配现有架构，我们可以直接实例化 ReActRunner 获取结果
            from gaoagent.core.runner.react_runner import ReActRunner
            from gaoagent.core.runner.run_logger import create_run_logger, set_current_run_logger, reset_current_run_logger
            from gaoagent.core.runner.tooling import default_tool_registry
            from gaoagent.core.runner.base_runner import RunnerConfig
            
            def _run_sync_task():
                run_logger = create_run_logger()
                token = set_current_run_logger(run_logger)
                try:
                    tools = default_tool_registry()
                    cfg = RunnerConfig()
                    runner = ReActRunner(config=cfg, tools=tools)
                    return runner.run(input_text, id=task.id) # 也可以把 A2A 的 task.id 透传给内部会话管理
                finally:
                    reset_current_run_logger(token)

            # 在异步循环中利用线程池跑同步阻塞的任务调度
            result = await asyncio.to_thread(_run_sync_task)
            
            # 4. 根据执行结果组装返回 Artifact
            if result.success:
                final_text = result.final_result or "任务执行成功，但没有最终文本输出。"
                yield Artifact(
                    task_id=task.id,
                    parts=[Part(type="text", text=final_text)],
                    is_final=True
                )
            else:
                error_msg = result.error or "未知执行错误"
                yield Artifact(
                    task_id=task.id,
                    parts=[Part(type="text", text=f"执行失败: {error_msg}")],
                    is_final=True
                )
                
        except Exception as e:
            err_trace = traceback.format_exc()
            Console.fatal(f"[A2A Server] 任务执行异常: {err_trace}")
            yield Artifact(
                task_id=task.id,
                parts=[Part(type="text", text=f"内部系统异常: {str(e)}")],
                is_final=True
            )
