from __future__ import annotations

import json
from typing import Any

from gaoagent.core.runner.BaseRunner import BaseRunner, RunResult, RunnerConfig, StepResult, RunnerContext
from gaoagent.core.runner.Console import Console
from gaoagent.core.runner.ReActRunner import ReActRunner
from gaoagent.core.runner.PromptBuilder import build_system_prompt
from gaoagent.core.runner.HttpClient import OpenAICompatibleHttpClient


class PlanAndExecuteRunner(BaseRunner):
    """
    PlanAndExecute 模式的执行器。
    
    核心流程:
    1. 制定任务计划
    2. 内部使用 ReActRunner 完成计划的子节点
    3. 将结果返回给 PlanAndExecuteRunner
    4. PlanAndExecuteRunner 评估是否需要调整计划，需要则调整，不需要则继续下一个子节点
    5. 所有任务完成，判断是否彻底结束
    """
    def __init__(
        self,
        *,
        config: RunnerConfig | None = None,
        tools=None,
    ) -> None:
        cfg = config or RunnerConfig()
        super().__init__(mode="plan", runner_config=cfg)
        self._tools = tools

    def decide(self, ctx: RunnerContext) -> StepResult:
        """
        PlanAndExecute 的控制流不在典型的 decide 单步循环中，而是由 run() 全局调度。
        为符合 BaseRunner 接口契约，提供空实现。
        """
        raise NotImplementedError("PlanAndExecuteRunner 使用自定义调度循环，不使用 decide() 方法。")

    def run(self, question: str, id: str | None = None) -> RunResult:
        if not question or not str(question).strip():
            Console.fatal("问题为空，无法规划任务。")
            return RunResult(success=False, error="Invalid question")

        if not self.request_base_info:
            Console.fatal("未配置 API，无法执行 PlanAndExecute。")
            return RunResult(success=False, error="No valid API configuration")

        self._client = OpenAICompatibleHttpClient(
            base_url=self.request_base_info.baseurl,
            api_key=self.request_base_info.api_key,
        )

        from gaoagent.core.runner.Utils import load_runner_state, save_runner_state
        
        state = None
        if id:
            state = load_runner_state(id, "plan")
        
        if state and state.get("question") == question:
            plan = state.get("plan", [])
            completed_tasks = state.get("completed_tasks", [])
            Console.info(f"从会话 {id} 恢复了任务计划，剩余 {len(plan)} 个步骤，已完成 {len(completed_tasks)} 个子任务。")
        else:
            Console.info("正在为任务制定初始计划...")
            plan = self._generate_plan(question)
            if not plan:
                Console.fatal("制定计划失败或返回为空。")
                return RunResult(success=False, error="Failed to generate plan.")
    
            Console.info(f"成功制定了包含 {len(plan)} 个步骤的计划：")
            for i, step in enumerate(plan):
                Console.info(f"  步骤 {i+1}: {step}")
    
            completed_tasks: list[dict[str, Any]] = []

        while plan:
            current_task = plan[0]
            if id:
                save_runner_state(id, "plan", {
                    "question": question,
                    "plan": plan,
                    "completed_tasks": completed_tasks
                })
            plan.pop(0)

            Console.info(f"\n>>> 开始执行子任务: {current_task}")

            # 组装给 ReActRunner 的提示词
            task_prompt = (
                f"【总目标】\n{question}\n\n"
                f"【当前子任务】\n{current_task}\n\n"
            )
            if completed_tasks:
                task_prompt += "【已完成的任务及结果】\n"
                for i, t in enumerate(completed_tasks):
                    task_prompt += f"子任务 {i+1}: {t['task']}\n执行结果: {t['result']}\n\n"
            
            task_prompt += "请完成【当前子任务】的要求。你可以自由调用工具来获取信息或执行操作。"

            # 内部使用 ReActRunner 完成子节点
            react_runner = ReActRunner(config=self.runner_config, tools=self._tools)
            # 子任务不使用全局 session id 避免互相干扰，只在内存中传递状态
            react_result = react_runner.run(task_prompt, id=None) 

            task_result_text = react_result.final_result if react_result.success else f"执行失败: {react_result.error}"
            
            completed_tasks.append({
                "task": current_task,
                "success": react_result.success,
                "result": task_result_text
            })

            Console.info(f"\n>>> 子任务执行完毕。成功状态: {react_result.success}")
            
            # 所有任务完成判断
            if not plan:
                Console.info("\n>>> 当前计划内任务已全部执行完毕，正在评估总任务是否彻底完成...")
                replan_decision = self._evaluate_and_replan(question, current_task, task_result_text, plan, completed_tasks)
                
                if replan_decision.get("is_finished"):
                    final_ans = replan_decision.get("final_answer", "任务已全部完成。")
                    Console.info(f"评估结果: 任务彻底完成。总结: {final_ans}")
                    if id:
                        save_runner_state(id, "plan", {})
                    return RunResult(success=True, final_result=final_ans)
                else:
                    new_plan = replan_decision.get("new_plan", [])
                    if new_plan:
                        Console.info("评估结果: 任务尚未彻底完成，需要追加新计划。")
                        plan.extend(new_plan)
                        for i, step in enumerate(new_plan):
                            Console.info(f"  新增步骤 {i+1}: {step}")
                    else:
                        Console.info("评估结果: 任务彻底完成 (未提供新计划)。")
                        if id:
                            save_runner_state(id, "plan", {})
                        return RunResult(success=True, final_result=replan_decision.get("final_answer", "任务已完成。"))
            else:
                # 评估是否需要调整后续计划
                Console.info("\n>>> 正在评估是否需要调整后续计划...")
                replan_decision = self._evaluate_and_replan(question, current_task, task_result_text, plan, completed_tasks)
                
                if replan_decision.get("is_finished"):
                    final_ans = replan_decision.get("final_answer", "任务已提前完成。")
                    Console.info(f"评估结果: 总目标已提前完成。总结: {final_ans}")
                    if id:
                        save_runner_state(id, "plan", {})
                    return RunResult(success=True, final_result=final_ans)
                
                if replan_decision.get("need_adjust"):
                    new_plan = replan_decision.get("new_plan", [])
                    Console.info("评估结果: 发现新情况，需要调整计划。新的后续计划为:")
                    plan = new_plan
                    for i, step in enumerate(plan):
                        Console.info(f"  调整后步骤 {i+1}: {step}")
                else:
                    Console.info("评估结果: 不需要调整计划，继续按原计划执行。")

        if id:
            save_runner_state(id, "plan", {})
        return RunResult(success=True, final_result="所有计划执行完毕。")

    def _generate_plan(self, question: str) -> list[str]:
        """
        调用 LLM 制定初始任务计划
        """
        prompt = (
            "你是一个高级任务规划器。请将用户的复杂任务拆解为可顺序执行的多个子任务。\n"
            "请严格输出 JSON 格式，不要包含 Markdown 标记或其他多余文本。\n"
            "JSON 格式要求如下：\n"
            "{\n"
            '  "plan": ["子任务1描述", "子任务2描述", "子任务3描述"]\n'
            "}\n"
            f"用户的总任务是：{question}"
        )
        messages = [
            {"role": "system", "content": build_system_prompt([], mode="plan")},
            {"role": "user", "content": prompt}
        ]
        response = self._client.post_chat_completions(
            model=self.request_base_info.modules,
            messages=messages
        )
        
        if response.ok and response.json:
            try:
                content = response.json.get("choices", [{}])[0].get("message", {}).get("content", "")
                content = content.replace("```json", "").replace("```", "").strip()
                data = json.loads(content)
                return data.get("plan", [])
            except Exception as e:
                Console.warn(f"解析初始计划 JSON 失败: {e}，原始返回: {response.text}")
        return []

    def _evaluate_and_replan(
        self, 
        question: str, 
        last_task: str, 
        last_result: str, 
        remaining_plan: list[str], 
        completed_tasks: list[dict]
    ) -> dict[str, Any]:
        """
        调用 LLM 评估当前执行状态，并决定是否结束、调整计划或保持原样
        """
        completed_str = ""
        for i, t in enumerate(completed_tasks):
            completed_str += f"子任务 {i+1}: {t['task']}\n执行结果: {t['result']}\n\n"

        prompt = (
            "你是一个高级任务评估与重新规划器。\n"
            f"【总目标】\n{question}\n\n"
            f"【目前已完成的所有任务及结果】\n{completed_str}\n"
            f"【刚刚完成的子任务】\n{last_task}\n"
            f"【该子任务的结果】\n{last_result}\n\n"
            f"【目前剩余的计划】\n{json.dumps(remaining_plan, ensure_ascii=False)}\n\n"
            "请评估：\n"
            "1. 结合已完成的所有结果，总目标是否已经彻底完成？\n"
            "2. 如果没有完成，目前的剩余计划是否需要调整（例如：前面步骤失败导致需要重试，或发现了新情况需要追加步骤）？\n"
            "请严格输出 JSON 格式，不要包含 Markdown 标记或其他多余文本。\n"
            "JSON 格式要求如下：\n"
            "{\n"
            '  "is_finished": false, // 总任务是否已彻底完成\n'
            '  "final_answer": "如果 is_finished 为 true，请给出给用户的总结性回复，否则留空",\n'
            '  "need_adjust": false, // 是否需要调整剩余计划\n'
            '  "new_plan": ["调整后的剩余子任务1", "调整后的剩余子任务2"] // 如果 need_adjust 为 true，请给出完整的后续执行计划列表；否则可留空\n'
            "}\n"
        )
        messages = [
            {"role": "system", "content": build_system_prompt([], mode="plan")},
            {"role": "user", "content": prompt}
        ]
        response = self._client.post_chat_completions(
            model=self.request_base_info.modules,
            messages=messages
        )
        
        default_res = {"is_finished": False, "need_adjust": False, "new_plan": remaining_plan}
        if response.ok and response.json:
            try:
                content = response.json.get("choices", [{}])[0].get("message", {}).get("content", "")
                content = content.replace("```json", "").replace("```", "").strip()
                data = json.loads(content)
                return data
            except Exception as e:
                Console.warn(f"解析评估结果 JSON 失败: {e}，原始返回: {response.text}")
                return default_res
        return default_res
