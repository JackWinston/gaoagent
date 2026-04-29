from __future__ import annotations

import json
from typing import Any

from gaoagent.core.runner.BaseRunner import BaseRunner, RunResult, RunnerConfig, StepResult, RunnerContext
from gaoagent.core.runner.Console import Console
from gaoagent.core.runner.HttpClient import OpenAICompatibleHttpClient
from gaoagent.core.runner.PromptBuilder import build_reflection_evaluation_prompt


class ReflectionRunner(BaseRunner):
    """
    Reflection 模式的执行器。
    
    核心流程:
    1. 接受一个 target_runner（其他 BaseRunner 实例）。
    2. 运行 target_runner 得到初始结果。
    3. 调用 LLM 评估目标是否达成（自我反思）。
    4. 如果达成，返回结果。
    5. 如果未达成，将反思的反馈意见附加到 question 中，让 target_runner 重新执行，直到达到最大反思次数。
    """
    def __init__(
        self,
        *,
        target_runner: BaseRunner,
        config: RunnerConfig | None = None,
        max_reflections: int = 3,
    ) -> None:
        cfg = config or RunnerConfig()
        super().__init__(mode="retry", runner_config=cfg)
        self.target_runner = target_runner
        self.max_reflections = max_reflections
        self._client: OpenAICompatibleHttpClient | None = None

    def decide(self, ctx: RunnerContext) -> StepResult:
        """
        ReflectionRunner 使用自定义调度循环，不使用典型的 decide 单步循环。
        为符合 BaseRunner 接口契约，提供空实现。
        """
        raise NotImplementedError("ReflectionRunner 使用自定义调度循环，不使用 decide() 方法。")

    def reflect(self, question: str, result: RunResult, id: str | None = None) -> RunResult:
        """
        核心反思流程：接受原始问题和初次执行结果，评估是否完成。
        如果未完成，则基于反思建议让 target_runner 重新执行。
        """
        current_question = question
        last_result = result

        for attempt in range(self.max_reflections):
            Console.info(f"\n>>> [Reflection] 正在评估任务是否已彻底完成 (反思轮次 {attempt + 1})...")
            evaluation = self._evaluate_result(question, last_result)
            
            is_finished = evaluation.get("is_finished", False)
            feedback = evaluation.get("feedback", "")
            
            if is_finished:
                Console.info(f"[Reflection] 评估结果: 任务已彻底完成。总结: {feedback}")
                if last_result.success and feedback:
                    last_result.final_result = f"{last_result.final_result}\n\n[最终评估结论]: {feedback}"
                return last_result
            else:
                Console.warn(f"[Reflection] 评估结果: 任务尚未彻底完成。反馈意见: {feedback}")
                if attempt < self.max_reflections - 1:
                    # 准备下一次重试的 question，将原始目标和反馈合并
                    current_question = (
                        f"【原始任务目标】\n{question}\n\n"
                        f"【上次执行结果】\n"
                        f"{last_result.final_result if last_result.success else last_result.error}\n\n"
                        f"【评估与反思意见】\n"
                        f"前一次尝试未能完全达成目标，或者存在错误。\n"
                        f"具体问题与改进建议如下：\n{feedback}\n\n"
                        f"请根据上述反思意见重新思考、调用工具并彻底完成任务。"
                    )
                    Console.info(f"\n>>> [Reflection] 第 {attempt + 1} 次基于反思意见的重新执行...")
                    last_result = self.target_runner.run(current_question, id=id)
                else:
                    Console.warn("[Reflection] 已达到最大反思重试次数，结束反思。")

        return last_result

    def run(self, question: str, id: str | None = None) -> RunResult:
        if not question or not str(question).strip():
            Console.fatal("问题为空，无法执行 Reflection。")
            return RunResult(success=False, error="Invalid question")

        if not self.request_base_info:
            Console.fatal("未配置 API，无法执行 Reflection。")
            return RunResult(success=False, error="No valid API configuration")

        self._client = OpenAICompatibleHttpClient(
            base_url=self.request_base_info.baseurl,
            api_key=self.request_base_info.api_key,
        )

        Console.info(f"\n>>> [Reflection] 开始执行初始任务...")
        # 1. 获取 target_runner 的初始结果
        initial_result = self.target_runner.run(question, id=id)
        
        # 2. 将 question 和 initial_result 传入 reflect 进行评估和可能的重试
        return self.reflect(question, initial_result, id=id)

    def _evaluate_result(self, original_question: str, result: RunResult) -> dict[str, Any]:
        """
        调用 LLM 评估当前执行状态，并决定是否完成。
        """
        result_text = result.final_result if result.success else f"执行失败: {result.error}"
        
        prompt = build_reflection_evaluation_prompt(original_question, result_text)
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # 确保 _client 不为空
        if not self._client:
            return {"is_finished": True, "feedback": "评估客户端未初始化，默认通过。"}

        response = self._client.post_chat_completions(
            model=self.request_base_info.modules,
            messages=messages
        )
        
        if response.ok and response.json:
            try:
                content = response.json.get("choices", [{}])[0].get("message", {}).get("content", "")
                content = content.replace("```json", "").replace("```", "").strip()
                data = json.loads(content)
                return data
            except Exception as e:
                Console.warn(f"解析评估 JSON 失败: {e}，原始返回: {response.text}")
        
        # 默认回退：如果 API 调用失败或解析失败，假设当前已完成以防止死循环
        Console.warn("评估失败，采取回退策略默认任务完成。")
        return {"is_finished": True, "feedback": "无法正常评估结果，已采取默认通过策略。"}
