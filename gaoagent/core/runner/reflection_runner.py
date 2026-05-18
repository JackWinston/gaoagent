from __future__ import annotations

import json
from typing import Any

from gaoagent.core.runner.base_runner import BaseRunner, RunResult, RunnerConfig, StepResult, RunnerContext
from gaoagent.core.runner.console import Console
from gaoagent.core.runner.http_client import OpenAICompatibleHttpClient
from gaoagent.core.runner.prompt_builder import build_reflection_evaluation_prompt


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
        self._project_overview_refresh_records: list[dict[str, Any]] = []

    def _merge_target_project_overview_refresh_records(self) -> None:
        """合并目标执行器产生的项目概览刷新触发记录。"""
        from gaoagent.core.project_overview_tool import ProjectOverviewTool

        self._project_overview_refresh_records = ProjectOverviewTool.merge_refresh_records_from_runner(
            self._project_overview_refresh_records,
            self.target_runner,
        )

    def _refresh_project_overview_after_reflection(self) -> None:
        """普通 retry 场景在整体结束后刷新一次 `project.md`。"""
        if str(getattr(self.runner_config, "scene", "default") or "default") != "default":
            return
        try:
            from gaoagent.core.project_overview_tool import ProjectOverviewTool

            if not ProjectOverviewTool.should_refresh_from_records(self._project_overview_refresh_records):
                return
            ProjectOverviewTool().refresh_current_project_overview_if_exists(
                refresh_records=ProjectOverviewTool.clone_refresh_records(self._project_overview_refresh_records)
            )
        except Exception as exc:
            Console.warn(f"项目概览刷新失败：{exc}")

    def decide(self, ctx: RunnerContext) -> StepResult:
        """
        ReflectionRunner 使用自定义调度循环，不使用典型的 decide 单步循环。
        为符合 BaseRunner 接口契约，提供空实现。
        """
        raise NotImplementedError("ReflectionRunner 使用自定义调度循环，不使用 decide() 方法。")

    def reflect(self, question: str, result: RunResult, id: str | None = None, state: dict | None = None) -> RunResult:
        """
        核心反思流程：接受原始问题和初次执行结果，评估是否完成。
        如果未完成，则基于反思建议让 target_runner 重新执行。
        """
        from gaoagent.core.runner.utils import save_runner_state

        current_question = question
        last_result = result
        start_attempt = 0

        if state:
            current_question = state.get("current_question", question)
            last_result_dict = state.get("last_result", {})
            last_result = RunResult(
                success=last_result_dict.get("success", False),
                error=last_result_dict.get("error", ""),
                final_result=last_result_dict.get("final_result", "")
            )
            start_attempt = state.get("attempt", 0)

        for attempt in range(start_attempt, self.max_reflections):
            if id:
                save_runner_state(id, "reflection", {
                    "original_question": question,
                    "current_question": current_question,
                    "last_result": {
                        "success": last_result.success,
                        "error": last_result.error,
                        "final_result": last_result.final_result
                    },
                    "attempt": attempt
                })

            Console.info(f"\n>>> [Reflection] 正在评估任务是否已彻底完成 (反思轮次 {attempt + 1})...")
            evaluation = self._evaluate_result(question, last_result)
            
            is_finished = evaluation.get("is_finished", False)
            feedback = evaluation.get("feedback", "")
            
            if is_finished:
                Console.info(f"[Reflection] 评估结果: 任务已彻底完成。总结: {feedback}")
                if last_result.success and feedback:
                    last_result.final_result = f"{last_result.final_result}\n\n[最终评估结论]: {feedback}"
                if id:
                    save_runner_state(id, "reflection", {})
                self._refresh_project_overview_after_reflection()
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
                    self._merge_target_project_overview_refresh_records()
                else:
                    Console.warn("[Reflection] 已达到最大反思重试次数，结束反思。")

        if id:
            save_runner_state(id, "reflection", {})
        self._refresh_project_overview_after_reflection()
        return last_result

    def run(self, question: str, id: str | None = None, images: str | None = None) -> RunResult:
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
        
        from gaoagent.core.runner.utils import load_runner_state
        
        state = load_runner_state(id, "reflection") if id else None
        
        if state and state.get("original_question") == question:
            Console.info(f"从会话 {id} 恢复了反思状态，继续上次反思过程...")
            return self.reflect(question, RunResult(success=True), id=id, state=state)

        # 1. 获取 target_runner 的初始结果
        initial_result = self.target_runner.run(question, id=id, images=images)
        self._merge_target_project_overview_refresh_records()
        
        # 2. 将 question 和 initial_result 传入 reflect 进行评估和可能的重试
        return self.reflect(question, initial_result, id=id)

    def _evaluate_result(self, original_question: str, result: RunResult) -> dict[str, Any]:
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
