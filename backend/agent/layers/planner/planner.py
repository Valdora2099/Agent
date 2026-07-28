# agent/layers/planner/planner.py

import json
import time

from agent.layers.layerContract import LayerContract
from agent.layers.planner.plannerContract import PlannerContract
from agent.llm.llmProviderContract import LLMProviderContract
from agent.context import Context


class Planner(LayerContract, PlannerContract):

    def __init__(self, planner_config: dict, llm_provider: LLMProviderContract,logger):
        self.llm = llm_provider
        self.temperature = planner_config.get("temperature")
        self.logger = logger

    # ---------------------------
    # PlannerContract method
    # ---------------------------
    def plan(self, task: str) -> dict:
        """
        Pure business logic. No Context dependency.
        """
        prompt = self._build_prompt(task)

        self.logger.prompt(prompt)

        response = self.llm.generate(prompt, temperature=self.temperature)

        self.logger.llm_response(response)

        raw = response['text'].strip()

        # Strip markdown fences if the model wraps its JSON despite instructions
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.replace("json", "", 1).strip()

        try:
            parsed = json.loads(raw)
            plan_steps = parsed['plan']
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[PLANNER ERROR] Failed to parse plan: {e}")
            print(f"[PLANNER ERROR] Raw response was: {response['text']}")
            plan_steps = []

        if not plan_steps:
            print(f"[PLANNER WARNING] Model returned an empty plan for task: {task}")

        self.logger.debug(plan_steps)
        metrics = {
            "input": response["input_tokens"],
            "output": response["output_tokens"],
            "time": response["time"]
        }

        self.logger.metrics(metrics)

        return {
            "plan": plan_steps,
            "metrics": metrics
        }


    def _build_prompt(self, task: str) -> str:
        return f"""You are a planning agent. Break down the task into clear, actionable steps.

Task: {task}

Respond ONLY in valid JSON, no markdown, no explanation, in this exact format:
{{
    "plan": [
        {{"step": 1, "description": "..."}},
        {{"step": 2, "description": "..."}},
        ...
    ]
}}"""

    # ---------------------------
    # LayerContract method
    # ---------------------------
    def run(self, context: Context) -> None:
        """
        Pipeline adapter. Reads from and writes to shared context.
        """
        result = self.plan(context.task)

        context.plan = result['plan']
        context.add_metrics("planner", result['metrics'])