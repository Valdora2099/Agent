# agent/layers/planner/planner.py

import time
import ollama
import json

from agent.layers.layerContract import LayerContract
from agent.layers.planner.plannerContract import PlannerContract
from agent.context import Context


class Planner(LayerContract, PlannerContract):

    def __init__(self, planner_config: dict, ollama_config: dict):
        self.model = ollama_config.get("model")
        self.base_url = ollama_config.get("base_url")
        self.temperature = planner_config.get("temperature", ollama_config.get("temperature"))

    # ---------------------------
    # PlannerContract method
    # ---------------------------
    def plan(self, task: str) -> dict:
        start_time = time.time()
        prompt = self._build_prompt(task)

        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={"temperature": self.temperature},
            stream=False
        )

        end_time = time.time()

        try:
            parsed = json.loads(response['response'])
            plan_steps = parsed['plan']
        except (json.JSONDecodeError, KeyError):
            plan_steps = []  # or raise, or retry

        return {
            "plan": plan_steps,   # now a list of step dicts, not a raw string
            "metrics": {
                "input": response.get('prompt_eval_count', 0),
                "output": response.get('eval_count', 0),
                "time": round(end_time - start_time, 3)
            }
        }

    def _build_prompt(self, task: str) -> str:
        return f"""You are a planning agent. Break down the task into clear, actionable steps.

    Task: {task}

    Respond ONLY in valid JSON, no other text, in this exact format:
    {{
        "plan": [
            {{"step": 1, "description": "..."}},
            {{"step": 2, "description": "..."}}
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