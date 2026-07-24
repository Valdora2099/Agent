# agent/agent.py

import json
import os

from agent.context import Context
from agent.pipeline import Pipeline


class Agent:
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)

        self.ollama_config = self.config["ollama"]
        self.agent_config = self.config["agent"]
        self.cost_config = self.config["cost_tracking"]

        self.pipeline: Pipeline = None  # injected via set_pipeline()
        self.verbose = self.agent_config.get("verbose", False)

    def set_pipeline(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

    def run(self, task: str) -> str:
        if self.pipeline is None:
            raise RuntimeError("Pipeline not set. Call agent.set_pipeline(pipeline) before run().")

        context = Context(task=task)

        iteration = 0
        max_iterations = self.agent_config.get("max_iterations", 5)

        while not context.done and iteration < max_iterations:
            self.pipeline.run(context)
            iteration += 1

        if self.verbose:
            self._print_summary(context)

        return context.result

    def _load_config(self, config_path: str) -> dict:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")

    def _print_summary(self, context: Context) -> None:
        if not self.cost_config.get("enabled"):
            return

        total_input = sum(m["input"] for m in context.metrics)
        total_output = sum(m["output"] for m in context.metrics)
        total_time = sum(m["time"] for m in context.metrics)

        print("\n[SUMMARY]")
        for m in context.metrics:
            print(f"  [{m['layer'].upper()}] Input: {m['input']} | Output: {m['output']} | Time: {m['time']}s")
        print(f"  [TOTAL] Tokens: {total_input + total_output} | Time: {total_time:.3f}s")