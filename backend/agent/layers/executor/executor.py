# agent/layers/executor/executor.py

import time
import json
import ollama

from agent.layers.layerContract import LayerContract
from agent.layers.executor.executorContract import ExecutorContract
from agent.context import Context


class Executor(LayerContract, ExecutorContract):

    def __init__(self, executor_config: dict, ollama_config: dict, tools: list):
        self.tools = tools  # injected once, not passed per call
        self.model = ollama_config.get("model")
        self.base_url = ollama_config.get("base_url")
        self.temperature = executor_config.get("temperature", ollama_config.get("temperature"))

    # ---------------------------
    # ExecutorContract method
    # ---------------------------
    def execute(self, plan: list) -> dict:
        """
        Pure business logic. No Context dependency.
        Executes each step in the plan, calling tools when needed.
        """
        start_time = time.time()

        observations = []
        total_input_tokens = 0
        total_output_tokens = 0
        final_output = None

        for step in plan:
            description = step['description']

            # _decide_tool now runs the tool itself and returns its output too
            decision_result = self._decide_tool(description)

            tool_used = decision_result['tool']
            output = decision_result['output']

            if tool_used is None:
                # No tool matched — fall back to plain generation
                response = ollama.generate(
                    model=self.model,
                    prompt=description,
                    options={"temperature": self.temperature},
                    stream=False
                )
                output = response['response']
                total_input_tokens += response.get('prompt_eval_count', 0)
                total_output_tokens += response.get('eval_count', 0)

            observations.append({
                "step": step['step'],
                "tool_used": tool_used,
                "output": output
            })

            final_output = output  # last step's output becomes the result

        end_time = time.time()

        return {
            "result": final_output,
            "observations": observations,
            "metrics": {
                "input": total_input_tokens,
                "output": total_output_tokens,
                "time": round(end_time - start_time, 3)
            }
        }

    def _decide_tool(self, step: str) -> dict:
        """
        Ask Ollama which tool (if any) should be used for this step.
        If a valid tool is chosen, run it and return its output too.

        Returns:
            dict:
                - tool: str or None - the tool name used, None if no tool
                - output: any - tool's output if a tool ran, else None
        """
        valid_tool_names = [t.get_definition()['name'] for t in self.tools]

        tools_text = ""
        for tool in self.tools:
            d = tool.get_definition()
            tools_text += f"\n- \"{d['name']}\": {d['description']}\n  parameters: {d['parameters']}"

        prompt = f"""You must choose a tool from this exact list: {valid_tool_names}
        Do NOT invent or rename tools. Copy the tool name exactly as written.

Available tools:
{tools_text}

Step to execute: {step}

Respond ONLY in valid JSON, no markdown, no explanation:
{{"tool": "<exact tool name from the list, or null>", "tool_input": {{...all required parameters filled in...}}}}

Example:
{{"tool": "createFile", "tool_input": {{"path": "downloads/hello.py", "content": "print('Hello, World!')"}}}}
"""

        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={"temperature": 0},
            stream=False
        )

        raw = response['response'].strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.replace("json", "", 1).strip()

        try:
            decision = json.loads(raw)
        except json.JSONDecodeError:
            decision = {"tool": None, "tool_input": None}

        tool_name = decision.get("tool")
        tool_input = decision.get("tool_input")

        # Validate tool name against actual registered tools
        if tool_name not in valid_tool_names:
            if tool_name is not None:
                print(f"[WARN] Model hallucinated unknown tool: {tool_name}")
            return {"tool": None, "output": None}

        # Run the tool and capture its output
        tool = self._get_tool(tool_name)
        tool_result = tool.run(tool_input)

        if not tool_result.get("success", True):
            print(f"[WARN] Tool '{tool_name}' failed: {tool_result.get('error')}")

        return {
            "tool": tool_name,
            "output": tool_result.get("result")
        }

    def _get_tool(self, tool_name: str):
        for tool in self.tools:
            if tool.get_definition()['name'] == tool_name:
                return tool
        raise ValueError(f"Tool '{tool_name}' not found in registered tools")

    # ---------------------------
    # LayerContract method
    # ---------------------------
    def run(self, context: Context) -> None:
        """
        Pipeline adapter. Reads from and writes to shared context.
        """
        result = self.execute(context.plan)

        context.observations = result['observations']
        context.result = result['result']
        context.add_metrics("executor", result['metrics'])