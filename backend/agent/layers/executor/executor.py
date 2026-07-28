import json
import time

from agent.context import Context
from agent.layers.layerContract import LayerContract
from agent.layers.executor.executorContract import ExecutorContract
from agent.llm.llmProviderContract import LLMProviderContract
from tools.tool_adapter import ToolAdapter


class Executor(LayerContract, ExecutorContract):

    def __init__(
        self,
        executor_config: dict,
        llm_provider: LLMProviderContract,
        tools: list,
        logger
    ):
        self.llm = llm_provider
        self.temperature = executor_config.get("temperature", 0)

        self.tools = tools
        self.qwen_tools = ToolAdapter.to_qwen(tools)

        self.logger = logger

    # ---------------------------------------------------------
    # ExecutorContract
    # ---------------------------------------------------------

    def execute(self, plan: list) -> dict:

        start = time.time()

        observations = []

        total_input = 0
        total_output = 0

        final_result = None

        # Shared conversation for every plan step
        messages = []

        for step in plan:

            self.logger.debug(f"Executing Step {step['step']}")
            self.logger.debug(step["description"])

            messages.append({
                "role": "user",
                "content": step["description"]
            })

            tools_used = []

            while True:

                self.logger.prompt(messages)

                response = self.llm.chat(
                    messages=messages,
                    temperature=self.temperature,
                    tools=self.qwen_tools
                )

                self.logger.llm_response(response)

                total_input += response["input_tokens"]
                total_output += response["output_tokens"]

                assistant = response["message"]

                messages.append(assistant)

                tool_calls = assistant.get("tool_calls", [])

                # Finished this step
                if not tool_calls:

                    self.logger.debug("Step completed.")

                    final_result = assistant.get("content", "")

                    observations.append({
                        "step": step["step"],
                        "tool_used": tools_used,
                        "output": final_result
                    })

                    break

                # Execute every requested tool
                for call in tool_calls:

                    function = call["function"]

                    tool_name = function["name"]
                    arguments = function["arguments"]

                    self.logger.tool_call(
                        tool_name,
                        arguments
                    )

                    tool = self._get_tool(tool_name)

                    tool_result = tool.run(arguments)

                    self.logger.tool_result(tool_result)

                    tools_used.append(tool_name)

                    messages.append({
                        "role": "tool",
                        "name": tool_name,
                        "content": json.dumps(tool_result)
                    })

        metrics = {
            "input": total_input,
            "output": total_output,
            "time": round(time.time() - start, 3)
        }

        self.logger.metrics(metrics)

        return {
            "result": final_result,
            "observations": observations,
            "metrics": metrics
        }

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def _get_tool(self, tool_name: str):

        for tool in self.tools:

            if tool.get_definition()["name"] == tool_name:
                return tool

        raise ValueError(f"Unknown tool '{tool_name}'")

    # ---------------------------------------------------------
    # LayerContract
    # ---------------------------------------------------------

    def run(self, context: Context) -> None:

        self.logger.info("Executor started.")

        result = self.execute(context.plan)

        context.result = result["result"]
        context.observations = result["observations"]

        context.add_metrics(
            "executor",
            result["metrics"]
        )

        self.logger.info("Executor finished.")