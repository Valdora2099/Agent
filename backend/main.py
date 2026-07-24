# main.py

from agent.agent import Agent
from agent.pipeline import Pipeline
from agent.layers.planner import Planner
from agent.layers.executor import Executor
from agent.layers.evaluator import Evaluator
from tools.createFile import CreateFileTool
from tools.editFile import EditFileTool
from tools.openFolder import OpenFolderTool

def build_tools(config: dict) -> list:
    """
    Instantiate only the tools listed as enabled in config.json
    """
    available_tools = {
        "createFile": CreateFileTool,
        "editFile": EditFileTool,
        "openFolder": OpenFolderTool,
    }

    tools = []
    for tool_name in config["tools"]["enabled"]:
        tool_class = available_tools.get(tool_name)
        if tool_class is None:
            raise ValueError(f"Unknown tool: {tool_name}")
        tools.append(tool_class())

    return tools


def build_pipeline(agent: Agent, tools: list) -> Pipeline:
    """
    Instantiate layers and wire them into a pipeline.
    """
    planner = Planner(
        planner_config=agent.config["layers"]["planner"],
        ollama_config=agent.ollama_config
    )

    executor = Executor(
        executor_config=agent.config["layers"]["executor"],
        ollama_config=agent.ollama_config,
        tools=tools
    )

    evaluator = Evaluator(
        evaluator_config=agent.config["layers"]["evaluator"],
        ollama_config=agent.ollama_config
    )

    return Pipeline(layers=[planner, executor, evaluator])


if __name__ == "__main__":
    # 1. Load agent + config
    agent = Agent(config_path="backend\\config.json")

    # 2. Build tools from config
    tools = build_tools(agent.config)

    # 3. Build pipeline from config + tools
    pipeline = build_pipeline(agent, tools)
    agent.set_pipeline(pipeline)

    # 4. Run
    task = "use the editFile tool to edit the file 'test.txt' and replace its contents with 'Hello, World!'"
    result = agent.run(task)

    print("\n[FINAL RESULT]")
    print(result)