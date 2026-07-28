from agent.agent import Agent
from agent.pipeline import Pipeline

from agent.layers.planner import Planner
from agent.layers.executor import Executor
from agent.layers.evaluator import Evaluator

from tools.createFile import CreateFileTool
from tools.editFile import EditFileTool
from tools.viewFolder import ViewFolderTool

import agent.llm;

def build_tools(config: dict) -> list:
    """
    Instantiate all enabled tools from config.json.
    """

    available_tools = {
        "createFile": CreateFileTool,
        "editFile": EditFileTool,
        "viewFolder": ViewFolderTool,
    }

    tools = []

    for tool_name in config["tools"]["enabled"]:

        tool_class = available_tools.get(tool_name)

        if tool_class is None:
            raise ValueError(f"Unknown tool '{tool_name}'")

        tools.append(tool_class())

    return tools


def build_pipeline(agent: Agent, tools: list) -> Pipeline:
    """
    Create the processing pipeline.
    """

    planner = Planner(
        planner_config=agent.config["layers"]["planner"],
        llm_provider=agent.llm,
        logger=agent.logger
    )

    executor = Executor(
        executor_config=agent.config["layers"]["executor"],
        llm_provider=agent.llm,
        tools=tools,
        logger=agent.logger
    )

    evaluator = Evaluator(
        evaluator_config=agent.config["layers"]["evaluator"],
        llm_provider=agent.llm,
        logger=agent.logger
    )

    return Pipeline([
        planner,
        executor,
        evaluator
    ])


def main():

    agent = Agent(
        config_path="backend/config.json"
    )

    tools = build_tools(agent.config)

    pipeline = build_pipeline(agent, tools)

    agent.set_pipeline(pipeline)

    task = (
        "Create a file named hello.txt containing the text Hello World."
    )

    result = agent.run(task)

    print("\n========== FINAL RESULT ==========\n")
    print(result)


if __name__ == "__main__":
    main()