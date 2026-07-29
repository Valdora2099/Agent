from agent.agent import Agent
from agent.pipeline import Pipeline

from agent.layers.planner import Planner
from agent.layers.executor import Executor
from agent.layers.evaluator import Evaluator

from agent.llm.providerFactory import ProviderFactory

from tools.webSearch import WebSearchTool
from tools.createFile import CreateFileTool
from tools.editFile import EditFileTool
from tools.viewFolder import ViewFolderTool

from dotenv import load_dotenv



def build_tools(config: dict) -> list:
    """
    Instantiate all enabled tools from config.json.
    """

    available_tools = {
        "createFile": CreateFileTool,
        "editFile": EditFileTool,
        "viewFolder": ViewFolderTool,
        "webSearch": WebSearchTool,
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

    load_dotenv()

    agent = Agent(
        config_path="backend/config.json"
    )

    agent.llm = ProviderFactory.create(
        agent.config["llm"]
    )

    tools = build_tools(agent.config)

    pipeline = build_pipeline(agent, tools)

    agent.set_pipeline(pipeline)

    task = (
        "Search the web and tell be about Yor Forger from spyxfamily."
    )

    result = agent.run(task)

    print("\n========== FINAL RESULT ==========\n")
    print(result)


if __name__ == "__main__":
    main()