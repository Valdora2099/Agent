from abc import ABC, abstractmethod
from typing import Any, Dict, List


class ExecutorContract(ABC):
    """
    Defines the interface for executing a plan.

    Implementations are responsible for carrying out each step,
    invoking tools when required, and returning the execution results.
    """

    @abstractmethod
    def execute(self, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executes a sequence of plan steps.

        Args:
            plan: A list of plan steps.

        Returns:
            {
                "result": Any,
                "observations": List[Dict[str, Any]],
                "metrics": {
                    "input": int,
                    "output": int,
                    "time": float
                }
            }
        """
        pass