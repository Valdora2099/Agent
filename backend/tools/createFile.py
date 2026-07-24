from tools.toolsContract import ToolContract
from typing import Dict, Any
import os


class CreateFileTool(ToolContract):

    def get_definition(self) -> Dict[str, Any]:
        return {
            "name": "createFile",
            "description": (
                "Creates a new file at the specified path. "
                "If the parent directory does not exist, it is created. "
                "Optionally writes content into the file."
            ),
            "parameters": {
                "path": {
                    "type": "string",
                    "description": "The path where the file should be created.",
                    "required": True,
                },
                "content": {
                    "type": "string",
                    "description": "The content to write into the file. Defaults to an empty string.",
                    "required": False,
                },
            },
        }

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            path = input_data["path"]
            content = input_data.get("content", "")

            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)

            with open(path, "w") as file:
                file.write(content)

            return {
                "result": f"File created successfully at {path}",
                "success": True,
                "error": None
            }

        except Exception as e:
            return {
                "result": None,
                "success": False,
                "error": str(e)
            }