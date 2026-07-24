from tools.toolsContract import ToolContract
from typing import Dict, Any
import os


class EditFileTool(ToolContract):

    def get_definition(self) -> Dict[str, Any]:
        return {
            "name": "editFile",
            "description": "Edits an existing file by replacing its contents.",
            "parameters": {
                "path": {
                    "type": "string",
                    "description": "The path of the file to edit.",
                    "required": True,
                },
                "content": {
                    "type": "string",
                    "description": "The new content that will replace the existing contents of the file.",
                    "required": True,
                },
            },
        }

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            path = input_data["path"]
            content = input_data["content"]

            if not os.path.exists(path):
                return {
                    "result": None,
                    "success": False,
                    "error": f"File not found: {path}"
                }

            with open(path, "w") as file:
                file.write(content)

            return {
                "result": f"File updated successfully at {path}",
                "success": True,
                "error": None
            }

        except Exception as e:
            return {
                "result": None,
                "success": False,
                "error": str(e)
            }