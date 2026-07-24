from tools.toolsContract import ToolContract
from typing import Dict, Any
import os


class OpenFolderTool(ToolContract):

    def get_definition(self) -> Dict[str, Any]:
        return {
            "name": "openFolder",
            "description": "Lists every file and folder inside the given directory.",
            "parameters": {
                "path": {
                    "type": "string",
                    "description": "The path of the folder whose contents should be listed.",
                    "required": True,
                },
            },
        }

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:

        path = input_data["path"]

        if not os.path.exists(path):
            return {
                "success": False,
                "message": "Folder does not exist.",
                "data": {}
            }

        if not os.path.isdir(path):
            return {
                "success": False,
                "message": "Provided path is not a folder.",
                "data": {}
            }

        items = os.listdir(path)

        return {
            "success": True,
            "message": "Folder opened successfully.",
            "data": {
                "path": path,
                "items": items
            }
        }