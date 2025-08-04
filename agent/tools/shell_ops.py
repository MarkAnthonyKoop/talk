# agent/tools/shell_ops.py
"""Shell operation tools in OpenAI function format."""

SHELL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_shell_command",
            "description": "Execute a shell command and return its output",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute"
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Working directory for the command",
                        "default": "."
                    }
                },
                "required": ["command"]
            }
        }
    }
]