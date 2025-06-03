from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Dict, List, Optional

from agent.messages import Message
from . import LLMBackend, LLMBackendError
from .openai_backend import OpenAIBackend  # reuse client initialisation

# —— Colour helpers (duplicated to avoid importing settings) ————————————
_ISATTY = sys.stdout.isatty()
_style = lambda txt, code: f"\033[{code}m{txt}\033[0m" if _ISATTY else txt
BLUE = lambda s: _style(s, "34")
YELLOW = lambda s: _style(s, "33")
BOLD = lambda s: _style(s, "1")

class ShellBackend(LLMBackend):
    TOOL_SCHEMA = [
        {
            "type": "function",
            "function": {
                "name": "run_shell_command",
                "description": "Execute a shell command and return its output.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]

    def __init__(self, cfg: Dict[str, str]):
        # Delegate all chat calls to OpenAIBackend
        self._delegate = OpenAIBackend(cfg)
        super().__init__(cfg)
        # Config knobs
        self.approve = bool(
            os.getenv("SHELL_APPROVE_COMMANDS", "1") if "approve_shell_commands" not in cfg else cfg["approve_shell_commands"]
        )
        self.timeout = int(
            os.getenv("SHELL_TIMEOUT", "60") if "command_timeout" not in cfg else cfg["command_timeout"]
        )

    # ------------------------------------------------------------------ #

    def _run_shell(self, cmd: str) -> str:
        try:
            cp = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=self.timeout
            )
            out = cp.stdout or ""
            err = cp.stderr or ""
            status = f"[exit {cp.returncode}]"
            return "\n".join(filter(None, [out, err, status]))
        except subprocess.TimeoutExpired:
            return f"[timeout > {self.timeout}s]"

    def _ask(self, cmd: str) -> bool:
        if not self.approve or not sys.stdout.isatty():
            return True

        print(YELLOW(f"\n⚠️  LLM suggests:\n{cmd}\n"))
        ans = input(BOLD("Run it? [y/N]: ")).lower().strip()
        return ans in ("y", "yes")

    # ------------------------------------------------------------------ #

    def complete(self, messages: List[Message]) -> Message:
        msg_dicts = [m.to_dict() for m in messages]
        while True:
            resp = self._delegate.client.chat.completions.create(
                model=self._delegate.model_name,
                messages=msg_dicts,
                tools=self.TOOL_SCHEMA,
            )

            msg = resp.choices[0].message
            if getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    fn = tc.function
                    if fn.name != "run_shell_command":
                        continue
                    args = json.loads(fn.arguments or "{}")
                    cmd = args.get("command", "")
                    output = "[denied]"
                    if cmd and self._ask(cmd):
                        print(BLUE(f"[exec] {cmd}"))
                        output = self._run_shell(cmd)
                        print(BLUE(output))

                    tool_result = Message(
                        role="tool",
                        content=output,
                        tool_call_id=tc.id,
                    )
                    msg_dicts.append(msg.to_dict())
                    msg_dicts.append(tool_result.to_dict())
                    break
                else:
                    break
            else:
                return Message.from_provider(msg.to_dict())
