#!/usr/bin/env python3
"""
Interactive CLI for the shell‑capable agent.
"""

from agent.settings import BOLD, GREEN
from special_agents.shell_agent import ShellAgent


def main():
    agent = ShellAgent()
    print(BOLD("Shell‑Agent CLI (Ctrl‑D to exit)\n"))
    while True:
        try:
            user = input(GREEN("you> "))
        except (EOFError, KeyboardInterrupt):
            break
        if not user.strip():
            continue
        print()
        print(agent.run(user))
        print()


if __name__ == "__main__":
    main()

