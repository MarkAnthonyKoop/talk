#!/usr/bin/env python3

"""
Minimal CLI chat wrapper for any backend.
    python3 talk.py -m openai -p "Hello!"
"""

import argparse
import sys

from agent.agent import Agent
from agent.settings import Settings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default=Settings.current().provider.type)
    parser.add_argument("-p", "--prompt", default=None)
    args = parser.parse_args()

    agent = Agent(overrides={"provider": {"type": args.model}})
    prompt = args.prompt or input("You: ")
    reply = agent.run(prompt)
    print("Assistant:", reply)

if __name__ == "__main__":
    sys.exit(main())
