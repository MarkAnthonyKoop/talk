from agent.agent import Agent

fw = Agent(cfg_overrides={
    "provider": {
        "type": "fireworks",
        "model_name": "accounts/fireworks/models/deepseek-llm-r1-32b"
    }
})
print(fw.run("Explain vector databases in one paragraph or two."))

