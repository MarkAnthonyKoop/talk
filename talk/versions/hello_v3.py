from agent.agent import Agent

bot = Agent(cfg_overrides={
    "provider": {
        "type": "fireworks",
        "model_name": "accounts/fireworks/models/deepseek-v3"
    }
})
print(bot.run("Hello there!"))

