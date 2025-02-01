# example_operator_usage.py
import asyncio
from app.agents.agent_factory import AgentFactory
from app.utils.logger import setup_logging

setup_logging()

async def main():
    factory = AgentFactory()
    operator_agent = factory.get_agent("OperatorAgent")
    
    user_prompt = "Analyze the current market trends and provide investment advice."
    final_response = await operator_agent.process(user_prompt)
    print("Response to User:\n", final_response)

if __name__ == "__main__":
    asyncio.run(main())
