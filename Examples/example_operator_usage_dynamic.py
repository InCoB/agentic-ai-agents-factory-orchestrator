# example_operator_usage_dynamic.py
import asyncio
import argparse
from app.agents.agent_factory import AgentFactory
from app.utils.logger import setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="Run OperatorAgent with dynamic workflow generation.")
    parser.add_argument("--prompt", type=str, default="design ideal start-up",
                        help="User query to process.")
    # This flag can be used to force dynamic workflow mode (default is dynamic)
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic workflow generation.")
    return parser.parse_args()

async def main():
    args = parse_args()
    setup_logging()
    factory = AgentFactory()
    operator_agent = factory.get_agent("OperatorAgent")
    # Enable dynamic workflow mode if flag provided.
    operator_agent.use_dynamic_workflow = args.dynamic or True
    final_response = await operator_agent.process(args.prompt)
    print("Dynamic Workflow Response:\n", final_response)

if __name__ == "__main__":
    asyncio.run(main())
