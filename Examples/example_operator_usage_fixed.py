# example_operator_usage_fixed.py
import asyncio
import argparse
from app.agents.agent_factory import AgentFactory
from app.utils.logger import setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="Run OperatorAgent with a fixed workflow.")
    parser.add_argument("--prompt", type=str, default="Find me a stock to invest right now.",
                        help="User query to process.")
    return parser.parse_args()

async def main():
    args = parse_args()
    setup_logging()
    factory = AgentFactory()
    operator_agent = factory.get_agent("OperatorAgent")
    # Force fixed workflow mode
    operator_agent.use_dynamic_workflow = False
    # Override the workflow determination with a fixed, comprehensive workflow:
    operator_agent._determine_workflow = lambda prompt: [
        {"agent": "DataAgent", "capability": "parse", "input": prompt},
        {"agent": "FinancialAgent", "description": "Fetch current stock metrics", "settings": {"data_source": "yfinance", "metrics": ["PE", "EPS", "Dividend", "Volatility"]}},
        {"agent": "CreativeAgent", "capability": "format", "description": "Generate creative insights on stock performance."},
        {"agent": "FactCheckerAgent", "description": "Verify the accuracy of the financial data."},
        {"agent": "SummarizerAgent", "description": "Summarize the gathered data into an overview."},
        {"agent": "DecisionAgent", "description": "Recommend the best stock based on risk, growth, and value."},
        {"agent": "ValidatorAgent", "description": "Evaluate the final output for clarity and correctness."}
    ]
    final_response = await operator_agent.process(args.prompt)
    print("Fixed Workflow Response:\n", final_response)

if __name__ == "__main__":
    asyncio.run(main())
