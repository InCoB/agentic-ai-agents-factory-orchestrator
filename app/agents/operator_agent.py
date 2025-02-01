# app/agents/operator_agent.py
import asyncio
import json
import logging
from typing import List, Dict, Any
from openai import AsyncOpenAI

from app.agents.base_agent import BaseAgent
from app.operators.operator import Operator
from app.config import settings

logger = logging.getLogger(__name__)

class OperatorAgent(BaseAgent):
    """
    OperatorAgent receives a user prompt, determines a workflow (dynamically via LLM or fixed),
    delegates tasks to other agents, optionally modifies their system prompts if indicated,
    and evaluates the final aggregated result.
    """
    def __init__(self, name: str, system_prompt: str, agent_factory, use_dynamic_workflow: bool = True) -> None:
        super().__init__(name, system_prompt)
        self.agent_factory = agent_factory
        self.use_dynamic_workflow = use_dynamic_workflow
        # Initialize the AsyncOpenAI client with the API key from settings.
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def process(self, input_text: str) -> str:
        logger.info(f"OperatorAgent received prompt: {input_text}")
        workflow = await self._determine_workflow(input_text)
        logger.info(f"Determined workflow: {workflow}")
        operator = Operator(self.agent_factory)
        final_output = await operator.orchestrate_workflow(workflow)
        logger.info(f"OperatorAgent collected workflow output: {final_output}")
        evaluator = self.agent_factory.get_agent("ValidatorAgent")
        evaluation_prompt = f"Evaluate the following output for clarity and correctness:\n{final_output}"
        evaluation = await evaluator.process(evaluation_prompt)
        logger.info(f"OperatorAgent received evaluation: {evaluation}")
        return f"Final Output:\n{final_output}\n\nEvaluation:\n{evaluation}"

    async def _determine_workflow(self, prompt: str) -> List[Dict[str, Any]]:
        """
        Determines the workflow steps based on the user prompt.
        If dynamic workflow generation is enabled, attempts to generate the workflow via the LLM.
        Otherwise, or on failure, uses the fixed workflow.
        Ensures that the first step always has an 'input' key.
        """
        if self.use_dynamic_workflow:
            workflow = await self._generate_workflow_with_llm(prompt)
            if workflow:
                # Ensure the first step has an 'input'; if not, add it.
                if not workflow[0].get("input"):
                    workflow[0]["input"] = prompt
                return workflow
            else:
                logger.warning("Dynamic workflow generation failed; using fixed workflow.")
        return self._fixed_workflow(prompt)

    async def _generate_workflow_with_llm(self, prompt: str) -> List[Dict[str, Any]]:
        """
        Uses an LLM (via OpenAI API) to generate a JSON-formatted workflow.
        The prompt instructs the LLM to include detailed keys:
          - 'agent': Name of the agent to invoke.
          - 'capability': (Optional) A capability to use.
          - 'intent': A concise description of what the step should accomplish.
          - 'modify_prompt': (Optional) Text to replace/augment the agent's system prompt.
          - 'input': (Optional) Custom input for the step.
          - 'settings': (Optional) Additional configuration parameters.
        Returns only a JSON array with no extra text.
        """
        workflow_prompt = (
            "You are an expert workflow generator for an autonomous AI system. Analyze the user query and generate a JSON array of workflow steps. "
            "Each step must include the following keys:\n"
            "  - 'agent': (string) Name of the agent (e.g., 'DataAgent', 'FinancialAgent', 'CreativeAgent', 'SummarizerAgent', 'DecisionAgent').\n"
            "  - 'capability': (optional string) A capability to use (e.g., 'parse', 'fetch_news').\n"
            "  - 'intent': (string) A concise statement of what the step should accomplish.\n"
            "  - 'modify_prompt': (optional string) If provided, this text should replace or augment the agent's system prompt for this step.\n"
            "  - 'input': (optional string) Custom input for the step.\n"
            "  - 'settings': (optional object) Additional parameters (e.g., API endpoints, metrics).\n"
            "Return only the JSON array with no extra text.\n\n"
            f"Query: {prompt}"
        )
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "system", "content": workflow_prompt}],
                temperature=0.2,
                max_tokens=750,
                top_p=0.8,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            workflow_json = response.choices[0].message.content.strip()
            workflow = json.loads(workflow_json)
            # Validate that the workflow is a list of steps and each step has at least the "agent" key.
            if isinstance(workflow, list) and all(isinstance(step, dict) and "agent" in step for step in workflow):
                return workflow
            else:
                logger.error("Dynamic workflow JSON is invalid.")
                return []
        except Exception as e:
            logger.error(f"Error generating dynamic workflow: {e}")
            return []

    def _fixed_workflow(self, prompt: str) -> List[Dict[str, Any]]:
        """
        Returns a fixed, predetermined workflow.
        The first step always includes the user's prompt as the input to ensure that no step is missing an initial input.
        This fixed workflow demonstrates the use of multiple agent capabilities.
        """
        return [
            {"agent": "DataAgent", "capability": "parse", "input": prompt, "intent": "Extract key market indicators from the user prompt."},
            {"agent": "FinancialAgent", "description": "Fetch current stock metrics", "settings": {"data_source": "yfinance", "metrics": ["PE", "EPS", "Dividend", "Volatility"]}, "intent": "Retrieve the latest financial data."},
            {"agent": "CreativeAgent", "capability": "format", "description": "Generate creative insights on stock performance.", "intent": "Generate innovative analysis on stock trends."},
            {"agent": "FactCheckerAgent", "description": "Verify the accuracy of the financial data.", "intent": "Ensure data accuracy and flag inconsistencies."},
            {"agent": "SummarizerAgent", "description": "Summarize the gathered data into an overview.", "intent": "Provide a concise summary of the data."},
            {"agent": "DecisionAgent", "description": "Recommend the best stock based on risk, growth, and value.", "intent": "Analyze and decide on the optimal stock for investment."},
            {"agent": "ValidatorAgent", "description": "Evaluate the final output for clarity and correctness.", "intent": "Validate and refine the aggregated output."}
        ]
