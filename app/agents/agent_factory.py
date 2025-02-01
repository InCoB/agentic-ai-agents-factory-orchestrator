# app/agents/agent_factory.py
import logging
from typing import Dict
from app.agents.base_agent import BaseAgent
from app.agents.prompts import PROMPTS  # Extended PROMPTS now include additional agents.
from app.agents.operator_agent import OperatorAgent
from app.agents.agent_factory_helpers import AIAgent
from app.utils.helpers import async_retry

logger = logging.getLogger(__name__)

class AgentFactory:
    def __init__(self) -> None:
        self.agents: Dict[str, BaseAgent] = {}
        self._create_agents()

    def _create_agents(self) -> None:
        # Create standard agents using AIAgent for each entry in the PROMPTS dictionary.
        for name, prompt in PROMPTS.items():
            self.agents[name] = AIAgent(name, prompt)
        
        # Create the OperatorAgent with an updated system prompt that explains its role.
        operator_prompt = (
            "You are an operator agent. You receive a user prompt, determine the workflow "
            "using either dynamic generation or a fixed set of steps, modify subordinate agents' "
            "system prompts if needed, and orchestrate other agents to produce a final, polished answer."
        )
        self.agents["OperatorAgent"] = OperatorAgent("OperatorAgent", operator_prompt, self)

    def get_agent(self, name: str) -> BaseAgent:
        return self.agents.get(name)
