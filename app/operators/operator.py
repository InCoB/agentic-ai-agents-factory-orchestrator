# app/operators/operator.py
import asyncio
import logging
from typing import List, Dict, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from app.agents.agent_factory import AgentFactory

logger = logging.getLogger(__name__)

class Operator:
    """
    Orchestrates a workflow by coordinating steps defined in a workflow.
    
    Each workflow is a list of steps where each step is a dict with keys:
      - 'agent': Name of the agent to invoke.
      - 'capability': (Optional) Specific capability to execute before processing.
      - 'input': (Optional) Explicit input for the step.
    """
    def __init__(self, agent_factory: "AgentFactory"):
        self.agent_factory = agent_factory

    async def execute_step(self, step: Dict[str, Any], previous_output: Optional[str]) -> str:
        agent_name = step.get("agent")
        if not agent_name:
            raise ValueError("Step missing required field 'agent'.")
        
        agent = self.agent_factory.get_agent(agent_name)
        if not agent:
            raise ValueError(f"Agent '{agent_name}' not found.")

        input_text = step.get("input", previous_output)
        if input_text is None:
            raise ValueError("No input available for the first step.")

        logger.info(f"Operator: Executing agent '{agent_name}' with input: {input_text}")

        capability_name = step.get("capability")
        if capability_name and hasattr(agent, capability_name):
            try:
                capability_fn = getattr(agent, capability_name)
                cap_result = await capability_fn(input_text)
                logger.info(f"Operator: Agent '{agent_name}' capability '{capability_name}' returned: {cap_result}")
                input_text = cap_result
            except Exception as e:
                logger.error(f"Operator: Error in capability '{capability_name}' for agent '{agent_name}': {e}")
                raise e

        output = await agent.process(input_text)
        logger.info(f"Operator: Agent '{agent_name}' returned output: {output}")
        return output

    async def orchestrate_workflow(self, workflow: List[Dict[str, Any]]) -> str:
        previous_output = None
        for idx, step in enumerate(workflow, start=1):
            logger.info(f"Operator: Starting step {idx}/{len(workflow)}.")
            previous_output = await self.execute_step(step, previous_output)
        logger.info("Operator: Workflow completed.")
        return previous_output
