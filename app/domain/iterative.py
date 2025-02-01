# app/domain/iterative.py
import asyncio
import logging
from typing import Tuple, List
from app.agents.agent_factory import AgentFactory

logger = logging.getLogger(__name__)

async def iterative_pipeline(query: str, iterations: int, factory: AgentFactory) -> str:
    history: List[str] = []
    current_input = query

    async def run_cycle(input_text: str, cycle: int) -> Tuple[str, str]:
        try:
            data_result, creative_result, fact_result = await asyncio.gather(
                factory.get_agent("DataAgent").process(input_text),
                factory.get_agent("CreativeAgent").process(input_text),
                factory.get_agent("FactCheckerAgent").process(input_text)
            )
            combined = "\n".join([data_result, creative_result, fact_result])
            summary = await factory.get_agent("SummarizerAgent").process(combined)
            critic = await factory.get_agent("CriticAgent").process(summary)
            refiner = await factory.get_agent("RefinerAgent").process(f"{summary}\n{critic}")
            detailer = await factory.get_agent("DetailerAgent").process(refiner)
            decision = await factory.get_agent("DecisionAgent").process(detailer)
            optimizer = await factory.get_agent("OptimizerAgent").process(decision)
            validator = await factory.get_agent("ValidatorAgent").process(optimizer)

            cycle_report = "\n".join([
                f"Cycle {cycle} Results:",
                data_result, creative_result, fact_result,
                summary, critic, refiner, detailer,
                decision, optimizer, validator
            ])
            new_input = f"{validator} {refiner}"
            return cycle_report, new_input
        except Exception as e:
            logger.error(f"Error in cycle {cycle}: {e}")
            return f"Cycle {cycle} encountered an error.", input_text

    for i in range(1, iterations + 1):
        cycle_report, new_input = await run_cycle(current_input, i)
        history.append(cycle_report)
        current_input = new_input
        logger.info(f"Completed cycle {i}")

    return "\n\n".join(history)
