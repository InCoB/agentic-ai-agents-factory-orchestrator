# app/domain/local_file.py
import os
import logging
from app.agents.agent_factory import AgentFactory

logger = logging.getLogger(__name__)

async def analyze_local_file(file_path: str) -> str:
    if not os.path.isfile(file_path):
        logger.info("File not found.")
        return "File not found."
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return "Error reading file."

    factory = AgentFactory()
    summary_result = await factory.get_agent("SummarizerAgent").process(f"Summarize the following text:\n{content}")
    refiner_result = await factory.get_agent("RefinerAgent").process(f"Refine this summary for clarity:\n{summary_result}")
    final_report = "\n".join(["Local File Analysis:", summary_result, refiner_result])
    logger.info("Local file analysis completed.")
    return final_report
