# app/agents/agent_factory_helpers.py
import asyncio
import logging
from app.agents.base_agent import BaseAgent
from app.agents.capabilities import ParsingCapability, FormattingCapability
from app.utils.helpers import async_retry
from openai import AsyncOpenAI  # Make sure your openai package is v1.x

# Instantiate the async client using the new interface.
aclient = AsyncOpenAI()

logger = logging.getLogger(__name__)

class AIAgent(BaseAgent):
    def __init__(self, name: str, system_prompt: str) -> None:
        super().__init__(name, system_prompt)
        # Attach default capabilities based on agent type.
        if name == "DataAgent":
            self.add_capability("parse", ParsingCapability())
        if name == "CreativeAgent":
            self.add_capability("format", FormattingCapability())

    @async_retry(retries=3)
    async def process(self, input_text: str) -> str:
        # Pre-process input using 'parse' capability if available.
        if hasattr(self, "parse"):
            try:
                parsed = await self.parse(input_text)
                input_text = f"Parsed input: {parsed}"
            except Exception as e:
                logger.error(f"Error during parsing in {self.name}: {e}")

        # Core processing using the new async API call for OpenAI.
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input_text}
        ]
        try:
            response = await aclient.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=200
            )
            result = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error in OpenAI call for agent {self.name}: {e}")
            raise e

        return f"[{self.name}]: {result}"
