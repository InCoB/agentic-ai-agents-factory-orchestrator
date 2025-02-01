# app/agents/capabilities.py
from abc import ABC, abstractmethod

class BaseCapability(ABC):
    @abstractmethod
    async def execute(self, *args, **kwargs) -> str:
        """Execute the capability with given arguments."""
        pass

class ParsingCapability(BaseCapability):
    async def execute(self, text: str) -> str:
        parsed_text = " ".join(text.split()).lower()
        return f"Parsed text: {parsed_text}"

class FormattingCapability(BaseCapability):
    async def execute(self, text: str) -> str:
        return f"*** {text} ***"

class DataTransformationCapability(BaseCapability):
    async def execute(self, data: str) -> str:
        # Example: transform comma-separated values into a list string.
        transformed = ", ".join(data.split(","))
        return f"Transformed data: {transformed}"

class APIIntegrationCapability(BaseCapability):
    async def execute(self, endpoint: str, params: dict) -> str:
        # Dummy implementation: simulate an API call.
        return f"API response from {endpoint} with params {params}"

class ResponseFormattingCapability(BaseCapability):
    async def execute(self, response: str) -> str:
        return f"Formatted Response: {response}"
