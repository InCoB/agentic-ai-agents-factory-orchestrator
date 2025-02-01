# app/agents/base_agent.py
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, name: str, system_prompt: str) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.capabilities = {}  # Dictionary to hold capabilities

    def add_capability(self, cap_name: str, cap_instance) -> None:
        """
        Add a capability to the agent.
        cap_instance must be an instance of a subclass of BaseCapability.
        Its execute method is attached as an attribute (e.g., self.parse).
        """
        self.capabilities[cap_name] = cap_instance
        setattr(self, cap_name, cap_instance.execute)

    @abstractmethod
    async def process(self, input_text: str) -> str:
        """Process the input text and return a response."""
        pass
