# tests/test_agents.py  
import asyncio
import time
import pytest
from app.agents.agent_factory import AgentFactory
from app.services.cache import simple_cache

@pytest.mark.asyncio
async def test_operator_agent_workflow():
    factory = AgentFactory()
    operator_agent = factory.get_agent("OperatorAgent")
    prompt = "Test prompt for operator agent."
    response = await operator_agent.process(prompt)
    assert "Final Output:" in response
    assert "Evaluation:" in response

@pytest.mark.asyncio
async def test_caching():
    @simple_cache
    async def get_data(x: int) -> int:
        await asyncio.sleep(0.1)
        return x * 2

    t0 = time.time()
    result1 = await get_data(10)
    result2 = await get_data(10)
    t1 = time.time()
    assert result1 == 20 and result2 == 20
    # Second call should be fast.
    assert (t1 - t0) < 0.3
