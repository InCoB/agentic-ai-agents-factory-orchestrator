# app/api/endpoints.py
from fastapi import APIRouter, HTTPException, Depends
from app.domain import iterative, financial, news, local_file
from app.agents.agent_factory import AgentFactory

router = APIRouter()

def get_agent_factory() -> AgentFactory:
    return AgentFactory()  # Creates a new instance; in production, consider a singleton

@router.post("/iterative")
async def run_iterative_analysis(query: str, iterations: int = 10, factory: AgentFactory = Depends(get_agent_factory)):
    try:
        result = await iterative.iterative_pipeline(query, iterations, factory)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/financial")
async def run_financial_analysis(ticker: str, period: str = "1y", factory: AgentFactory = Depends(get_agent_factory)):
    try:
        result = await financial.analyze_financial(ticker, period, factory)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/news")
async def run_news_analysis(topic: str, factory: AgentFactory = Depends(get_agent_factory)):
    try:
        result = await news.analyze_news(topic, factory)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/local-file")
async def run_local_file_analysis(file_path: str, factory: AgentFactory = Depends(get_agent_factory)):
    try:
        result = await local_file.analyze_local_file(file_path, factory)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health():
    return {"status": "ok"}
