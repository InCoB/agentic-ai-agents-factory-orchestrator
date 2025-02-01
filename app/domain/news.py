# app/domain/news.py
import logging
import httpx
from app.agents.agent_factory import AgentFactory
from app.config import settings

logger = logging.getLogger(__name__)

async def analyze_news(topic: str, factory: AgentFactory) -> str:
    api_key = settings.news_api_key
    url = "https://newsapi.org/v2/everything"
    params = {"q": topic, "apiKey": api_key, "pageSize": 5, "language": "en"}
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params, timeout=10.0)
            response.raise_for_status()
            articles = response.json().get("articles", [])
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return "Error fetching news."

    if not articles:
        return "No news articles found."

    news_str = "\n".join([f"{a['title']} - {a['source']['name']}" for a in articles])
    summary_result = await factory.get_agent("SummarizerAgent").process(f"Summarize these news headlines:\n{news_str}")
    decision_result = await factory.get_agent("DecisionAgent").process("Based on the news summary, what is the general sentiment?")
    final_report = "\n".join(["News Headlines:", news_str, summary_result, decision_result])
    logger.info("News analysis completed.")
    return final_report
