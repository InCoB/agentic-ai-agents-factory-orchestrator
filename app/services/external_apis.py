# app/services/external_apis.py
import requests
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def fetch_news(api_key: str, topic: str, page_size: int = 5) -> Dict[str, Any]:
    url = "https://newsapi.org/v2/everything"
    params = {"q": topic, "apiKey": api_key, "pageSize": page_size, "language": "en"}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Error fetching news: {e}")
        return {}
