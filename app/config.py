# app/config.py
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import Dict

class Settings(BaseSettings):
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    news_api_key: str = Field("", env="NEWS_API_KEY")
    environment: str = Field("development", env="ENVIRONMENT")
    feature_flags: Dict[str, bool] = Field(default_factory=lambda: {
        "use_dynamic_workflow": True,
        "enable_caching": True,
        "enable_metrics": True
    })

    @validator("openai_api_key")
    def validate_openai_key(cls, v: str) -> str:
        if not v or v == "your_openai_api_key_here":
            raise ValueError("A valid OpenAI API key must be provided!")
        return v

    class Config:
        env_file = ".env"

settings = Settings()
