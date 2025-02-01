# main.py
import uvicorn
from fastapi import FastAPI
from app.api import endpoints, middleware
from app.config import settings
from app.utils.logger import setup_logging

# Set up basic logging configuration.
setup_logging()

app = FastAPI(title="Scalable Python AI App")

# Add middleware for structured logging.
app.middleware("http")(middleware.log_requests)

# Include API routes.
app.include_router(endpoints.router, prefix="/api")

# Custom log configuration to avoid the unpacking error.
log_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        },
    },
    "handlers": {
        "default": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "uvicorn": {"handlers": ["default"], "level": "INFO"},
        "uvicorn.error": {"handlers": ["default"], "level": "INFO"},
        "uvicorn.access": {"handlers": ["default"], "level": "INFO", "propagate": False},
    },
}

if __name__ == "__main__":
    import openai
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=log_config)
