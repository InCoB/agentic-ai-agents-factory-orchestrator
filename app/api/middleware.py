# app/api/middleware.py
import time
import uuid
import json
import logging
from fastapi import Request

logger = logging.getLogger("uvicorn.access")

async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start_time = time.time()

    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000

    # Convert log details into a JSON string to avoid formatting issues.
    log_details = json.dumps({
        "request_id": request_id,
        "method": request.method,
        "url": str(request.url),
        "status_code": response.status_code,
        "process_time_ms": f"{process_time:.2f}"
    })

    logger.info(log_details)
    return response
