# Dockerfile (Multi-Stage Build)
FROM python:3.9-slim as base
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM base as final
COPY . .
ENV OPENAI_API_KEY=your_openai_api_key_here
ENV NEWS_API_KEY=your_news_api_key_here
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:8000/api/health || exit 1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
