FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# --timeout must exceed the slowest /parse request: it chains ~4 sequential LLM
# round-trips, and gunicorn's 30s default kills the worker mid-request (curl error 52).
CMD ["gunicorn", "main:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:9001", "--timeout", "300", "--graceful-timeout", "30"]
