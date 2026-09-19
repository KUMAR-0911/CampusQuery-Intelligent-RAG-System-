FROM python:3.11-slim-bookworm

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies — all ML inference runs via cloud APIs
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend application code
COPY . .

EXPOSE 8000

# Minimize memory usage for Render free tier (512Mi)
ENV MALLOC_ARENA_MAX=2

# Run FastAPI with 2 Uvicorn workers for better request concurrency.
# 2 workers on 512MB Render Free Tier: ~30MB extra RAM, 2x throughput.
# Requests no longer queue behind slow LLM calls when multiple users are active.
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2 --timeout-keep-alive 65"
