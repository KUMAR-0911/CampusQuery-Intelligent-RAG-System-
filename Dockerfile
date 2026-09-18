FROM python:3.11-slim-bookworm

WORKDIR /app

# Copy requirements from backend directory
COPY backend/requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application code into container
COPY backend/ .

EXPOSE 8000

# Memory optimization for free tier (512Mi)
ENV MALLOC_ARENA_MAX=2
ENV WEB_CONCURRENCY=1

# Run the FastAPI application
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"
