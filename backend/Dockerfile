FROM python:3.11-slim-bookworm

WORKDIR /app

# Install minimal system dependencies required for fastembed
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install CPU-only PyTorch first to prevent massive CUDA binaries from sentence-transformers
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend application code
COPY . .

EXPOSE 8000

ENV MALLOC_ARENA_MAX=2
ENV WEB_CONCURRENCY=1

# Run the FastAPI application using Uvicorn with dynamic port
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"
