FROM python:3.11-slim-bookworm

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies (no PyTorch needed — embedding & reranking use HF Inference API)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend application code
COPY . .

# NOTE: All ML inference (embeddings, reranking, guardrails, query rewriting)
# runs via HF Inference API — no local models are downloaded or loaded.

EXPOSE 8000

# Memory and threading optimizations for constrained deployment environments
ENV MALLOC_ARENA_MAX=2
ENV WEB_CONCURRENCY=1
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

# Run the FastAPI application using Uvicorn with dynamic port
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"


