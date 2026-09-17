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

# Pre-download ML models during the build phase to prevent RAM spikes on startup
ENV HF_HOME=/model_cache/huggingface
ENV FASTEMBED_CACHE_PATH=/model_cache/fastembed
RUN python -c "import os; from fastembed import TextEmbedding; TextEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2', cache_dir=os.getenv('FASTEMBED_CACHE_PATH'))"

# Since we are using the remote Inference API now, we MUST allow internet access!
# (Removed HF_HUB_OFFLINE to allow the API to connect)

EXPOSE 8000

ENV MALLOC_ARENA_MAX=2
ENV WEB_CONCURRENCY=1
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV RERANKER_MODEL=""
# Run the FastAPI application using Uvicorn with dynamic port
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"
