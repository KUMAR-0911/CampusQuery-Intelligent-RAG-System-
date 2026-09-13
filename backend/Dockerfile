FROM python:3.11-slim-bookworm

WORKDIR /app

# Install system dependencies required for pgvector and psycopg2
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend application code
COPY . .

EXPOSE 8000

# Run the FastAPI application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
