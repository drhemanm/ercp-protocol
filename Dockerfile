FROM python:3.10-slim

LABEL maintainer="EvoLogics AI Lab <team@evologics.ai>"
LABEL description="ERCP Protocol Reference Server - Production Ready"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download and cache ML models to reduce startup time
RUN python -c "import spacy; spacy.cli.download('en_core_web_sm')" && \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" && \
    python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('gpt2'); AutoModelForCausalLM.from_pretrained('gpt2')"

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 ercp && \
    chown -R ercp:ercp /app

USER ercp

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["uvicorn", "server.ercp_server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]
