# =========================
# STAGE 1 — Builder
# =========================
FROM python:3.10-slim AS builder

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Create virtual env & install deps
RUN uv venv && \
    . .venv/bin/activate && \
    uv sync --frozen

# =========================
# STAGE 2 — Runtime
# =========================
FROM python:3.10-slim

WORKDIR /app

# Copy venv from builder
COPY --from=builder /app/.venv /app/.venv

# Activate venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy project files
COPY src/ src/
COPY app/ app/
COPY model.pkl /home/ayush/ishu/MLE-TRAINING/models/model.pkl
COPY preprocessor.pkl /home/ayush/ishu/MLE-TRAINING/models/preprocessor.pkl

COPY pyproject.toml .

# ensure models directory exists
# copy preprocessor to root path (code expects it here)
RUN cp /home/ayush/ishu/MLE-TRAINING/models/preprocessor.pkl /home/ayush/ishu/MLE-TRAINING/preprocessor.pkl

# Expose ports
EXPOSE 8000 8501

# Default command
CMD ["bash", "-c", "uvicorn app.api:app --host 0.0.0.0 --port 8000 & streamlit run app/frontend.py --server.port 8501 --server.address 0.0.0.0"]
