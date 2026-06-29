# ──────────────────────────────────────────────
# Stage 1 – builder
#   Install Python deps into a clean venv so the
#   final image stays small and reproducible.
# ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install deps into an isolated venv
COPY requirements.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip --quiet && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ──────────────────────────────────────────────
# Stage 2 – runtime
# ──────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN useradd --create-home appuser
USER appuser

WORKDIR /app

# Copy the venv from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application source
COPY --chown=appuser:appuser train.py   ./train.py
COPY --chown=appuser:appuser app/       ./app/

# Activate venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Train the model at image build time so the container is self-contained
RUN python train.py

EXPOSE 8000

# Health-check so orchestrators (Docker Compose, k8s) know the app is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
