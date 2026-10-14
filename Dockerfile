# BOUNDARY — offline deterministic invoice-exception workflow
# Production image: Python 3.11, non-root runtime, Railway PORT-compatible.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first to leverage layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application source, templates, and static assets.
# app.py resolves templates/ and static/ as siblings of boundary/, so the
# three directories must sit side by side under /app.
COPY boundary ./boundary
COPY templates ./templates
COPY static ./static

# Non-root runtime user and a writable data directory for the synthetic SQLite journal.
RUN useradd --uid 10001 --create-home boundary \
    && mkdir -p /data \
    && chown -R boundary:boundary /data

# boundary.serve seeds the journal here (idempotently) and serves on PORT.
ENV BOUNDARY_DB_PATH=/data/boundary.db

USER boundary

# Railway injects PORT; 8000 is the local default.
EXPOSE 8000

CMD ["python", "-m", "boundary.serve"]
