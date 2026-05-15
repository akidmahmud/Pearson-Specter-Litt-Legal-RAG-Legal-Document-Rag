# Pearson Specter Litt — Legal RAG System
# Combines React frontend + FastAPI backend in one container

FROM python:3.10-slim

WORKDIR /app

# ── System dependencies ────────────────────────────────────────────────────────
# curl        : Node.js setup script + healthcheck
# libgl1      : required by OpenCV (used by EasyOCR)
# libglib2.0-0: required by OpenCV
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    libgl1 \
    libglib2.0-0 \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user (HF Spaces requirement) ─────────────────────────────────────
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# ── Python dependencies ────────────────────────────────────────────────────────
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Frontend build ─────────────────────────────────────────────────────────────
COPY --chown=user frontend ./frontend

WORKDIR /app/frontend
RUN npm ci --legacy-peer-deps && \
    npm run build && \
    rm -rf node_modules

WORKDIR /app

# Copy built frontend into static/ with the flat structure app.py expects:
#   static/index.html
#   static/css/...        ← NOT static/static/css/
#   static/js/...
#
# cp -r build/. copies everything including the nested build/static/ subdir.
# The second cp moves that subdir's contents up one level, then removes it.
RUN mkdir -p static && \
    cp -r frontend/build/. static/ && \
    cp -r frontend/build/static/. static/ && \
    rm -rf static/static

# ── Backend code ───────────────────────────────────────────────────────────────
COPY --chown=user app.py .
COPY --chown=user src ./src
COPY --chown=user config ./config

# ── Runtime directories ────────────────────────────────────────────────────────
RUN mkdir -p uploads chroma_db && \
    chown -R user:user /app

USER user

# ── Environment ────────────────────────────────────────────────────────────────
# OPENCODE_API_KEY must be supplied at runtime — do NOT bake secrets into the image.
# Pass it with:  docker run -e OPENCODE_API_KEY=sk-... ...
# Or on HF Spaces: add it as a Space secret.
ENV PORT=7860 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/api/health || exit 1

CMD ["python", "app.py"]
