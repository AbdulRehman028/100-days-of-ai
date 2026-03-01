# ══════════════════════════════════════════════
# Day 59 — Haiku Generator · Hugging Face Spaces
# Docker deployment (CPU, free tier)
# ══════════════════════════════════════════════

FROM python:3.10-slim

# System deps for torch CPU
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── 1. Install Python packages ──────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt gunicorn

# ── 2. Pre-download model into HF cache ─────
#    This makes local_files_only=True work at runtime
#    (your app.py already uses that flag)
RUN python -c "\
from transformers import AutoTokenizer, AutoModelForCausalLM; \
print('Downloading TinyLlama...'); \
AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0'); \
AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0'); \
print('Model cached successfully!')"

# ── 3. Copy your app files ──────────────────
COPY . .

# Ensure the haiku_collection directory exists and is writable
RUN mkdir -p /app/haiku_collection && chmod 777 /app/haiku_collection

# ── 4. Expose port 7860 (HF Spaces standard) ─
EXPOSE 7860

# ── 5. Run with gunicorn (production server) ─
#    - Bypasses the if __name__ == '__main__' block
#    - So it uses port 7860 without changing app.py
#    - timeout=300 → CPU inference can be slow, give it time
#    - workers=1  → only 1 model loaded in memory (saves RAM)
CMD ["gunicorn", "app:app", \
     "--bind", "0.0.0.0:7860", \
     "--timeout", "300", \
     "--workers", "1", \
     "--threads", "2"]
