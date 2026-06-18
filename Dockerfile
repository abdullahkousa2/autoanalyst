# Minimal image for the AutoAnalyst demo — no torch, no GPU (Groq is an API).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    AUTOANALYST_ALLOW_UPLOAD=0 \
    AUTOANALYST_SAMPLES_DIR=/app/samples \
    AUTOANALYST_MODEL=llama-3.3-70b-versatile

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY autoanalyst/ ./autoanalyst/
COPY app/ ./app/
COPY samples/ ./samples/

# run as a non-root user (HF Spaces convention: uid 1000)
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

EXPOSE 8000
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
