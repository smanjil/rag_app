FROM --platform=linux/amd64 python:3.11-slim

# ---- Environment ----
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_CACHE=/root/.cache/huggingface

WORKDIR /app

# ---- System dependencies ----
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Install Python dependencies ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy project ----
COPY . .

# ---- Expose port ----
EXPOSE 8000

# ---- Start app ----
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]