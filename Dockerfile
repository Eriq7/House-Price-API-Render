# ===== Base image =====
FROM python:3.12-slim

# ===== Minimal OS deps (certs/timezone + libgomp for xgboost) =====
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates tzdata libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# ===== Python runtime settings =====
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ===== Workdir =====
WORKDIR /app

# ===== Install Python deps (cache-friendly) =====
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ===== Copy only inference code/model =====
# 必须确保 house_price_api/model/ 下已有你训练好的模型文件
COPY house_price_api ./house_price_api

# ===== Container internal port =====
EXPOSE 8003

# ===== Start command =====
# 入口：house_price_api/app/main.py 且变量名为 app
CMD ["uvicorn", "house_price_api.app.main:app", "--host", "0.0.0.0", "--port", "$PORT"]
