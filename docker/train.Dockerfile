FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip git curl \
    && rm -rf /var/lib/apt/lists/*
RUN python3.11 -m pip install --upgrade pip

WORKDIR /workspace
COPY train/requirements-train.txt ./train/requirements-train.txt
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 \
    && pip install --no-cache-dir -r train/requirements-train.txt

# 只复制训练需要的目录（.dockerignore 会进一步排除 .env / models / saves 等）
COPY train/ ./train/
COPY agent/ ./agent/
COPY backend/app/ ./backend/app/
COPY data/raw/regulations/ ./data/raw/regulations/

# 非 root 用户运行
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /workspace
USER appuser

CMD ["bash"]
