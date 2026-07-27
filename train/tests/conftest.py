"""train 测试路径引导：项目根（agent/train 可导入）+ backend 目录（app.* 可导入）。"""
import os
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_BACKEND_ROOT = str(Path(__file__).resolve().parents[2] / "backend")
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("LLM_API_KEY", "sk-test-placeholder")
os.environ.setdefault("QDRANT_HOST", "127.0.0.1")
