"""pytest 共享夹具与路径配置。

将项目根目录加入 sys.path，使 `agent.*` 与 `app.*` 可被导入。
"""
import sys
from pathlib import Path

# 项目根目录 = backend/tests 的上两级
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# backend 目录（使 app.* 可导入）
_BACKEND_ROOT = str(Path(__file__).resolve().parents[1])
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

# 强制使用测试环境配置：避免读取真实 .env 中的 API Key
import os
os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("LLM_API_KEY", "sk-test-placeholder")
os.environ.setdefault("AMAP_API_KEY", "")
os.environ.setdefault("QDRANT_HOST", "127.0.0.1")

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    """返回项目根目录。"""
    return Path(_PROJECT_ROOT)
