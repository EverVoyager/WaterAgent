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

import pytest

# 联机测试文件（需外部服务 / HF Hub 网络下载）
_INTEGRATION_FILES = {
    "test_sft_dataset.py",  # fixture AutoTokenizer.from_pretrained 从 HF Hub 下载，无网络挂死
}

# 慢测试文件（import 重型依赖 torch/peft/transformers，耗时 >20s）
_SLOW_FILES = {
    "test_merge.py",  # import torch + peft + transformers，import 即慢
}

# collect_ignore：在 collection 阶段直接跳过这些文件（连 import 都不做），
# 避免重型 ML 包 import 拖慢/挂死测试套件。开启对应开关时才收集。
collect_ignore = []
if os.environ.get("RUN_INTEGRATION") != "1":
    collect_ignore.extend(_INTEGRATION_FILES)
if os.environ.get("RUN_SLOW") != "1":
    collect_ignore.extend(_SLOW_FILES)


def pytest_collection_modifyitems(config, items):
    """给已收集的测试项加标记（unit/integration/slow），便于 -m 过滤。"""
    for item in items:
        filename = Path(item.fspath).name
        if filename in _INTEGRATION_FILES:
            item.add_marker(pytest.mark.integration)
        elif filename in _SLOW_FILES:
            item.add_marker(pytest.mark.slow)
        else:
            item.add_marker(pytest.mark.unit)
