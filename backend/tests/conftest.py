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

# 已知联机测试文件（需外部服务：后端 API / LLM / Qdrant / 外部 HTTP）
# 不在此列表的测试默认视为离线单元测试，无外部服务也能秒过。
_INTEGRATION_FILES = {
    "test_hydro_source.py",    # qqjjsj.com 实时水情爬虫
    "test_round2_fix.py",      # 后端 /api/agent/query 端到端
    "test_sse_stream.py",      # 后端 SSE 流式 + LLM
    "test_stage_f_tools.py",   # fetch_hydrology / fetch_weather 外部 API
}


def pytest_collection_modifyitems(config, items):
    """自动给联机测试文件加 integration 标记。

    无 RUN_INTEGRATION=1 时跳过联机测试，避免无外部服务时测试套件挂死
    （Qdrant/HTTP 重试无快速失败）。CI 与本地默认只跑离线子集。
    """
    skip_marker = pytest.mark.skip(
        reason="联机测试需 RUN_INTEGRATION=1 才运行；默认跳过避免无外部服务时挂死"
    )
    run_integration = os.environ.get("RUN_INTEGRATION") == "1"
    for item in items:
        filename = Path(item.fspath).name
        if filename in _INTEGRATION_FILES:
            item.add_marker(pytest.mark.integration)
            if not run_integration:
                item.add_marker(skip_marker)
        else:
            item.add_marker(pytest.mark.unit)


@pytest.fixture(scope="session")
def project_root() -> Path:
    """返回项目根目录。"""
    return Path(_PROJECT_ROOT)
