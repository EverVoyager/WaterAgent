"""项目审查问题的离线回归，不访问业务数据库或外部数据源。"""
import inspect
from datetime import datetime, timezone
from itertools import permutations
from unittest.mock import Mock

import httpx
import pytest
from pydantic import ValidationError

from agent.data import weather
from agent.graph import nodes
from agent.graph.synthesizer import compute_warning_level
from agent.skills import importer
from agent.tools.mock_executor import execute_tool
from app.api import sessions
from app.core.config import Settings, get_settings
from app.main import create_app


@pytest.mark.parametrize("levels,expected", [([1, 2, 4], "I"), ([1, 2], "II")])
def test_water_risk_uses_all_stations(levels, expected):
    for order in permutations(levels):
        data = {str(i): {"water_level_m": value, "warning_level_m": 2,
                         "guaranteed_level_m": 3} for i, value in enumerate(order)}
        assert compute_warning_level(data)[0] == expected


@pytest.mark.parametrize("description,expected", [
    ("暴雨", 10), ("大暴雨", 20), ("特大暴雨", 30), ("雷阵雨", 1.5),
    ("大雨转大暴雨", 20), ("晴", 0),
])
def test_rainfall_matches_specific_description(description, expected):
    assert weather._normalize_rainfall(description) == expected


@pytest.mark.parametrize("utc_hour,expected", [(15, [10, 25, 25]), (20, [10, 25, 30])])
def test_forecast_uses_beijing_dates_and_daylight(monkeypatch, utc_hour, expected):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 22, utc_hour, tzinfo=timezone.utc)

    monkeypatch.setattr(weather, "datetime", Clock)
    casts = [
        {"date": "2026-10-22", "daytemp": "20", "nighttemp": "10"},
        {"date": "2026-10-23", "daytemp": "30", "nighttemp": "25"},
    ]
    series, _, _ = weather._build_hourly_series({"temperature": "10"}, casts, 3)
    assert [item["temperature_c"] for item in series] == expected
    with pytest.raises(RuntimeError, match="缺少"):
        weather._build_hourly_series({}, [], 3)
    with pytest.raises(RuntimeError, match="缺少"):
        weather._build_hourly_series({}, [{"date": "2000-01-01"}], 3)


@pytest.mark.parametrize("new_rain,empty_series", [(0.0, False), (0.0, True), (40.0, False)])
def test_new_weather_replaces_previous_rain_before_runoff(monkeypatch, new_rain, empty_series):
    from agent.tools.real_executor import predict_runoff_real
    from agent.tools.schemas import PredictRunoffParams

    series = [] if empty_series else [{"rainfall_mm": new_rain, "time": "2026-10-23T00:00:00Z"}]

    def execute(name, args):
        if name == "get_weather":
            return {"location": "吴堡", "hours": 6,
                    "total_rainfall_mm": new_rain, "series": series}
        return predict_runoff_real(PredictRunoffParams(**args))

    monkeypatch.setattr(nodes, "_cached_execute_tool", execute)
    result = nodes.executor_node({
        "planned_calls": [
            {"name": "get_weather", "arguments": {"location": "吴堡"}},
            {"name": "predict_runoff", "arguments": {"station": "吴堡", "rainfall_mm": None}},
        ],
        "tool_results": {"get_weather": {"total_rainfall_mm": 80.0,
                                         "series": [{"rainfall_mm": 80.0}]}},
    })
    runoff = result["tool_calls"][-1]
    assert not runoff["error"]
    assert runoff["arguments"]["rainfall_mm"] == new_rain
    assert runoff["arguments"].get("rainfall_series", []) == series
    if new_rain == 0:
        assert runoff["result"]["runoff_depth_mm"] == 0


def test_missing_weather_stays_unknown_and_latest_zero_is_preserved():
    assert nodes._collect_weather_context({}) == (None, None)
    assert nodes._collect_weather_context({
        "get_weather": {"total_rainfall_mm": 20},
        "get_weather_1": {"total_rainfall_mm": 0},
    }) == (0, None)


@pytest.mark.parametrize("invalid,write_failure", [(True, False), (False, True), (False, False)])
def test_skill_overwrite_validates_then_updates_without_delete(monkeypatch, invalid, write_failure):
    old = object()
    delete = Mock()
    create = Mock()
    update = Mock(return_value=old, side_effect=RuntimeError("write failed") if write_failure else None)
    monkeypatch.setattr(importer, "get_skill", lambda name: old)
    monkeypatch.setattr(importer, "delete_skill", delete, raising=False)
    monkeypatch.setattr(importer, "create_skill", create)
    monkeypatch.setattr(importer, "update_skill", update)
    monkeypatch.setattr(importer, "invalidate_cache", lambda: None)
    description = "short" if invalid else "A sufficiently long description"
    content = f"---\nname: existing_skill\ndescription: {description}\n---\nValid instruction text here"
    if invalid or write_failure:
        with pytest.raises(ValueError if invalid else RuntimeError):
            importer.import_skill_from_md(content, "overwrite")
    else:
        result = importer.import_skill_from_md(content, "overwrite")
        assert result.skill is old
        assert result.action == "overwritten"
    assert update.call_count == (0 if invalid else 1)
    delete.assert_not_called()
    create.assert_not_called()


def test_memory_topic_dispatch_reads_real_file(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "MEMORY_DIR", str(tmp_path))
    (tmp_path / "example.md").write_text("Saved topic content", encoding="utf-8")
    result = execute_tool("read_memory_topic", {"topic": "example"})
    assert result["found"]
    assert "Saved topic content" in result["content"]


def test_test_fixture_refuses_business_database(monkeypatch):
    from backend.tests import conftest

    store = Mock()
    monkeypatch.setattr(conftest, "is_skill_store_enabled", lambda: True)
    monkeypatch.setattr(conftest, "get_settings", lambda: Mock(MYSQL_DATABASE="water_agent"))
    monkeypatch.setattr(conftest, "get_skill_store", lambda: store)
    with pytest.raises(pytest.fail.Exception, match="_test"):
        next(conftest._restore_skills_table.__wrapped__())
    store.delete_all.assert_not_called()


def test_test_fixture_propagates_database_errors(monkeypatch):
    from backend.tests import conftest

    store = Mock()
    store.delete_all.side_effect = RuntimeError("database unavailable")
    monkeypatch.setattr(conftest, "is_skill_store_enabled", lambda: True)
    monkeypatch.setattr(conftest, "get_settings", lambda: Mock(MYSQL_DATABASE="water_agent_test"))
    monkeypatch.setattr(conftest, "get_skill_store", lambda: store)
    with pytest.raises(RuntimeError, match="unavailable"):
        next(conftest._restore_skills_table.__wrapped__())
    assert store.delete_all.call_count == 1


def test_production_requires_access_key():
    with pytest.raises(ValidationError, match="API_ACCESS_KEY"):
        Settings(_env_file=None, APP_ENV="production", LLM_API_KEY="local",
                 AMAP_API_KEY="test", API_ACCESS_KEY="")


@pytest.mark.parametrize("key", [None, "wrong", "shared-secret"])
async def test_access_key_protects_sessions_skills_and_agent(monkeypatch, key):
    monkeypatch.setattr(get_settings(), "API_ACCESS_KEY", "shared-secret")
    app = create_app()
    store = Mock()
    store.list_sessions_with_messages.return_value = []
    monkeypatch.setattr(sessions, "is_session_enabled", lambda: True)
    monkeypatch.setattr(sessions, "get_session_store", lambda: store)
    headers = {"X-API-Key": key} if key else {}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/api/health")).status_code == 200
        response = await client.get("/api/sessions", headers=headers)
        assert response.status_code == (200 if key == "shared-secret" else 401)
        assert (await client.get("/api/access", headers=headers)).status_code == response.status_code
        if key != "shared-secret":
            for method, path, body in [
                ("DELETE", "/api/sessions/private", None),
                ("PUT", "/api/skills/private", {"instructions": "overwrite"}),
                ("POST", "/api/agent/query", {"query": "test"}),
            ]:
                assert (await client.request(method, path, headers=headers, json=body)).status_code == 401
            store.list_sessions_with_messages.assert_not_called()


async def test_memory_admin_key_remains_separate(monkeypatch, tmp_path):
    settings = get_settings()
    monkeypatch.setattr(settings, "API_ACCESS_KEY", "shared-secret")
    monkeypatch.setattr(settings, "MEMORY_ADMIN_API_KEY", "admin-secret")
    monkeypatch.setattr(settings, "MEMORY_FILE", str(tmp_path / "MEMORY.md"))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://test") as client:
        assert (await client.get("/api/memories/longterm", headers={"X-API-Key": "shared-secret"})).status_code == 401
        assert (await client.get("/api/memories/longterm", headers={"X-API-Key": "admin-secret"})).status_code == 200
        assert (await client.get("/api/access", headers={"X-API-Key": "admin-secret"})).status_code == 401


async def test_production_cors_allows_key_header(monkeypatch):
    monkeypatch.setattr(get_settings(), "APP_ENV", "production")
    monkeypatch.setattr(get_settings(), "API_ACCESS_KEY", "shared-secret")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://test") as client:
        response = await client.options("/api/sessions", headers={
            "Origin": "http://localhost:5173", "Access-Control-Request-Method": "PUT",
            "Access-Control-Request-Headers": "X-API-Key,Content-Type",
        })
        assert response.status_code == 200
        assert "x-api-key" in response.headers["access-control-allow-headers"].lower()


def test_session_routes_run_in_thread_pool():
    for route in sessions.router.routes:
        assert not inspect.iscoroutinefunction(route.endpoint)
