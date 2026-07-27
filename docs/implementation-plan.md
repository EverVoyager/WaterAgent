# 训练流水线与全栈 Docker 化 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> 关联文档：[需求分析](requirements-analysis.md) | [设计文档](design.md)

**Goal:** 交付 Hermes 训练集（3k-5k 条）→ QLoRA SFT → GRPO 规则对齐 → 三模型评估 → docker-compose 全栈部署，全程单卡 24GB 可运行。

**Architecture:** 训练侧与推理侧经 OpenAI 兼容协议解耦（vLLM 托管微调模型，backend 仅改环境变量切换）；规则权威单一来源于 `agent/graph/synthesizer.py`，训练侧只 import 不复制；训练时工具执行全部走确定性 mock（Task 3 的场景注入）。

**Tech Stack:** PyTorch、trl(SFTTrainer/GRPOTrainer)、peft、bitsandbytes、vLLM、transformers、datasets、DashScope qwen-plus（教师）、Docker / docker-compose。

**通用约定（每个任务都遵守，不再重复）：**
- TDD：先写失败测试 → 跑红 → 最小实现 → 跑绿 → 提交。
- 测试统一放 `train/tests/`，命名 `test_<模块>.py`；运行 `python -m pytest train/tests/ -v`。
- 每个任务提交一次，commit message 用 conventional commits（`feat:` / `test:` / `chore:`）。
- 新代码须过 `python -m ruff check train/`（行宽 100，遵循根 `pyproject.toml`）。
- 每个任务结束回归 `python -m pytest backend/tests/ -q`，保证现有 90 单测全绿。

---

## Phase 0 — 基础设施

### Task 1: train 包骨架 + 训练依赖 + pytest/ruff 纳入

**Files:**
- Create: `train/__init__.py`
- Create: `train/tests/__init__.py`
- Create: `train/tests/conftest.py`（关键：train 测试的 sys.path 引导）
- Create: `train/requirements-train.txt`
- Modify: `pyproject.toml`（testpaths 加入 train/tests；isort first-party 加 train）

- [ ] **Step 1: 写失败测试**

`train/tests/test_smoke.py`：

```python
"""骨架冒烟：train 包可导入，且能引用 agent/backend 现有模块。"""


def test_train_package_importable():
    import train  # noqa: F401


def test_existing_modules_importable():
    import agent.tools.schemas  # noqa: F401
    import app.core.config  # noqa: F401
```

- [ ] **Step 2: 跑红**

Run: `python -m pytest train/tests/test_smoke.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'train'`；且 testpaths 未含 train/tests 时提示 no tests ran）

- [ ] **Step 3: 最小实现**

`train/__init__.py` 与 `train/tests/__init__.py` 均为空文件。

`train/tests/conftest.py`（**必须**：`backend/tests/conftest.py` 只对 backend 测试生效，
train 测试需要自己的路径引导；同时设测试环境变量避免读真实 .env）：

```python
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
```

`pyproject.toml` 修改两处：

```toml
testpaths = ["backend/tests", "train/tests"]
```

```toml
[tool.ruff.lint.isort]
known-first-party = ["agent", "app", "backend", "train"]
```

`train/requirements-train.txt`（训练专用，与后端依赖隔离；Windows 开发机先装 CPU 版 torch 跑单测，GPU 训练容器内装 CUDA 版）：

```text
# 训练核心（版本锁定，保证可复现）
torch==2.5.1
transformers==4.49.0
trl==0.19.1
peft==0.14.0
bitsandbytes==0.45.5 ; sys_platform != "win32"
datasets==3.3.2
accelerate==1.4.0
vllm==0.7.3 ; sys_platform != "win32"   # vLLM 仅 Linux/容器
# 数据生成与评估
openai>=1.60.0
pyyaml>=6.0
```

安装（开发机）：`pip install -r train/requirements-train.txt`

- [ ] **Step 4: 跑绿**

Run: `python -m pytest train/tests/ -v; python -m pytest backend/tests/ -q`
Expected: 冒烟 PASS；后端 90 测试全 PASS

- [ ] **Step 5: 提交**

```bash
git add train/ pyproject.toml
git commit -m "chore: scaffold train package with pinned training deps"
```

---

### Task 2: Hermes 格式序列化/解析（hermes_format.py）

**Files:**
- Create: `train/data_gen/__init__.py`
- Create: `train/data_gen/hermes_format.py`
- Test: `train/tests/test_hermes_format.py`

格式契约（Qwen 聊天模板承载 Hermes 标签，与 vLLM 部署时 Qwen2.5 原生解析一致）：

- assistant 段内工具调用：`<tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>`
- tool 角色回复：`<tool_response>\n{json}\n</tool_response>`
- 一条轨迹 = `messages: [{"role": "system"|"user"|"assistant"|"tool", "content": str}, ...]`

- [ ] **Step 1: 写失败测试**

`train/tests/test_hermes_format.py`：

```python
"""Hermes 轨迹序列化/解析互逆 + 非法输入拒绝。"""
import pytest

from train.data_gen.hermes_format import (
    extract_tool_calls,
    make_tool_call_text,
    parse_final_answer,
    parse_trace,
    round_trip_ok,
)


def test_make_and_extract_tool_call_roundtrip():
    text = make_tool_call_text("get_hydrology", {"station": "吴堡", "metric": "both"})
    calls = extract_tool_calls(text)
    assert calls == [{"name": "get_hydrology", "arguments": {"station": "吴堡", "metric": "both"}}]


def test_extract_multiple_tool_calls():
    text = (
        make_tool_call_text("get_weather", {"location": "吴堡", "hours": 24})
        + make_tool_call_text("get_hydrology", {"station": "吴堡", "metric": "both"})
    )
    assert [c["name"] for c in extract_tool_calls(text)] == ["get_weather", "get_hydrology"]


def test_extract_rejects_bad_json():
    assert extract_tool_calls("<tool_call>\n{not json}\n</tool_call>") == []


def test_parse_trace_rejects_non_json_tool_response():
    msgs = [
        {"role": "user", "content": "查水情"},
        {"role": "assistant", "content": make_tool_call_text("get_hydrology", {"station": "吴堡"})},
        {"role": "tool", "content": "不是JSON"},
    ]
    assert parse_trace(msgs) is None


def test_parse_final_answer_level():
    msgs = [{"role": "assistant", "content": "……综上，发布Ⅱ级（橙色）预警……"}]
    assert parse_final_answer(msgs) == "II"


def test_round_trip_ok():
    msgs = [
        {"role": "user", "content": "吴堡水情？"},
        {"role": "assistant", "content": make_tool_call_text("get_hydrology", {"station": "吴堡"})},
        {"role": "tool", "content": '{"station": "吴堡", "flow_m3_s": 3250.0}'},
        {"role": "assistant", "content": "吴堡站流量 3250m³/s，达到Ⅱ级预警。"},
    ]
    assert round_trip_ok(msgs)
```

- [ ] **Step 2: 跑红**

Run: `python -m pytest train/tests/test_hermes_format.py -v`
Expected: FAIL（ModuleNotFoundError: train.data_gen）

- [ ] **Step 3: 最小实现**

`train/data_gen/__init__.py` 为空文件。

`train/data_gen/hermes_format.py`：

```python
"""Hermes 格式序列化/解析（训练集、过滤器、奖励函数、评估共用唯一实现）。

标签约定（Qwen 原生）：
- assistant 内容中的工具调用：<tool_call>\n{json}\n</tool_call>
- tool 消息内容：<tool_response>\n{json}\n</tool_response>（裸 JSON 也接受）
"""
import json
import re
from typing import Any, Optional

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
# 等级归一化：先匹配长词（Ⅳ>Ⅲ>Ⅱ>Ⅰ），避免子串误配
_LEVEL_MAP = [
    (re.compile(r"Ⅳ|IV|4\s*级|四\s*级"), "IV"),
    (re.compile(r"Ⅲ|III|3\s*级|三\s*级"), "III"),
    (re.compile(r"Ⅱ|II|2\s*级|二\s*级"), "II"),
    (re.compile(r"Ⅰ|(?<!I)I(?!I)|1\s*级|一\s*级"), "I"),
]


def make_tool_call_text(name: str, arguments: dict) -> str:
    """序列化单个工具调用块。"""
    return f"<tool_call>\n{json.dumps({'name': name, 'arguments': arguments}, ensure_ascii=False)}\n</tool_call>"


def make_tool_response_text(result: dict) -> str:
    """序列化 tool 消息内容。"""
    return f"<tool_response>\n{json.dumps(result, ensure_ascii=False)}\n</tool_response>"


def extract_tool_calls(assistant_content: str) -> list[dict]:
    """从 assistant 文本提取全部工具调用；任一 JSON 非法则该次返回 []。"""
    calls = []
    for m in _TOOL_CALL_RE.finditer(assistant_content):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            return []
        if not isinstance(obj, dict) or "name" not in obj or "arguments" not in obj:
            return []
        if not isinstance(obj["arguments"], dict):
            return []
        calls.append({"name": obj["name"], "arguments": obj["arguments"]})
    return calls


def parse_trace(messages: list[dict]) -> Optional[dict]:
    """解析完整轨迹。返回 {"tool_calls": [...], "tool_results": [...], "final": str}；
    任一结构非法（坏 JSON、tool 消息无前置调用）返回 None。"""
    tool_calls: list[dict] = []
    tool_results: list[dict] = []
    final = ""
    pending_calls = 0
    for msg in messages:
        role, content = msg.get("role"), msg.get("content", "")
        if role == "assistant":
            calls = extract_tool_calls(content)
            if "<tool_call>" in content and not calls:
                return None  # 有标签但全部解析失败
            if calls:
                tool_calls.extend(calls)
                pending_calls += len(calls)
            else:
                final = content  # 无调用的 assistant 段 = 最终回答
        elif role == "tool":
            if pending_calls <= 0:
                return None
            raw = content
            m = re.match(r"\s*<tool_response>\s*(\{.*\})\s*</tool_response>\s*$", raw, re.DOTALL)
            payload = m.group(1) if m else raw
            try:
                tool_results.append(json.loads(payload))
            except json.JSONDecodeError:
                return None
            pending_calls -= 1
    return {"tool_calls": tool_calls, "tool_results": tool_results, "final": final}


def parse_final_answer(messages: list[dict]) -> Optional[str]:
    """从轨迹最后一个 assistant 段提取归一化等级（I/II/III/IV），无则 None。"""
    final = next((m["content"] for m in reversed(messages) if m.get("role") == "assistant"), "")
    for pattern, level in _LEVEL_MAP:
        if pattern.search(final):
            return level
    return None


def round_trip_ok(messages: list[dict]) -> bool:
    """parse_trace 成功且最终段可提取等级（chatty 样本除外，调用方自行豁免）。"""
    return parse_trace(messages) is not None
```

- [ ] **Step 4: 跑绿**

Run: `python -m pytest train/tests/test_hermes_format.py -v`
Expected: 6 passed

- [ ] **Step 5: 提交**

```bash
git add train/data_gen/ train/tests/test_hermes_format.py
git commit -m "feat(train): add Hermes trace serializer/parser with round-trip tests"
```

---

## Phase 1 — 训练数据生成（M1）

### Task 3: mock_executor 确定性场景注入（向后兼容扩展）

**Files:**
- Modify: `agent/tools/mock_executor.py`（新增可选参数，不改默认行为）
- Test: `train/tests/test_mock_deterministic.py`

设计依据（design.md 第 2 章）：仅新增可选参数 `overrides`/`seed`，默认 `None` 走原逻辑；
现有 90 后端测试不得受影响。

- [ ] **Step 1: 写失败测试**

`train/tests/test_mock_deterministic.py`：

```python
"""确定性 mock：同 seed+overrides 数值完全一致；覆盖值生效。

注意：mock 结果含真实时间戳（fetched_at / series[].time），确定性只保证
数值字段，比较时剔除时间字段。
"""
from agent.tools.mock_executor import execute_tool


def _strip_times(obj):
    if isinstance(obj, dict):
        return {k: _strip_times(v) for k, v in obj.items()
                if not (k.endswith("_at") or k == "time")}
    if isinstance(obj, list):
        return [_strip_times(x) for x in obj]
    return obj


def test_deterministic_with_seed():
    a = execute_tool("get_hydrology", {"station": "吴堡", "metric": "both"}, seed=42)
    b = execute_tool("get_hydrology", {"station": "吴堡", "metric": "both"}, seed=42)
    assert _strip_times(a) == _strip_times(b)


def test_overrides_inject_values():
    out = execute_tool(
        "get_hydrology",
        {"station": "吴堡", "metric": "both"},
        overrides={"flow_m3_s": 5200.0, "water_level_m": 644.5},
        seed=42,
    )
    assert out["flow_m3_s"] == 5200.0
    assert out["water_level_m"] == 644.5


def test_overrides_none_keeps_existing_signature():
    # 不传新参数 = 现状行为（只断言结构，不断言具体随机值）
    out = execute_tool("get_weather", {"location": "吴堡", "hours": 6})
    assert "series" in out and len(out["series"]) == 6


def test_runoff_peak_override():
    out = execute_tool(
        "predict_runoff",
        {"station": "吴堡", "lead_time_hours": 24},
        overrides={"peak_flow_m3_s": 6100.0},
        seed=7,
    )
    assert out["peak_flow_m3_s"] == 6100.0
    assert max(s["predicted_flow_m3_s"] for s in out["series"]) <= 6100.0
```

- [ ] **Step 2: 跑红**

Run: `python -m pytest train/tests/test_mock_deterministic.py -v`
Expected: FAIL（TypeError: unexpected keyword argument 'seed'/'overrides'）

- [ ] **Step 3: 最小实现**

`agent/tools/mock_executor.py` 修改三处：

① 文件头部 import 保持不变，在各 `_mock_*` 函数签名追加 `overrides: dict | None = None`，
函数末尾 return 前合并覆盖值。以 `_mock_get_hydrology` 为例：

```python
def _mock_get_hydrology(params: GetHydrologyParams, overrides: dict | None = None) -> Dict[str, Any]:
    """模拟水文站实时水情。"""
    station_data = {
        "吴堡": {"base_level": 640.5, "base_flow": 1200},
        "龙门": {"base_level": 382.3, "base_flow": 2400},
        "府谷": {"base_level": 810.2, "base_flow": 850},
    }
    base = station_data.get(params.station, {"base_level": 500.0, "base_flow": 1000})
    result = {
        "station": params.station,
        "fetched_at": _now_iso(),
    }
    if params.metric in ("water_level", "both"):
        result["water_level_m"] = round(base["base_level"] + random.uniform(-0.5, 2.5), 2)
        result["warning_level_m"] = round(base["base_level"] + 2.0, 2)
        result["guaranteed_level_m"] = round(base["base_level"] + 3.5, 2)
    if params.metric in ("flow", "both"):
        result["flow_m3_s"] = round(base["base_flow"] * random.uniform(1.0, 2.5), 0)
        result["warning_flow_m3_s"] = round(base["base_flow"] * 2.0, 0)
    if overrides:
        result.update(overrides)
    return result
```

`_mock_predict_runoff` 特殊：overrides 需在生成 series **之前**生效（peak 驱动曲线）：

```python
def _mock_predict_runoff(params: PredictRunoffParams, overrides: dict | None = None) -> Dict[str, Any]:
    """模拟径流流量预测 API 返回。"""
    ov = overrides or {}
    peak_flow = round(float(ov.get("peak_flow_m3_s", random.uniform(3000, 8000))), 0)
    series = []
    base_time = datetime.now(timezone.utc)
    for i in range(0, params.lead_time_hours + 1, 3):
        ratio = 1.0 - abs(i - params.lead_time_hours / 2) / (params.lead_time_hours / 2)
        flow = round(peak_flow * max(ratio, 0.3), 0)
        series.append({
            "time": (base_time + timedelta(hours=i)).isoformat(),
            "predicted_flow_m3_s": flow,
        })
    result = {
        "station": params.station,
        "lead_time_hours": params.lead_time_hours,
        "peak_flow_m3_s": peak_flow,
        "peak_time": series[len(series) // 2]["time"] if series else None,
        "series": series,
        "model": "mock-lstm-v0.1",
        "predicted_at": _now_iso(),
    }
    for k, v in ov.items():
        if k != "peak_flow_m3_s":  # peak 已用于曲线，其余键直接覆盖
            result[k] = v
    return result
```

其余 `_mock_get_weather` / `_mock_query_gis_terrain` / `_mock_search_regulation` /
`_mock_generate_plan` 同样追加 `overrides: dict | None = None` 并在 return 前
`if overrides: result.update(overrides)`。

② `execute_tool` 新签名：

```python
def execute_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    overrides: Dict[str, Any] | None = None,
    seed: int | None = None,
) -> Dict[str, Any]:
```

函数体两处改动：
- 开头：`if seed is not None: random.seed(seed)`（模块级 random 已 import）
- mock 分支：`mock_result = impl(params, overrides=overrides)`
- **真实实现分支不动**（overrides/seed 只作用于 mock 回放；训练侧调用前确保
  `real_executor` 不可用或在训练侧直接调用 mock 路径，见 Task 6 说明）

- [ ] **Step 4: 跑绿 + 回归**

Run: `python -m pytest train/tests/test_mock_deterministic.py -v; python -m pytest backend/tests/ -q`
Expected: 新测试 4 passed；后端 90 测试全 PASS

- [ ] **Step 5: 提交**

```bash
git add agent/tools/mock_executor.py train/tests/test_mock_deterministic.py
git commit -m "feat(train): deterministic mock executor via optional overrides/seed"
```

---

### Task 4: 场景生成器（scenario.py）

**Files:**
- Create: `train/data_gen/scenario.py`
- Test: `train/tests/test_scenario.py`

- [ ] **Step 1: 写失败测试**

`train/tests/test_scenario.py`：

```python
"""场景生成器：确定性、配额均衡、等级真值正确、种子区间隔离。"""
from train.data_gen.scenario import Scenario, generate_scenarios


def test_deterministic_same_seed():
    a = generate_scenarios(n=20, seed=1)
    b = generate_scenarios(n=20, seed=1)
    assert [s.scenario_id for s in a] == [s.scenario_id for s in b]
    assert [s.expected_level for s in a] == [s.expected_level for s in b]


def test_level_quota_balanced():
    scenarios = generate_scenarios(n=400, seed=1)
    biz = [s for s in scenarios if s.query_type != "chatty"]
    for level in ("I", "II", "III", "IV"):
        ratio = sum(1 for s in biz if s.expected_level == level) / len(biz)
        assert 0.20 <= ratio <= 0.30, f"{level} 占比 {ratio:.2f} 超出 ±5% 容差"


def test_level_truth_matches_thresholds():
    scenarios = generate_scenarios(n=200, seed=2)
    for s in scenarios:
        if s.query_type == "chatty":
            continue
        flow = s.tool_overrides["get_hydrology"]["flow_m3_s"]
        if s.expected_level == "I":
            assert flow >= 5000
        elif s.expected_level == "II":
            assert 3000 <= flow < 5000
        elif s.expected_level == "III":
            assert 2000 <= flow < 3000
        elif s.expected_level == "IV":
            assert flow < 2000


def test_seed_ranges_do_not_overlap():
    train = generate_scenarios(n=50, seed=1000)
    other = generate_scenarios(n=50, seed=101000)
    assert {s.scenario_id for s in train}.isdisjoint({s.scenario_id for s in other})


def test_chatty_ratio():
    scenarios = generate_scenarios(n=200, seed=3, chatty_ratio=0.08)
    ratio = sum(1 for s in scenarios if s.query_type == "chatty") / len(scenarios)
    assert 0.05 <= ratio <= 0.11


def test_scenario_fields_complete():
    s = generate_scenarios(n=1, seed=5)[0]
    assert isinstance(s, Scenario)
    assert s.station and s.query and s.query_type
    assert "get_hydrology" in s.tool_overrides or s.query_type == "chatty"
```

- [ ] **Step 2: 跑红**

Run: `python -m pytest train/tests/test_scenario.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 最小实现**

`train/data_gen/scenario.py`：

```python
"""防汛场景生成器：组合维度生成确定性场景，携带等级真值与 mock 覆盖值。

等级真值直接由流量档位决定（与 synthesizer 阈值同源）：
  I 级 >=5000 | II 级 [3000,5000) | III 级 [2000,3000) | IV 级 <2000  m³/s
种子区间约定（保证 SFT / GRPO / 评估零重叠）：
  SFT:   seed in [0, 100_000)
  GRPO:  seed in [100_000, 200_000)
  EVAL:  seed in [200_000, 300_000)
"""
import random
from dataclasses import dataclass, field

STATIONS = ["吴堡", "龙门", "府谷"]
STATION_BASE_LEVEL = {"吴堡": 640.5, "龙门": 382.3, "府谷": 810.2}
QUERY_TYPES = ["single_tool", "multi_tool", "plan_only"]
PERSONAS = ["防汛值班员", "乡镇干部", "沿河企业负责人"]

_LEVEL_TO_FLOW_RANGE = {
    "I": (5000.0, 6500.0),
    "II": (3000.0, 4999.0),
    "III": (2000.0, 2999.0),
    "IV": (500.0, 1999.0),
}

_QUERY_TEMPLATES = {
    "single_tool": ["{station}站现在水情怎么样？", "查一下{station}水文站的实时流量和水位。"],
    "multi_tool": [
        "{station}站未来24小时有洪水风险吗？需要预警吗？",
        "我是{persona}，{station}站一带在下雨，帮我研判一下防汛形势。",
    ],
    "plan_only": ["{station}站已达{level_cn}预警，请生成{persona}的应急处置预案。"],
    "chatty": ["今天天气真好", "你会做什么？", "讲讲黄河的历史吧"],
}

_LEVEL_CN = {"I": "Ⅰ级", "II": "Ⅱ级", "III": "Ⅲ级", "IV": "Ⅳ级"}


@dataclass
class Scenario:
    scenario_id: str
    station: str
    query: str
    query_type: str  # single_tool / multi_tool / plan_only / chatty
    expected_level: str  # chatty 时为 ""
    tool_overrides: dict = field(default_factory=dict)  # 工具名 -> overrides
    reference_tools: list = field(default_factory=list)  # 期望调用的工具集合


def _make_overrides(rng: random.Random, station: str, level: str) -> dict:
    """按等级档位生成各工具 mock 覆盖值（同 rng 保证确定性）。"""
    lo, hi = _LEVEL_TO_FLOW_RANGE[level]
    flow = round(rng.uniform(lo, hi), 1)
    base_level = STATION_BASE_LEVEL[station]
    warn = round(base_level + 2.0, 2)
    guar = round(base_level + 3.5, 2)
    # 水位状态与等级对齐：I 级超保证，II 级超警戒，III/IV 正常
    if level == "I":
        water_level = round(guar + rng.uniform(0.0, 0.5), 2)
    elif level == "II":
        water_level = round(warn + rng.uniform(0.0, 0.4), 2)
    else:
        water_level = round(base_level + rng.uniform(-0.3, 0.5), 2)
    rain = {"I": 120.0, "II": 75.0, "III": 30.0, "IV": 8.0}[level]
    # peak 取 flow 的 1.0-1.1 倍但不越过本档上限 hi，防止跨档改变等级真值
    # （如 II 档 flow=4900 × 1.15 = 5635 ≥ 5000 会被规则引擎误判为 I 级）
    peak = round(min(flow * rng.uniform(1.0, 1.1), hi), 1)
    return {
        "get_weather": {
            "total_rainfall_mm": rain,
            "max_hourly_rainfall_mm": round(rain / 24, 1),
        },
        "get_hydrology": {
            "flow_m3_s": flow,
            "water_level_m": water_level,
            "warning_level_m": warn,
            "guaranteed_level_m": guar,
        },
        "predict_runoff": {"peak_flow_m3_s": peak},
    }


def generate_scenarios(n: int, seed: int, chatty_ratio: float = 0.08) -> list:
    """生成 n 条确定性场景。等级在业务场景内均匀轮换，chatty 按比例混入。"""
    rng = random.Random(seed)
    n_chatty = round(n * chatty_ratio)
    n_biz = n - n_chatty
    scenarios = []
    levels_cycle = ["I", "II", "III", "IV"]

    for i in range(n_biz):
        level = levels_cycle[i % 4]  # 轮换保证严格均衡
        station = rng.choice(STATIONS)
        persona = rng.choice(PERSONAS)
        qtype = rng.choice(QUERY_TYPES)
        template = rng.choice(_QUERY_TEMPLATES[qtype])
        query = template.format(station=station, persona=persona, level_cn=_LEVEL_CN[level])
        ref_tools = {
            "single_tool": ["get_hydrology"],
            "multi_tool": ["get_hydrology", "get_weather", "predict_runoff"],
            "plan_only": ["search_regulation", "generate_plan"],
        }[qtype]
        scenarios.append(Scenario(
            scenario_id=f"scn-{seed}-{i}",
            station=station,
            query=query,
            query_type=qtype,
            expected_level=level,
            tool_overrides=_make_overrides(rng, station, level),
            reference_tools=ref_tools,
        ))

    for j in range(n_chatty):
        query = rng.choice(_QUERY_TEMPLATES["chatty"])
        scenarios.append(Scenario(
            scenario_id=f"scn-{seed}-chatty-{j}",
            station=rng.choice(STATIONS),
            query=query,
            query_type="chatty",
            expected_level="",
            tool_overrides={},
            reference_tools=[],
        ))

    rng.shuffle(scenarios)
    return scenarios
```

- [ ] **Step 4: 跑绿**

Run: `python -m pytest train/tests/test_scenario.py -v`
Expected: 6 passed

- [ ] **Step 5: 提交**

```bash
git add train/data_gen/scenario.py train/tests/test_scenario.py
git commit -m "feat(train): scenario generator with level truth and seed isolation"
```

---

### Task 5: 三道规则过滤（filters.py）

**Files:**
- Create: `train/data_gen/filters.py`
- Test: `train/tests/test_filters.py`

- [ ] **Step 1: 写失败测试**

`train/tests/test_filters.py`：

```python
"""三道过滤：F1 参数合法 / F2 序列合法 / F3 等级一致（chatty 豁免）。"""
from train.data_gen.filters import FilterResult, filter_trace
from train.data_gen.hermes_format import make_tool_call_text, make_tool_response_text
from train.data_gen.scenario import generate_scenarios


def _scenario():
    return generate_scenarios(n=1, seed=42)[0]


def _trace_with(calls_and_results, final_text):
    msgs = [{"role": "user", "content": "q"}]
    for call, result in calls_and_results:
        msgs.append({"role": "assistant", "content": make_tool_call_text(call[0], call[1])})
        msgs.append({"role": "tool", "content": make_tool_response_text(result)})
    msgs.append({"role": "assistant", "content": final_text})
    return msgs


def _hydro_result(scn):
    return {"station": scn.station, **scn.tool_overrides["get_hydrology"]}


def test_f1_rejects_invalid_params():
    scn = _scenario()
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": make_tool_call_text("get_hydrology", {"station": "吴堡", "metric": "非法值"})},
        {"role": "tool", "content": make_tool_response_text(_hydro_result(scn))},
        {"role": "assistant", "content": "Ⅱ级预警"},
    ]
    r = filter_trace(msgs, scn)
    assert r == FilterResult.REJECT_F1


def test_f2_rejects_runoff_before_weather():
    scn = _scenario()
    msgs = _trace_with(
        [(("predict_runoff", {"station": scn.station}), {"peak_flow_m3_s": 4000.0, "series": []})],
        "Ⅱ级预警",
    )
    assert filter_trace(msgs, scn) == FilterResult.REJECT_F2


def test_f2_rejects_unknown_tool():
    scn = _scenario()
    msgs = _trace_with(
        [(("hack_tool", {}), {"x": 1})],
        "Ⅳ级",
    )
    assert filter_trace(msgs, scn) == FilterResult.REJECT_F2


def test_f3_rejects_level_mismatch():
    scn = _scenario()
    msgs = _trace_with(
        [(("get_hydrology", {"station": scn.station, "metric": "both"}), _hydro_result(scn))],
        "当前水情平稳，Ⅳ级蓝色预警。",
    )
    assert filter_trace(msgs, scn) == FilterResult.REJECT_F3


def test_accept_valid_trace():
    scn = _scenario()
    msgs = _trace_with(
        [
            (("get_weather", {"location": scn.station, "hours": 24}),
             {"location": scn.station, **scn.tool_overrides["get_weather"]}),
            (("get_hydrology", {"station": scn.station, "metric": "both"}), _hydro_result(scn)),
            (("predict_runoff", {"station": scn.station, "lead_time_hours": 24}),
             {"station": scn.station, "series": [],
              "peak_flow_m3_s": scn.tool_overrides["predict_runoff"]["peak_flow_m3_s"]}),
        ],
        f"流量 {scn.tool_overrides['get_hydrology']['flow_m3_s']}m³/s，发布{scn.expected_level}级预警。",
    )
    assert filter_trace(msgs, scn) == FilterResult.ACCEPT


def test_chatty_exemption():
    scn = generate_scenarios(n=200, seed=3, chatty_ratio=0.5)
    chatty = next(s for s in scn if s.query_type == "chatty")
    msgs = [
        {"role": "user", "content": chatty.query},
        {"role": "assistant", "content": "我是防汛预警智能体，可以帮你查水情、研判预警。"},
    ]
    assert filter_trace(msgs, chatty) == FilterResult.ACCEPT
    bad = [
        {"role": "user", "content": chatty.query},
        {"role": "assistant", "content": "发布Ⅰ级预警！"},
    ]
    assert filter_trace(bad, chatty) == FilterResult.REJECT_F3


def test_level_cn_text_normalized():
    # 中文数字等级也能归一化（Ⅱ级 vs II）
    scn = next(s for s in generate_scenarios(n=10, seed=9) if s.expected_level == "II")
    msgs = _trace_with(
        [(("get_hydrology", {"station": scn.station, "metric": "both"}), _hydro_result(scn))],
        "发布Ⅱ级（橙色）预警。",
    )
    assert filter_trace(msgs, scn) == FilterResult.ACCEPT
```

- [ ] **Step 2: 跑红**

Run: `python -m pytest train/tests/test_filters.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 最小实现**

`train/data_gen/filters.py`：

```python
"""三道规则过滤：F1 参数合法 / F2 序列合法 / F3 等级一致。

过滤权威：等级真值由 agent.graph.synthesizer.compute_warning_level 对回放
工具结果重算得出，与线上规则引擎单一同源。
"""
from enum import Enum

from agent.graph.synthesizer import compute_warning_level
from agent.tools.schemas import TOOL_PARAM_MODELS
from train.data_gen.hermes_format import parse_final_answer, parse_trace
from train.data_gen.scenario import Scenario


class FilterResult(Enum):
    ACCEPT = "accept"
    REJECT_F1 = "reject_f1_params"
    REJECT_F2 = "reject_f2_sequence"
    REJECT_F3 = "reject_f3_level"


def _f1_params_valid(tool_calls: list) -> bool:
    for call in tool_calls:
        model = TOOL_PARAM_MODELS.get(call["name"])
        if model is None:
            return False
        try:
            model(**call["arguments"])
        except Exception:
            return False
    return True


def _f2_sequence_valid(tool_calls: list) -> bool:
    names = [c["name"] for c in tool_calls]
    if len(names) != len(set(names)):  # 重复调用
        return False
    if "predict_runoff" in names and "get_weather" not in names[: names.index("predict_runoff")]:
        return False
    if "generate_plan" in names and names.index("generate_plan") != len(names) - 1:
        return False
    return True


def filter_trace(messages: list, scenario: Scenario) -> FilterResult:
    """对单条轨迹执行三道过滤。"""
    if scenario.query_type == "chatty":
        trace = parse_trace(messages)
        if trace is None or trace["tool_calls"]:
            return FilterResult.REJECT_F2
        # 闲聊样本要求最终段不含任何等级字样
        return FilterResult.ACCEPT if parse_final_answer(messages) is None else FilterResult.REJECT_F3

    trace = parse_trace(messages)
    if trace is None:
        return FilterResult.REJECT_F1
    if not _f1_params_valid(trace["tool_calls"]):
        return FilterResult.REJECT_F1
    if not _f2_sequence_valid(trace["tool_calls"]):
        return FilterResult.REJECT_F2
    # F3：用回放结果重算规则等级，与轨迹最终等级比对
    tool_results = {f"tool_{i}": r for i, r in enumerate(trace["tool_results"])}
    truth, _ = compute_warning_level(tool_results)
    model_level = parse_final_answer(messages)
    if model_level is None or model_level != truth or truth != scenario.expected_level:
        return FilterResult.REJECT_F3
    return FilterResult.ACCEPT
```

- [ ] **Step 4: 跑绿**

Run: `python -m pytest train/tests/test_filters.py -v`
Expected: 7 passed

- [ ] **Step 5: 提交**

```bash
git add train/data_gen/filters.py train/tests/test_filters.py
git commit -m "feat(train): three-gate rule filters reusing synthesizer as truth source"
```

---

### Task 6: 教师合成客户端（teacher.py）

**Files:**
- Create: `train/data_gen/teacher.py`
- Test: `train/tests/test_teacher.py`

要点：OpenAI 客户端指向 DashScope；工具结果用 `execute_tool(..., overrides, seed)` 确定性回放
（回放前 monkeypatch `agent.tools.real_executor.real_execute_tool` 抛 RuntimeError，强制走 mock，
防止训练数据生成依赖 Qdrant 等外部状态）；令牌桶限速；断点续传。

- [ ] **Step 1: 写失败测试（OpenAI 调用全部 mock 掉）**

`train/tests/test_teacher.py`：

```python
"""教师合成：轨迹拼装、断点续传、轮次上限。"""
import json
from pathlib import Path
from unittest.mock import MagicMock

from train.data_gen.scenario import generate_scenarios
from train.data_gen.teacher import synthesize_one, synthesize_dataset


def _scenario(qtype="multi_tool"):
    return next(s for s in generate_scenarios(n=20, seed=7) if s.query_type == qtype)


def _fc_response(tool_name: str, arguments: dict, call_id: str = "call_1"):
    """构造 OpenAI SDK 风格的 tool_calls 响应对象。"""
    call = MagicMock()
    call.id = call_id
    call.type = "function"
    call.function.name = tool_name
    call.function.arguments = json.dumps(arguments, ensure_ascii=False)
    msg = MagicMock()
    msg.tool_calls = [call]
    msg.content = None
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _text_response(text: str):
    msg = MagicMock()
    msg.tool_calls = None
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def test_synthesize_one_builds_trace():
    scn = _scenario()
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _fc_response("get_weather", {"location": scn.station, "hours": 24}, "call_1"),
        _fc_response("get_hydrology", {"station": scn.station, "metric": "both"}, "call_2"),
        _fc_response("predict_runoff", {"station": scn.station, "lead_time_hours": 24}, "call_3"),
        _text_response("流量 3250m³/s，发布Ⅱ级预警。"),
    ]
    trace = synthesize_one(client, "fake-model", scn, max_rounds=8)
    assert trace is not None
    roles = [m["role"] for m in trace]
    assert roles[0] == "system" and roles[1] == "user"
    assert "assistant" in roles and "tool" in roles
    # mock 覆盖值已注入回放结果
    hydro = next(m for m in trace if m["role"] == "tool" and scn.station in m["content"])
    assert str(scn.tool_overrides["get_hydrology"]["flow_m3_s"]) in hydro["content"]


def test_synthesize_one_gives_up_at_max_rounds():
    scn = _scenario()
    client = MagicMock()
    client.chat.completions.create.return_value = _fc_response(
        "get_hydrology", {"station": scn.station, "metric": "both"}
    )
    assert synthesize_one(client, "m", scn, max_rounds=2) is None


def test_resume_skips_completed(tmp_path: Path):
    out = tmp_path / "raw.jsonl"
    scn1, scn2 = generate_scenarios(n=2, seed=11)[:2]
    out.write_text(json.dumps({"scenario_id": scn1.scenario_id, "messages": []}, ensure_ascii=False) + "\n")
    client = MagicMock()
    client.chat.completions.create.return_value = _text_response("Ⅳ级，水情平稳。")
    written = synthesize_dataset(client, "m", [scn1, scn2], out, rpm=10000)
    assert written == 1  # scn1 已存在被跳过
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
```

- [ ] **Step 2: 跑红**

Run: `python -m pytest train/tests/test_teacher.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 最小实现**

`train/data_gen/teacher.py`：

```python
"""教师模型合成 Hermes 轨迹（DashScope qwen-plus）。

- 双轨消息：发给 API 的 api_messages 用原生 tool_calls/tool_call_id 结构
  （OpenAI 兼容服务硬性要求）；落盘的 messages 用 Hermes 文本格式（训练格式）
- 工具结果确定性回放：execute_tool(overrides=场景覆盖值, seed=hash(scenario_id))
- 强制走 mock：回放前屏蔽 real_executor，避免依赖 Qdrant/外部 API
- 限速：简单令牌间隔（60/rpm 秒）；断点续传：按 scenario_id 跳过
"""
import json
import logging
import time
from pathlib import Path
from typing import Optional

from agent.tools import mock_executor
from agent.tools.schemas import build_openai_tools
from backend.app.core.llm import get_default_system_prompt
from train.data_gen.hermes_format import make_tool_call_text, make_tool_response_text
from train.data_gen.scenario import Scenario

logger = logging.getLogger(__name__)


def _force_mock() -> None:
    """让 execute_tool 的真实实现分支永远不可用（仅作用于本进程）。"""
    import agent.tools.real_executor as re_mod

    def _unavailable(*a, **k):
        raise RuntimeError("data-gen: real executor disabled")

    re_mod.real_execute_tool = _unavailable


def _replay_tool(scn: Scenario, name: str, arguments: dict) -> dict:
    overrides = scn.tool_overrides.get(name)
    seed = abs(hash(f"{scn.scenario_id}:{name}")) % (2**31)
    return mock_executor.execute_tool(name, arguments, overrides=overrides, seed=seed)


def synthesize_one(client, model: str, scn: Scenario, max_rounds: int = 8) -> Optional[list]:
    """单场景多轮合成。达到轮次上限或教师输出非法 → None。

    返回 Hermes 文本格式轨迹（落盘用），不含 API 原生 tool_calls 结构。
    """
    _force_mock()
    # 落盘轨迹（Hermes 文本格式）
    messages = [
        {"role": "system", "content": get_default_system_prompt()},
        {"role": "user", "content": scn.query},
    ]
    # API 消息（原生格式；assistant 带 tool_calls 数组、tool 带 tool_call_id）
    api_messages = list(messages)
    for _ in range(max_rounds):
        resp = client.chat.completions.create(
            model=model, messages=api_messages, tools=build_openai_tools(), temperature=0.7,
        )
        msg = resp.choices[0].message
        sdk_calls = getattr(msg, "tool_calls", None) or []
        if not sdk_calls:
            if not msg.content:
                return None
            messages.append({"role": "assistant", "content": msg.content})
            return messages
        # Hermes 文本：拼接本轮全部调用块
        hermes_text = ""
        for tc in sdk_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                return None
            hermes_text += make_tool_call_text(tc.function.name, args)
        messages.append({"role": "assistant", "content": hermes_text})
        # API 侧：原生 assistant tool_calls 消息 + 逐个 tool 回复
        api_messages.append(msg)
        for tc in sdk_calls:
            args = json.loads(tc.function.arguments)
            try:
                result = _replay_tool(scn, tc.function.name, args)
            except ValueError:
                return None  # 教师产出非法工具/参数，直接丢弃（不进过滤流程）
            messages.append({"role": "tool", "content": make_tool_response_text(result)})
            api_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False),
            })
    return None


def load_done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(json.loads(line)["scenario_id"])
    return done


def synthesize_dataset(client, model: str, scenarios: list, out_path: Path, rpm: int = 30) -> int:
    """批量合成 + 追加写盘 + 断点续传。返回本次新写入条数。"""
    done = load_done_ids(out_path)
    interval = 60.0 / max(rpm, 1)
    written = 0
    with out_path.open("a", encoding="utf-8") as f:
        for scn in scenarios:
            if scn.scenario_id in done:
                continue
            t0 = time.time()
            try:
                trace = synthesize_one(client, model, scn)
            except Exception as e:
                logger.warning("[teacher] %s 合成异常跳过: %s", scn.scenario_id, e)
                trace = None
            if trace is not None:
                f.write(json.dumps({
                    "scenario_id": scn.scenario_id,
                    "level": scn.expected_level,
                    "query_type": scn.query_type,
                    "messages": trace,
                }, ensure_ascii=False) + "\n")
                f.flush()
                written += 1
            time.sleep(max(0.0, interval - (time.time() - t0)))
    return written
```

- [ ] **Step 4: 跑绿**

Run: `python -m pytest train/tests/test_teacher.py -v`
Expected: 3 passed

- [ ] **Step 5: 提交**

```bash
git add train/data_gen/teacher.py train/tests/test_teacher.py
git commit -m "feat(train): teacher synthesis client with deterministic replay and resume"
```

---

### Task 7: 数据集编排 + 统计 + 全量合成（M1 验收）

**Files:**
- Create: `train/data_gen/build_dataset.py`
- Create: `train/data_gen/stats.py`
- Test: `train/tests/test_stats.py`

- [ ] **Step 1: 写失败测试**

`train/tests/test_stats.py`：

```python
"""统计报告：分布与过滤率正确。"""
from train.data_gen.stats import summarize


def test_summarize_counts():
    records = [
        {"level": "I", "query_type": "multi_tool", "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "<tool_call>\n{\"name\": \"get_hydrology\", \"arguments\": {}}\n</tool_call>"},
            {"role": "tool", "content": "{}"},
            {"role": "assistant", "content": "Ⅰ级"},
        ]},
        {"level": "II", "query_type": "single_tool", "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "Ⅱ级"},
        ]},
    ]
    rejects = {"reject_f1_params": 1, "reject_f2_sequence": 2, "reject_f3_level": 3}
    s = summarize(records, rejects, total_scenarios=8)
    assert s["accepted"] == 2
    assert s["level_dist"] == {"I": 1, "II": 1}
    assert s["reject_dist"]["reject_f3_level"] == 3
    assert s["accept_rate"] == 0.25
    assert s["avg_rounds"] == 1.5  # (2 assistant 段 + 1) / 2
```

- [ ] **Step 2: 跑红**

Run: `python -m pytest train/tests/test_stats.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现**

`train/data_gen/stats.py`：

```python
"""数据集统计报告（FR-D6）。"""
import json
from collections import Counter
from pathlib import Path


def summarize(records: list, reject_counter: dict, total_scenarios: int) -> dict:
    accepted = len(records)
    level_dist = Counter(r["level"] for r in records if r.get("level"))
    qtype_dist = Counter(r["query_type"] for r in records)
    rounds = [
        sum(1 for m in r["messages"] if m["role"] == "assistant") for r in records
    ]
    return {
        "total_scenarios": total_scenarios,
        "accepted": accepted,
        "accept_rate": round(accepted / max(total_scenarios, 1), 4),
        "level_dist": dict(level_dist),
        "query_type_dist": dict(qtype_dist),
        "reject_dist": dict(reject_counter),
        "avg_rounds": round(sum(rounds) / max(len(rounds), 1), 2),
    }


def print_report(summary: dict) -> str:
    lines = ["# 数据集统计报告", ""]
    for k, v in summary.items():
        lines.append(f"- **{k}**: {v}")
    text = "\n".join(lines)
    print(text)
    return text
```

`train/data_gen/build_dataset.py`：

```python
"""数据集编排入口：场景 → 教师合成 → 三道过滤 → JSONL + train/val 切分 + 统计。

用法：
  python -m train.data_gen.build_dataset --n 5000 --seed 1000 \
      --out train/lora/data/hermes_fc_v1.jsonl --rpm 30
干跑（不调用教师 API，验证编排）：
  python -m train.data_gen.build_dataset --n 20 --seed 1000 --dry-run
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from openai import OpenAI

from backend.app.core.config import get_settings
from train.data_gen.filters import FilterResult, filter_trace
from train.data_gen.scenario import generate_scenarios
from train.data_gen.stats import print_report, summarize
from train.data_gen.teacher import synthesize_dataset


def split_train_val(records: list, val_ratio: float, seed: int) -> tuple:
    rng = random.Random(seed)
    by_level: dict[str, list] = {}
    for r in records:
        by_level.setdefault(r["level"] or "chatty", []).append(r)
    train, val = [], []
    for group in by_level.values():
        rng.shuffle(group)
        k = max(1, round(len(group) * val_ratio))
        val.extend(group[:k])
        train.extend(group[k:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("train/lora/data/hermes_fc_v1.jsonl"))
    parser.add_argument("--rpm", type=int, default=30)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true", help="只生成场景并打印配额，不调 API")
    args = parser.parse_args()

    scenarios = generate_scenarios(n=args.n, seed=args.seed)
    print(f"[build] 场景 {len(scenarios)} 条（seed={args.seed}）")
    if args.dry_run:
        print(Counter(s.expected_level or "chatty" for s in scenarios))
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.out.with_suffix(".raw.jsonl")

    settings = get_settings()
    client = OpenAI(api_key=settings.LLM_API_KEY, base_url=settings.LLM_BASE_URL)
    written = synthesize_dataset(client, settings.LLM_MODEL, scenarios, raw_path, rpm=args.rpm)
    print(f"[build] 本次合成 {written} 条 → {raw_path}")

    records, rejects = [], Counter()
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        scn = next(s for s in scenarios if s.scenario_id == rec["scenario_id"])
        result = filter_trace(rec["messages"], scn)
        if result == FilterResult.ACCEPT:
            records.append(rec)
        else:
            rejects[result.value] += 1

    train, val = split_train_val(records, args.val_ratio, args.seed)
    with args.out.open("w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    val_path = args.out.with_suffix(".val.jsonl")
    with val_path.open("w", encoding="utf-8") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = summarize(records, dict(rejects), len(scenarios))
    print_report(summary)
    print(f"[build] train={len(train)} → {args.out}  val={len(val)} → {val_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑绿 + 干跑验证**

Run: `python -m pytest train/tests/test_stats.py -v; python -m train.data_gen.build_dataset --n 20 --seed 1000 --dry-run`
Expected: 1 passed；干跑打印等级计数（I/II/III/IV 各 ~5，chatty ~2）

- [ ] **Step 5: 全量合成（M1 验收，需 LLM_API_KEY）**

```bash
python -m train.data_gen.build_dataset --n 5000 --seed 1000 --rpm 30
```

Expected（验收 AC-1）：accept_rate ≥ 0.70；产出 3k-5k 条；`train/lora/data/` 下
`hermes_fc_v1.jsonl` + `hermes_fc_v1.val.jsonl`；抽样 200 条人工核验等级一致率 ≥ 98%。
若 accept_rate < 0.70：调整 `get_default_system_prompt` 的教师侧提示（如显式要求
"最终回答必须含 Ⅰ/Ⅱ/Ⅲ/Ⅳ 级字样"）后重跑，**禁止放宽过滤器**。

- [ ] **Step 6: 提交**

```bash
git add train/data_gen/build_dataset.py train/data_gen/stats.py train/tests/test_stats.py
git commit -m "feat(train): dataset orchestration with filtering, split and stats report"
```

数据文件大（>10MB），确认 `.gitignore` 已含 `train/lora/data/` 与 `train/**/outputs/`：

```bash
echo "train/lora/data/" >> .gitignore; echo "train/**/outputs/" >> .gitignore
git add .gitignore; git commit -m "chore: ignore generated datasets and training outputs"
```

---

## Phase 2 — LoRA SFT（M2）

### Task 8: SFT 数据预处理（模板渲染 + completion mask）

**Files:**
- Create: `train/lora/__init__.py`
- Create: `train/lora/dataset.py`
- Test: `train/tests/test_sft_dataset.py`

- [ ] **Step 1: 写失败测试（用小 tokenizer，不下载 7B）**

`train/tests/test_sft_dataset.py`：

```python
"""SFT 预处理：assistant-only labels，非 assistant 段 -100。"""
import pytest

transformers = pytest.importorskip("transformers")
from transformers import AutoTokenizer  # noqa: E402

from train.lora.dataset import build_sft_dataset, render_messages  # noqa: E402


@pytest.fixture(scope="module")
def tokenizer():
    # 轻量 tokenizer 验证逻辑；训练时换成 Qwen2.5-7B-Instruct
    return AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-Qwen2ForCausalLM")


def _record():
    return {"messages": [
        {"role": "system", "content": "你是防汛智能体"},
        {"role": "user", "content": "吴堡水情？"},
        {"role": "assistant", "content": "<tool_call>\n{\"name\": \"get_hydrology\", \"arguments\": {\"station\": \"吴堡\"}}\n</tool_call>"},
        {"role": "tool", "content": "<tool_response>\n{\"flow_m3_s\": 3250.0}\n</tool_response>"},
        {"role": "assistant", "content": "发布Ⅱ级预警。"},
    ]}


def test_render_contains_hermes_tags(tokenizer):
    text = render_messages(_record()["messages"], tokenizer)
    assert "<tool_call>" in text and "<tool_response>" in text
    assert "<|im_start|>" in text


def test_assistant_only_loss(tokenizer):
    ds = build_sft_dataset([_record()], tokenizer, max_len=512)
    sample = ds[0]
    ids, labels = sample["input_ids"], sample["labels"]
    assert len(ids) == len(labels)
    # 有监督 token（labels != -100）非空且占比合理
    supervised = [i for i, lab in zip(ids, labels) if lab != -100]
    assert 0 < len(supervised) < len(ids)
    # 监督区解码后应包含最终等级文本
    text = tokenizer.decode(supervised)
    assert "预警" in text or "<tool_call>" in text
    # system/user 段不监督：首段 labels 全 -100
    assert labels[0] == -100
```

- [ ] **Step 2: 跑红**

Run: `python -m pytest train/tests/test_sft_dataset.py -v`
Expected: FAIL（ModuleNotFoundError: train.lora）

- [ ] **Step 3: 实现**

`train/lora/__init__.py` 空文件。`train/lora/dataset.py`：

```python
"""SFT 数据集预处理：Qwen 聊天模板渲染 + assistant-only loss。

不依赖 tokenizer 自带 {% generation %} 标记（小 tokenizer 没有），
改为分段编码：assistant 段（含 <|im_start|>assistant\n 前缀）计 loss，其余 -100。
"""
from datasets import Dataset

IM_START, IM_END = "<|im_start|>", "<|im_end|>"


def render_messages(messages: list, tokenizer=None) -> str:
    """messages → Qwen 模板文本（与 Qwen2.5 官方 apply_chat_template 等价）。"""
    parts = []
    for m in messages:
        parts.append(f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n")
    return "".join(parts)


def _encode_with_masks(messages: list, tokenizer, max_len: int) -> dict:
    """分段编码，构造 input_ids / labels / attention_mask。"""
    input_ids: list[int] = []
    labels: list[int] = []
    for m in messages:
        seg = f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n"
        ids = tokenizer(seg, add_special_tokens=False)["input_ids"]
        if m["role"] == "assistant":
            seg_labels = list(ids)  # assistant 段计 loss
        else:
            seg_labels = [-100] * len(ids)
        input_ids.extend(ids)
        labels.extend(seg_labels)
    input_ids, labels = input_ids[:max_len], labels[:max_len]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


def build_sft_dataset(records: list, tokenizer, max_len: int = 4096) -> Dataset:
    """JSONL records → HF Dataset（已编码）。"""
    features = [_encode_with_masks(r["messages"], tokenizer, max_len) for r in records]
    return Dataset.from_list(features)


def load_jsonl(path) -> list:
    import json
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
```

- [ ] **Step 4: 跑绿**

Run: `python -m pytest train/tests/test_sft_dataset.py -v`
Expected: 2 passed（hf-internal-testing 模型首次会从 HF 拉取，需联网；离线环境预先缓存）

- [ ] **Step 5: 提交**

```bash
git add train/lora/__init__.py train/lora/dataset.py train/tests/test_sft_dataset.py
git commit -m "feat(train): SFT preprocessing with assistant-only completion mask"
```

---

### Task 9: SFT 训练脚本 + smoke run

**Files:**
- Create: `train/lora/configs/sft_qlora.yaml`
- Create: `train/lora/train_sft.py`

- [ ] **Step 1: 配置**

`train/lora/configs/sft_qlora.yaml`：

```yaml
base_model: Qwen/Qwen2.5-7B-Instruct
data: train/lora/data/hermes_fc_v1.jsonl
val_data: train/lora/data/hermes_fc_v1.val.jsonl
output_dir: train/lora/outputs/sft
max_len: 4096
qlora:
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_use_double_quant: true
  compute_dtype: bfloat16
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
train:
  per_device_batch: 2
  grad_accum: 16
  lr: 1.0e-4
  epochs: 3
  warmup_ratio: 0.03
  lr_scheduler: cosine
  gradient_checkpointing: true
  logging_steps: 10
  save_steps: 200
  seed: 42
smoke:
  enabled: false   # --smoke 时覆盖为 true
  n_samples: 10
  max_steps: 5
```

- [ ] **Step 2: 训练脚本**

`train/lora/train_sft.py`：

```python
"""QLoRA SFT 入口（trl SFTTrainer）。

用法：
  smoke: python -m train.lora.train_sft --smoke
  全量:  python -m train.lora.train_sft
"""
import argparse

import torch
import yaml
from datasets import concatenate_datasets
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from train.lora.dataset import build_sft_dataset, load_jsonl


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="train/lora/configs/sft_qlora.yaml")
    parser.add_argument("--smoke", action="store_true", help="10 样本 5 步验证全流程")
    args = parser.parse_args()
    cfg = load_config(args.config)

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    bnb = BitsAndBytesConfig(
        load_in_4bit=cfg["qlora"]["load_in_4bit"],
        bnb_4bit_quant_type=cfg["qlora"]["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=cfg["qlora"]["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], quantization_config=bnb, device_map="auto",
    )

    records = load_jsonl(cfg["data"])
    if args.smoke:
        records = records[: cfg["smoke"]["n_samples"]]
    train_ds = build_sft_dataset(records, tokenizer, cfg["max_len"])

    sft_cfg = SFTConfig(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["train"]["per_device_batch"],
        gradient_accumulation_steps=cfg["train"]["grad_accum"],
        learning_rate=cfg["train"]["lr"],
        num_train_epochs=1 if args.smoke else cfg["train"]["epochs"],
        max_steps=cfg["smoke"]["max_steps"] if args.smoke else -1,
        warmup_ratio=cfg["train"]["warmup_ratio"],
        lr_scheduler_type=cfg["train"]["lr_scheduler"],
        gradient_checkpointing=cfg["train"]["gradient_checkpointing"],
        logging_steps=cfg["train"]["logging_steps"],
        save_steps=cfg["train"]["save_steps"],
        save_total_limit=3,
        bf16=True,
        seed=cfg["train"]["seed"],
        report_to=[],
        dataset_num_proc=1,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        peft_config=LoraConfig(
            r=cfg["lora"]["r"],
            lora_alpha=cfg["lora"]["alpha"],
            lora_dropout=cfg["lora"]["dropout"],
            target_modules=cfg["lora"]["target_modules"],
            task_type="CAUSAL_LM",
        ),
    )
    trainer.train()
    trainer.save_model(f"{cfg['output_dir']}/adapter")
    print(f"[sft] adapter → {cfg['output_dir']}/adapter")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: smoke run（M2 前置验收）**

```bash
python -m train.lora.train_sft --smoke
```

Expected：5 步内 loss 正常下降（非 NaN/inf），产出 adapter 目录；
`nvidia-smi` 峰值显存 < 24GB。OOM 时按 design.md 4.3 降级：per_device 2→1 →
seq 4096→3072 → target_modules 裁剪 [q,k,v,o]_proj。

- [ ] **Step 4: 全量训练**

```bash
python -m train.lora.train_sft
```

Expected：≤ 12h（NFR-2）；loss 曲线收敛；保留 best/last checkpoint。

- [ ] **Step 5: 提交**

```bash
git add train/lora/configs/sft_qlora.yaml train/lora/train_sft.py
git commit -m "feat(train): QLoRA SFT trainer with smoke mode"
```

---

### Task 10: adapter 合并（merge.py）

**Files:**
- Create: `train/lora/merge.py`
- Test: `train/tests/test_merge.py`

- [ ] **Step 1: 写失败测试**

`train/tests/test_merge.py`：

```python
"""合并脚本：PEFT 模型 merge_and_unload 被调用，输出目录正确。"""
from unittest.mock import MagicMock, patch

from train.lora.merge import merge_adapter


def test_merge_calls_peft_api(tmp_path):
    fake_model = MagicMock()
    fake_model.merge_and_unload.return_value = fake_model
    with patch("train.lora.merge.AutoModelForCausalLM") as m_auto, \
         patch("train.lora.merge.PeftModel") as m_peft, \
         patch("train.lora.merge.AutoTokenizer") as m_tok:
        m_peft.from_pretrained.return_value = fake_model
        out = merge_adapter("base-x", "adapter-y", str(tmp_path / "merged"))
    m_auto.from_pretrained.assert_called_once()
    m_peft.from_pretrained.assert_called_once()
    fake_model.merge_and_unload.assert_called_once()
    fake_model.save_pretrained.assert_called_once_with(str(tmp_path / "merged"))
    assert out.endswith("merged")
```

- [ ] **Step 2: 跑红**

Run: `python -m pytest train/tests/test_merge.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现**

`train/lora/merge.py`：

```python
"""LoRA adapter 合并为全量权重（BF16，供 vLLM 加载）。

用法：python -m train.lora.merge [--adapter train/lora/outputs/sft/adapter]
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_adapter(base_model: str, adapter_path: str, out_dir: str) -> str:
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, adapter_path)
    merged = model.merge_and_unload()
    merged.save_pretrained(out_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(out_dir)
    print(f"[merge] → {out_dir}")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter", default="train/lora/outputs/sft/adapter")
    parser.add_argument("--out", default="train/lora/outputs/sft/merged")
    args = parser.parse_args()
    merge_adapter(args.base, args.adapter, args.out)
```

- [ ] **Step 4: 跑绿 + 实际合并**

Run: `python -m pytest train/tests/test_merge.py -v; python -m train.lora.merge`
Expected: 1 passed；`train/lora/outputs/sft/merged/` 含 safetensors + tokenizer（~15GB）

- [ ] **Step 5: 提交**

```bash
git add train/lora/merge.py train/tests/test_merge.py
git commit -m "feat(train): LoRA adapter merge script for vLLM serving"
```

---

## Phase 3 — GRPO 对齐（M3）

### Task 11: 规则奖励四模块 + composite

**Files:**
- Create: `train/rewards/__init__.py`
- Create: `train/rewards/format_gate.py`
- Create: `train/rewards/r1_level.py`
- Create: `train/rewards/r2_tool_call.py`
- Create: `train/rewards/r3_plan.py`
- Create: `train/rewards/composite.py`
- Test: `train/tests/test_rewards.py`

- [ ] **Step 1: 写失败测试**

`train/tests/test_rewards.py`：

```python
"""奖励函数：门控 / 等级 / 工具 / 预案，全对=1.0，格式坏=0。"""
from unittest.mock import patch

from train.data_gen.scenario import generate_scenarios
from train.rewards.composite import compute_reward


def _scn(level="II"):
    # 必须选 multi_tool 场景：其 reference_tools 覆盖三工具，
    # 与 _good_completion 的调用集合一致
    return next(s for s in generate_scenarios(n=50, seed=21)
                if s.expected_level == level and s.query_type == "multi_tool")


def _good_completion(scn):
    flow = scn.tool_overrides["get_hydrology"]["flow_m3_s"]
    return (
        "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"%s\", \"hours\": 24}}\n</tool_call>"
        "<tool_call>\n{\"name\": \"get_hydrology\", \"arguments\": {\"station\": \"%s\", \"metric\": \"both\"}}\n</tool_call>"
        "<tool_call>\n{\"name\": \"predict_runoff\", \"arguments\": {\"station\": \"%s\", \"lead_time_hours\": 24}}\n</tool_call>"
        "\n综上：流量 %.0fm³/s，发布Ⅱ级（橙色）预警。"
        "\n预案：12 小时内组织危险区域群众转移，调集抢险物资，"
        "吕梁市防汛抗旱指挥部牵头负责。依据《黄河防汛预案》第三章第十二条。"
    ) % (scn.station, scn.station, scn.station, flow)


def test_full_marks():
    scn = _scn("II")
    rag_hits = [{"title": "黄河防汛预案", "article": "第三章 第十二条", "content": "……"}]
    r, parts = compute_reward(_good_completion(scn), scn, rag_hits=rag_hits)
    assert r == 1.0
    assert parts == {"r1": 0.4, "r2": 0.3, "r3": 0.3}


def test_format_gate_zero():
    scn = _scn("II")
    r, parts = compute_reward("<tool_call>{bad json}</tool_call>", scn, rag_hits=[])
    assert r == 0.0 and parts == {}


def test_adjacent_level_partial_credit():
    scn = _scn("II")
    completion = _good_completion(scn).replace("Ⅱ级（橙色）", "Ⅲ级（黄色）")
    r, parts = compute_reward(completion, scn, rag_hits=[{"title": "黄河防汛预案", "article": "第三章 第十二条"}])
    assert parts["r1"] == 0.2  # 相邻一级部分分
    assert 0.0 < r < 1.0


def test_r2_missing_weather_before_runoff():
    scn = _scn("II")
    completion = (
        "<tool_call>\n{\"name\": \"predict_runoff\", \"arguments\": {\"station\": \"吴堡\"}}\n</tool_call>"
        "\n发布Ⅱ级预警。转移群众，调集物资，指挥部负责，12 小时。依据《黄河防汛预案》第三章第十二条。"
    )
    r, parts = compute_reward(completion, scn, rag_hits=[{"title": "黄河防汛预案", "article": "第三章 第十二条"}])
    assert parts["r2"] < 0.3


def test_r3_requires_rag_hit():
    scn = _scn("II")
    r_with, p_with = compute_reward(_good_completion(scn), scn,
                                    rag_hits=[{"title": "黄河防汛预案", "article": "第三章 第十二条"}])
    r_without, p_without = compute_reward(_good_completion(scn), scn, rag_hits=[])
    assert p_with["r3"] == 0.3
    assert p_without["r3"] == 0.15  # 要素在、无 RAG 命中
```

- [ ] **Step 2: 跑红**

Run: `python -m pytest train/tests/test_rewards.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现**

`train/rewards/__init__.py` 空文件。

`train/rewards/format_gate.py`：

```python
"""格式门控：工具调用块全部可解析且最终段非空才放行。"""
from train.data_gen.hermes_format import extract_tool_calls


def gate_pass(completion: str) -> bool:
    has_tag = "<tool_call>" in completion
    calls = extract_tool_calls(completion)
    if has_tag and not calls:
        return False  # 有标签但 JSON 全坏
    final = completion.split("</tool_call>")[-1].strip()
    return bool(final)
```

`train/rewards/r1_level.py`：

```python
"""r1 等级正确性（0.4）：与规则引擎真值一致满分，相邻一级半分。

chatty 场景（truth_level=""）：模型不输出等级得满分，输出等级得 0 分。
"""
from train.data_gen.hermes_format import _LEVEL_MAP

_ADJACENT = {"I": {"II"}, "II": {"I", "III"}, "III": {"II", "IV"}, "IV": {"III"}}


def _extract(text: str) -> str | None:
    for pattern, level in _LEVEL_MAP:
        if pattern.search(text):
            return level
    return None


def r1_score(completion: str, truth_level: str) -> float:
    final = completion.split("</tool_call>")[-1]
    level = _extract(final)
    if truth_level == "":
        return 0.4 if level is None else 0.0
    if level is None:
        return 0.0
    if level == truth_level:
        return 0.4
    if level in _ADJACENT.get(truth_level, set()):
        return 0.2
    return 0.0
```

`train/rewards/r2_tool_call.py`：

```python
"""r2 工具调用正确性（0.3）：参数校验 0.1 + 工具命中 0.1 + 顺序合法 0.1。"""
from agent.tools.schemas import TOOL_PARAM_MODELS
from train.data_gen.hermes_format import extract_tool_calls


def r2_score(completion: str, reference_tools: list) -> float:
    calls = extract_tool_calls(completion)
    if not calls:
        return 0.0
    score = 0.0
    # 0.1 参数全部合法
    ok = True
    for c in calls:
        model = TOOL_PARAM_MODELS.get(c["name"])
        if model is None:
            ok = False
            break
        try:
            model(**c["arguments"])
        except Exception:
            ok = False
            break
    if ok:
        score += 0.1
    # 0.1 调用集合 ⊆ 参考集且至少 1 个必需工具
    names = {c["name"] for c in calls}
    if names and names.issubset(set(reference_tools)) and names:
        score += 0.1
    # 0.1 顺序合法
    seq = [c["name"] for c in calls]
    order_ok = True
    if "predict_runoff" in seq and "get_weather" not in seq[: seq.index("predict_runoff")]:
        order_ok = False
    if "generate_plan" in seq and seq.index("generate_plan") != len(seq) - 1:
        order_ok = False
    if order_ok:
        score += 0.1
    return score
```

`train/rewards/r3_plan.py`：

```python
"""r3 预案质量与法规依据（0.3）：四要素 0.15 + RAG 引用一致 0.15。"""
import re

_PLAN_ELEMENTS = [
    re.compile(r"转移|撤离|疏散"),
    re.compile(r"物资|编织袋|冲锋舟|沙袋|抢险队"),
    re.compile(r"指挥部|责任人|牵头|防指"),
    re.compile(r"\d+\s*小时|时限|立即|小时\s*内"),
]


def _normalize(text: str) -> str:
    return re.sub(r"[《》\s]", "", text)


def r3_score(completion: str, rag_hits: list) -> float:
    final = completion.split("</tool_call>")[-1]
    score = 0.0
    if all(p.search(final) for p in _PLAN_ELEMENTS):
        score += 0.15
    # 引用条款命中 RAG 检索结果（标题或条文号交集非空）
    final_norm = _normalize(final)
    for hit in rag_hits:
        title = _normalize(str(hit.get("title", "")))
        article = _normalize(str(hit.get("article", "")))
        if (title and title in final_norm) or (article and article in final_norm):
            score += 0.15
            break
    return score
```

`train/rewards/composite.py`：

```python
"""奖励合成：reward = gate × (r1 + r2 + r3)；分量日志供曲线绘制。"""
import json
import time
from pathlib import Path

from agent.graph.synthesizer import compute_warning_level
from train.data_gen.scenario import Scenario
from train.rewards.format_gate import gate_pass
from train.rewards.r1_level import r1_score
from train.rewards.r2_tool_call import r2_score
from train.rewards.r3_plan import r3_score

_LOG_PATH = Path("train/grpo/outputs/reward_log.jsonl")


def _truth_level(scn: Scenario) -> str:
    """等级真值：用场景覆盖值重构工具结果，经规则引擎重算（单一权威来源）。

    与 scenario 生成时的 expected_level 一致（生成器按阈值构造），此处重算
    是为了让奖励逻辑直接依赖规则引擎而非元数据副本，防双份规则漂移。
    """
    tool_results = {}
    for tool_name, key in (("get_hydrology", "hydro"), ("get_weather", "wx"),
                           ("predict_runoff", "runoff")):
        if scn.tool_overrides.get(tool_name):
            tool_results[key] = scn.tool_overrides[tool_name]
    if not tool_results:
        return scn.expected_level  # chatty：无工具数据，真值即空串
    level, _ = compute_warning_level(tool_results)
    return level


def compute_reward(completion: str, scn: Scenario, rag_hits: list) -> tuple:
    """返回 (reward, parts)。格式门控失败 → (0.0, {})。"""
    if not gate_pass(completion):
        return 0.0, {}
    parts = {
        "r1": r1_score(completion, _truth_level(scn)),
        "r2": r2_score(completion, scn.reference_tools),
        "r3": r3_score(completion, rag_hits),
    }
    return sum(parts.values()), parts


def log_reward_parts(step: int, parts_list: list, path: Path = _LOG_PATH) -> None:
    """每 step 记录三分量均值（NFR-7）。"""
    if not parts_list:
        return
    valid = [p for p in parts_list if p]
    means = {k: round(sum(p.get(k, 0.0) for p in valid) / max(len(valid), 1), 4)
             for k in ("r1", "r2", "r3")}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.time(), "step": step, **means}) + "\n")
```

- [ ] **Step 4: 跑绿**

Run: `python -m pytest train/tests/test_rewards.py -v`
Expected: 5 passed

- [ ] **Step 5: 提交**

```bash
git add train/rewards/ train/tests/test_rewards.py
git commit -m "feat(train): rule-based GRPO rewards (level/tools/plan) with format gate"
```

---

### Task 12: GRPO prompts + 多轮 rollout 环境

**Files:**
- Create: `train/grpo/__init__.py`
- Create: `train/grpo/prompts.py`
- Create: `train/grpo/rollouts.py`
- Test: `train/tests/test_grpo_prompts.py`

- [ ] **Step 1: 写失败测试**

`train/tests/test_grpo_prompts.py`：

```python
"""GRPO prompts 与 SFT 集零重叠；rollout 回放确定性。"""
from unittest.mock import patch

from train.data_gen.scenario import generate_scenarios
from train.grpo.prompts import build_grpo_prompts
from train.grpo.rollouts import replay_tool_call


def test_prompts_disjoint_from_sft():
    sft = generate_scenarios(n=100, seed=1000)
    grpo = build_grpo_prompts(n=100)
    assert {s.scenario_id for s in sft}.isdisjoint({p["scenario"].scenario_id for p in grpo})


def test_prompt_carries_system_and_query():
    p = build_grpo_prompts(n=1)[0]
    assert p["prompt"][0]["role"] == "system"
    assert p["prompt"][-1]["role"] == "user"
    assert p["scenario"].expected_level in ("I", "II", "III", "IV", "")


def test_replay_uses_scenario_overrides():
    scn = next(s for s in generate_scenarios(n=10, seed=100500) if s.query_type != "chatty")
    out = replay_tool_call(scn, "get_hydrology", {"station": scn.station, "metric": "both"})
    assert out["flow_m3_s"] == scn.tool_overrides["get_hydrology"]["flow_m3_s"]
    assert out["source"] == "mock"
```

- [ ] **Step 2: 跑红**

Run: `python -m pytest train/tests/test_grpo_prompts.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现**

`train/grpo/__init__.py` 空文件。`train/grpo/prompts.py`：

```python
"""GRPO prompt 集：独立种子区间 [100_000, 200_000)，与 SFT/评估零重叠。"""
from backend.app.core.llm import get_default_system_prompt
from train.data_gen.scenario import generate_scenarios

GRPO_SEED_BASE = 100_000


def build_grpo_prompts(n: int, seed: int = GRPO_SEED_BASE) -> list:
    scenarios = generate_scenarios(n=n, seed=seed)
    return [
        {
            "prompt": [
                {"role": "system", "content": get_default_system_prompt()},
                {"role": "user", "content": scn.query},
            ],
            "scenario": scn,
        }
        for scn in scenarios
    ]
```

`train/grpo/rollouts.py`：

```python
"""GRPO 工具回放：与数据生成共用同一确定性回放逻辑。

说明：trl GRPOTrainer 标准流程为单轮补全——模型在一次生成中输出完整
调用计划（多个 <tool_call> 块）+ 最终研判段，无需交互式多轮 rollout；
本模块只提供确定性回放（供多轮扩展与评估复用）。
"""
from agent.tools import mock_executor
from train.data_gen.scenario import Scenario


def replay_tool_call(scn: Scenario, name: str, arguments: dict) -> dict:
    overrides = scn.tool_overrides.get(name)
    seed = abs(hash(f"{scn.scenario_id}:{name}")) % (2**31)
    return mock_executor.execute_tool(name, arguments, overrides=overrides, seed=seed)
```

- [ ] **Step 4: 跑绿**

Run: `python -m pytest train/tests/test_grpo_prompts.py -v`
Expected: 3 passed

- [ ] **Step 5: 提交**

```bash
git add train/grpo/__init__.py train/grpo/prompts.py train/grpo/rollouts.py train/tests/test_grpo_prompts.py
git commit -m "feat(train): GRPO prompts with seed isolation and deterministic rollout"
```

---

### Task 13: GRPO 训练脚本 + smoke run（M3 验收）

**Files:**
- Create: `train/grpo/configs/grpo.yaml`
- Create: `train/grpo/train_grpo.py`

- [ ] **Step 1: 配置**

`train/grpo/configs/grpo.yaml`：

```yaml
sft_model: train/lora/outputs/sft/merged   # 从 SFT 合并权重出发
output_dir: train/grpo/outputs
n_prompts: 512
grpo:
  group_size: 8            # OOM → 4
  temperature: 1.0
  max_completion_length: 1024
  lr: 1.0e-6
  beta: 0.04               # KL 系数
  epochs: 2
  per_device_batch: 1
  grad_accum: 8
  gradient_checkpointing: true
  save_steps: 50
  seed: 42
vllm:
  mode: colocate
  gpu_memory_utilization: 0.35   # OOM → 0.25
smoke:
  enabled: false
  n_prompts: 8
  max_steps: 3
```

- [ ] **Step 2: 训练脚本**

`train/grpo/train_grpo.py`：

```python
"""GRPO 对齐入口（trl GRPOTrainer + vLLM colocate + 规则奖励）。

用法：
  smoke: python -m train.grpo.train_grpo --smoke
  全量:  python -m train.grpo.train_grpo
奖励接线：trl 的 reward_funcs 接收 completions 与 dataset 列；
scenario 以 JSON 字符串存列（避免 HF datasets 对嵌套 dict 的 struct 类型强转），
奖励函数内还原并调 compute_reward。
补全格式：单轮——模型一次输出多个 <tool_call> 块 + 最终研判段。
"""
import argparse
import json

import yaml
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from train.data_gen.scenario import Scenario
from train.grpo.prompts import build_grpo_prompts
from train.rewards.composite import compute_reward, log_reward_parts

_STEP = {"n": 0}


def _serialize(scn: Scenario) -> str:
    return json.dumps({
        "scenario_id": scn.scenario_id,
        "expected_level": scn.expected_level,
        "reference_tools": scn.reference_tools,
        "tool_overrides": scn.tool_overrides,
        "query_type": scn.query_type,
        "station": scn.station,
        "query": scn.query,
    }, ensure_ascii=False)


def _deserialize(s: str) -> Scenario:
    d = json.loads(s)
    return Scenario(
        scenario_id=d["scenario_id"], station=d["station"], query=d["query"],
        query_type=d["query_type"], expected_level=d["expected_level"],
        tool_overrides=d["tool_overrides"], reference_tools=d["reference_tools"],
    )


def rule_reward(completions, scenario, **kwargs) -> list:
    """trl reward_func：每条补全算 reward = gate × (r1+r2+r3)。

    rag_hits：训练期用确定性 mock 检索结果（mock_executor._mock_search_regulation
    的固定文档集），与场景无关，保证可复现。
    """
    from agent.tools.mock_executor import _mock_search_regulation
    from agent.tools.schemas import SearchRegulationParams

    rewards = []
    parts_list = []
    for completion, scn_json in zip(completions, scenario, strict=True):
        scn = _deserialize(scn_json)
        rag = _mock_search_regulation(SearchRegulationParams(query=scn.query, top_k=3))["hits"]
        text = completion if isinstance(completion, str) else completion[-1]["content"]
        r, parts = compute_reward(text, scn, rag_hits=rag)
        rewards.append(r)
        parts_list.append(parts)
    _STEP["n"] += 1
    log_reward_parts(_STEP["n"], parts_list)
    return rewards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="train/grpo/configs/grpo.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    n = cfg["smoke"]["n_prompts"] if args.smoke else cfg["n_prompts"]
    prompts = build_grpo_prompts(n=n)
    dataset = Dataset.from_list([
        {"prompt": p["prompt"], "scenario": _serialize(p["scenario"])} for p in prompts
    ])

    gcfg = GRPOConfig(
        output_dir=cfg["output_dir"],
        learning_rate=cfg["grpo"]["lr"],
        beta=cfg["grpo"]["beta"],
        num_generations=cfg["grpo"]["group_size"],
        temperature=cfg["grpo"]["temperature"],
        max_completion_length=cfg["grpo"]["max_completion_length"],
        per_device_train_batch_size=cfg["grpo"]["per_device_batch"],
        gradient_accumulation_steps=cfg["grpo"]["grad_accum"],
        num_train_epochs=1 if args.smoke else cfg["grpo"]["epochs"],
        max_steps=cfg["smoke"]["max_steps"] if args.smoke else -1,
        gradient_checkpointing=cfg["grpo"]["gradient_checkpointing"],
        save_steps=cfg["grpo"]["save_steps"],
        logging_steps=1,
        bf16=True,
        seed=cfg["grpo"]["seed"],
        report_to=[],
        use_vllm=True,
        vllm_mode=cfg["vllm"]["mode"],
        vllm_gpu_memory_utilization=cfg["vllm"]["gpu_memory_utilization"],
    )
    trainer = GRPOTrainer(
        model=cfg["sft_model"],
        args=gcfg,
        train_dataset=dataset,
        reward_funcs=rule_reward,
    )
    trainer.train()
    trainer.save_model(f"{cfg['output_dir']}/adapter")
    print(f"[grpo] adapter → {cfg['output_dir']}/adapter；合并复用 train/lora/merge.py")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: smoke run + 全量（M3 验收）**

```bash
python -m train.grpo.train_grpo --smoke   # 3 步跑通，reward_log.jsonl 有分量记录
python -m train.grpo.train_grpo           # ≤ 24h（NFR-2）
python -m train.lora.merge --adapter train/grpo/outputs/adapter --out train/grpo/outputs/merged
```

Expected：奖励曲线（`train/grpo/outputs/reward_log.jsonl`）r1/r2/r3 均值随 step 上升；
OOM 按 design.md 5.4 降级（G 8→4 → completion 1024→768 → vLLM 显存 0.35→0.25）。

- [ ] **Step 4: 提交**

```bash
git add train/grpo/configs/grpo.yaml train/grpo/train_grpo.py
git commit -m "feat(train): GRPO trainer with vLLM colocate and rule rewards"
```

---

## Phase 4 — 评估（M4）

### Task 14: 三模型对比评估

**Files:**
- Create: `train/eval/__init__.py`
- Create: `train/eval/run_eval.py`
- Create: `train/eval/report.py`
- Test: `train/tests/test_eval.py`

- [ ] **Step 1: 写失败测试**

`train/tests/test_eval.py`：

```python
"""评估：种子区间隔离 + 报告渲染。"""
from train.data_gen.scenario import generate_scenarios
from train.eval.report import render_report
from train.eval.run_eval import EVAL_SEED_BASE, build_eval_scenarios


def test_eval_seed_disjoint():
    sft = generate_scenarios(n=50, seed=1000)
    grpo = generate_scenarios(n=50, seed=100_000)
    ev = build_eval_scenarios(n=50)
    sft_ids = {s.scenario_id for s in sft}
    grpo_ids = {s.scenario_id for s in grpo}
    ev_ids = {s.scenario_id for s in ev}
    assert ev_ids.isdisjoint(sft_ids) and ev_ids.isdisjoint(grpo_ids)
    assert EVAL_SEED_BASE == 200_000


def test_render_report_table():
    results = {
        "base": {"level_acc": 0.35, "r1": 0.1, "r2": 0.1, "r3": 0.05, "tool_ok": 0.3},
        "sft": {"level_acc": 0.82, "r1": 0.35, "r2": 0.28, "r3": 0.2, "tool_ok": 0.9},
        "sft_grpo": {"level_acc": 0.91, "r1": 0.39, "r2": 0.3, "r3": 0.27, "tool_ok": 0.95},
    }
    md = render_report(results)
    assert "| base |" in md and "| sft_grpo |" in md
    assert "0.91" in md and "+0.09" in md  # GRPO 相对 SFT 提升
```

- [ ] **Step 2: 跑红**

Run: `python -m pytest train/tests/test_eval.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现**

`train/eval/__init__.py` 空文件。`train/eval/run_eval.py`：

```python
"""离线评估：base / SFT / SFT+GRPO 三版模型 × 300 条 held-out 场景。

经 vLLM OpenAI 兼容服务批量推理（评估前手动起服务，见 Step 4），
工具回放确定性 mock，指标复用 train/rewards。
"""
import json
from pathlib import Path

from openai import OpenAI

from agent.tools.mock_executor import _mock_search_regulation
from agent.tools.schemas import SearchRegulationParams, build_openai_tools
from train.data_gen.scenario import generate_scenarios
from train.rewards.composite import compute_reward

EVAL_SEED_BASE = 200_000


def build_eval_scenarios(n: int = 300) -> list:
    return generate_scenarios(n=n, seed=EVAL_SEED_BASE)


def _extract_level(text: str) -> str | None:
    from train.data_gen.hermes_format import _LEVEL_MAP
    for pattern, level in _LEVEL_MAP:
        if pattern.search(text):
            return level
    return None


def eval_model(client: OpenAI, model: str, scenarios: list, max_rounds: int = 8) -> dict:
    from train.data_gen.teacher import synthesize_one  # 复用多轮合成循环（回放一致）

    n_correct, n_tool_ok, rewards, parts_all = 0, 0, [], []
    for scn in scenarios:
        trace = synthesize_one(client, model, scn, max_rounds=max_rounds)
        if trace is None:
            rewards.append(0.0)
            continue
        final = trace[-1]["content"] if trace[-1]["role"] == "assistant" else ""
        completion = "".join(m["content"] for m in trace if m["role"] == "assistant")
        rag = _mock_search_regulation(SearchRegulationParams(query=scn.query, top_k=3))["hits"]
        r, parts = compute_reward(completion, scn, rag_hits=rag)
        rewards.append(r)
        parts_all.append(parts)
        # 等级准确率：归一化提取后比较（"Ⅱ级" → "II"），而非子串匹配
        if scn.expected_level and _extract_level(final) == scn.expected_level:
            n_correct += 1
        if any(m["role"] == "tool" for m in trace):
            n_tool_ok += 1
    n = max(len(scenarios), 1)

    def _mean(xs: list) -> float:
        return round(sum(xs) / max(len(xs), 1), 4)

    return {
        "level_acc": round(n_correct / n, 4),
        "tool_ok": round(n_tool_ok / n, 4),
        "reward": _mean(rewards),
        "r1": _mean([p.get("r1", 0.0) for p in parts_all]),
        "r2": _mean([p.get("r2", 0.0) for p in parts_all]),
        "r3": _mean([p.get("r3", 0.0) for p in parts_all]),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8001/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--models", nargs="+", required=True,
                        help="如: Qwen/Qwen2.5-7B-Instruct sft-merged grpo-merged")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--out", type=Path, default=Path("train/eval/outputs/eval_results.json"))
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    scenarios = build_eval_scenarios(args.n)
    results = {m: eval_model(client, m, scenarios) for m in args.models}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] → {args.out}")


if __name__ == "__main__":
    main()
```

`train/eval/report.py`：

```python
"""评估对比报告（Markdown，AC-2/AC-3 证据）。"""

def render_report(results: dict) -> str:
    lines = [
        "# 三模型对比评估报告", "",
        "| 模型 | 等级准确率 | reward | r1 | r2 | r3 | 工具成功率 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in results.items():
        lines.append(
            f"| {name} | {m['level_acc']:.2f} | {m['reward']:.2f} | "
            f"{m['r1']:.2f} | {m['r2']:.2f} | {m['r3']:.2f} | {m['tool_ok']:.2f} |"
        )
    if "sft" in results and "sft_grpo" in results:
        delta = results["sft_grpo"]["level_acc"] - results["sft"]["level_acc"]
        lines += ["", f"**GRPO 相对 SFT 等级准确率提升：{delta:+.2f}**（验收目标 ≥ +0.05）"]
    return "\n".join(lines)
```

- [ ] **Step 4: 跑绿 + 执行评估（M4 验收）**

先起 vLLM 服务（GPU 机/容器内，一次挂一版模型轮换评估）。
**必须**加 `--enable-auto-tool-choice --tool-call-parser hermes`，否则服务端不解析
`<tool_call>` 输出，评估拿不到工具调用（Qwen2.5 对应 hermes 解析器）：

```bash
python -m pytest train/tests/test_eval.py -v   # 2 passed
# base 模型（评估后停掉，依次换 sft/grpo 的 merged 目录）
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8001 \
  --served-model-name Qwen/Qwen2.5-7B-Instruct \
  --enable-auto-tool-choice --tool-call-parser hermes
# 换 train/lora/outputs/sft/merged（--served-model-name sft-merged），
# 再换 train/grpo/outputs/merged（--served-model-name grpo-merged）
python -m train.eval.run_eval --models Qwen/Qwen2.5-7B-Instruct sft-merged grpo-merged --n 300
python -c "import json; from train.eval.report import render_report; \
print(render_report(json.load(open('train/eval/outputs/eval_results.json'))))"
```

Expected（验收 AC-2/AC-3）：base ≤ 0.40；SFT ≥ 0.80；SFT+GRPO 比 SFT ≥ +0.05。
报告写入 `train/eval/outputs/eval_report.md`。

- [ ] **Step 5: 提交**

```bash
git add train/eval/ train/tests/test_eval.py
git commit -m "feat(train): three-model eval with held-out scenarios and markdown report"
```

---

## Phase 5 — 全栈 Docker 化（M5）

### Task 15: 后端/前端 Dockerfile + nginx

**Files:**
- Create: `docker/backend.Dockerfile`
- Create: `docker/frontend.Dockerfile`
- Create: `docker/nginx/default.conf`

- [ ] **Step 1: 后端镜像**

`docker/backend.Dockerfile`：

```dockerfile
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY agent/ ./agent/
COPY backend/ ./backend/
COPY data/raw/regulations/ ./data/raw/regulations/

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --retries=5 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/api/health')"

CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

> 前置检查：`backend/requirements.txt` 若不存在，先生成
> `pip freeze > backend/requirements.txt` 并精简为直接依赖（fastapi/uvicorn/openai/
> qdrant-client/langgraph/langchain/pydantic/structlog/slowapi/httpx/pytest 除外）。

- [ ] **Step 2: 前端镜像 + nginx**

`docker/frontend.Dockerfile`：

```dockerfile
FROM node:20-alpine AS build
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY docker/nginx/default.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
```

`docker/nginx/default.conf`：

```nginx
server {
    listen 80;
    server_name _;

    location /api/ {
        proxy_pass http://backend:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        # SSE 必需
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 600s;
    }

    location / {
        root /usr/share/nginx/html;
        try_files $uri $uri/ /index.html;
    }
}
```

- [ ] **Step 3: 本地构建验证**

```bash
docker build -f docker/backend.Dockerfile -t wateragents-backend:dev .
docker build -f docker/frontend.Dockerfile -t wateragents-frontend:dev .
```

Expected：两镜像构建成功。

- [ ] **Step 4: 提交**

```bash
git add docker/backend.Dockerfile docker/frontend.Dockerfile docker/nginx/default.conf
git commit -m "feat(docker): backend/frontend images with SSE-aware nginx proxy"
```

---

### Task 16: docker-compose 全栈编排 + .env.example

**Files:**
- Create: `docker/docker-compose.yml`
- Create: `docker/.env.example`
- Modify: `backend/app/core/config.py`（仅当缺 `LLM_PROVIDER` 约定时补读，行为不变）

- [ ] **Step 1: compose**

`docker/docker-compose.yml`：

```yaml
name: wateragents

services:
  qdrant:
    image: qdrant/qdrant:v1.12.6
    volumes:
      - qdrant_storage:/qdrant/storage
    healthcheck:
      test: ["CMD", "bash", "-c", ":> /dev/tcp/localhost/6333"]
      interval: 10s
      timeout: 3s
      retries: 10

  vllm:
    image: vllm/vllm-openai:v0.7.3
    profiles: ["local-llm"]   # 用本地微调模型时启用：docker compose --profile local-llm up
    # 注意：--enable-auto-tool-choice --tool-call-parser hermes 必需，
    # 否则服务端不解析 <tool_call>，planner 的 Function Calling 链路失效
    command: >
      --model /models/grpo-merged
      --served-model-name water-agent-fc
      --max-model-len 8192
      --gpu-memory-utilization 0.85
      --enable-auto-tool-choice
      --tool-call-parser hermes
    volumes:
      - ${MODEL_DIR:-../train/grpo/outputs}:/models:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 20

  backend:
    build:
      context: ..
      dockerfile: docker/backend.Dockerfile
    env_file: .env
    environment:
      LLM_API_KEY: ${LLM_API_KEY}
      LLM_BASE_URL: ${LLM_BASE_URL}
      LLM_MODEL: ${LLM_MODEL}
      QDRANT_HOST: qdrant
      QDRANT_PORT: "6333"
      AMAP_API_KEY: ${AMAP_API_KEY:-}
    depends_on:
      qdrant:
        condition: service_healthy
    ports:
      - "8000:8000"

  frontend:
    build:
      context: ..
      dockerfile: docker/frontend.Dockerfile
    depends_on:
      - backend
    ports:
      - "8080:80"

volumes:
  qdrant_storage:
```

`docker/.env.example`：

```bash
# ===== DashScope（默认链路）=====
LLM_API_KEY=sk-your-dashscope-key
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-plus
# ===== 本地微调模型（--profile local-llm 时改用以下两行）=====
# LLM_BASE_URL=http://vllm:8000/v1
# LLM_MODEL=water-agent-fc
AMAP_API_KEY=
MODEL_DIR=../train/grpo/outputs
```

- [ ] **Step 2: 起栈冒烟（验收 AC-4）**

```bash
cd docker
cp .env.example .env   # 填入真实 key
docker compose up --build -d
curl http://localhost:8000/api/health/ready
# 端到端 SSE 冒烟
curl -N -X POST http://localhost:8080/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "吴堡站现在水情怎么样？"}'
```

Expected：健康检查 ready；SSE 流式输出含预警等级字段；前端 `http://localhost:8080` 可对话。
本地模型链路：.env 改指 vllm 两行后，**先起 vllm 等待 healthy（约 2-5 分钟加载权重），
再起其余服务**（compose 不跨 profile 编排 depends_on）：

```bash
docker compose --profile local-llm up -d vllm
docker compose --profile local-llm ps   # 确认 vllm healthy
docker compose --profile local-llm up -d
```

同样执行 SSE 冒烟（FR-R2）。

- [ ] **Step 3: 提交**

```bash
git add docker/docker-compose.yml docker/.env.example
git commit -m "feat(docker): full-stack compose with optional vLLM local-llm profile"
```

---

### Task 17: 训练容器 + README 更新（收尾）

**Files:**
- Create: `docker/train.Dockerfile`
- Modify: `docker/docker-compose.yml`（追加 train 服务）
- Modify: `README.md`（新增训练与部署章节）

- [ ] **Step 1: 训练镜像**

`docker/train.Dockerfile`：

```dockerfile
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

COPY . .
CMD ["bash"]
```

compose 追加（在 `vllm` 服务后）：

```yaml
  train:
    build:
      context: ..
      dockerfile: docker/train.Dockerfile
    profiles: ["train"]
    volumes:
      - ..:/workspace
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: python -m train.lora.train_sft
```

- [ ] **Step 2: README 更新**

在 README「快速开始」后追加两节（内容取自本文档 Phase 1-5 的命令序列）：

```markdown
## 训练（微调对齐）

见 docs/implementation-plan.md 完整计划。简版：
1. 数据集：`python -m train.data_gen.build_dataset --n 5000 --seed 1000 --rpm 30`
2. SFT：`python -m train.lora.train_sft --smoke` 验证后全量 `python -m train.lora.train_sft`
3. 合并：`python -m train.lora.merge`
4. GRPO：`python -m train.grpo.train_grpo`，再次 merge
5. 评估：`python -m train.eval.run_eval --models <base> sft-merged grpo-merged --n 300`

## Docker 部署

见 docker/.env.example 配置密钥后：
- DashScope 链路：`cd docker && docker compose up --build -d`，前端 http://localhost:8080
- 本地微调模型：.env 改指 vllm 后 `docker compose --profile local-llm up -d`
- 训练容器：`docker compose --profile train run --rm train bash`
```

- [ ] **Step 3: 终验（全验收清单）**

| 验收 | 命令 | 标准 |
|---|---|---|
| AC-1 数据集 | 查看 `hermes_fc_v1.jsonl` 行数与统计报告 | 3k-5k 条，accept_rate ≥ 0.70 |
| AC-2 SFT | `train/eval/outputs/eval_results.json` | SFT level_acc ≥ 0.80 |
| AC-3 GRPO | 同上 + reward 曲线 | 比 SFT ≥ +0.05 |
| AC-4 部署 | compose 冒烟 | 前端可对话，SSE 含等级 |
| AC-5 回归 | `python -m pytest backend/tests/ train/tests/ -q; python -m ruff check agent/ backend/app/ train/` | 全绿零告警 |

- [ ] **Step 4: 提交**

```bash
git add docker/train.Dockerfile docker/docker-compose.yml README.md
git commit -m "feat(docker): optional train container and README training/deploy sections"
```

---

## 里程碑总览

| 里程碑 | 任务 | 验收 |
|---|---|---|
| M0 基设 | T1-T2 | train 包可测，Hermes 格式 round-trip |
| M1 数据 | T3-T7 | 3k-5k 条，accept_rate ≥ 70%，三道过滤单测全绿 |
| M2 SFT | T8-T10 | smoke 5 步跑通，全量 ≤ 12h，merged 产出 |
| M3 GRPO | T11-T13 | 奖励单测全绿，r1/r2/r3 曲线上升 |
| M4 评估 | T14 | base/SFT/GRPO 对比报告达标 |
| M5 部署 | T15-T17 | compose 一键起栈，双链路冒烟通过，回归全绿 |

**依赖顺序**：T1→T2→T3→T4→T5→T6→T7（数据）→T8→T9→T10（SFT）→T11→T12→T13（GRPO）→T14→T15→T16→T17。
T11-T12（奖励/prompts）与 T8-T10（SFT 训练机时）可并行。
