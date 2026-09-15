"""预警阈值档案：带版本的唯一配置源（外置化改造）。

背景（2026-09-14 外置化改造）：阈值原为三处硬编码——utils 常量、
synthesizer 提示词 import 时烙印、scenario 造数据档位复制。生产环境中
官方核定值修订时需要"改一处配置全链路生效 + 历史可追溯"，故收敛为本档案：

- config/thresholds.json：唯一数值来源（version + effective_date + 阈值 +
  站点表 + 造数据参数）。环境变量 WATERAGENTS_THRESHOLDS_FILE 可覆盖路径；
- get_thresholds()：进程内缓存的档案视图，规则引擎/提示词渲染运行时读取；
- reload_thresholds()：换配置后原地同步 utils.WARNING_THRESHOLDS——
  引擎（synthesizer.compute_warning_level）、事实门（reflection）、默认
  系统提示（llm.get_default_system_prompt）均按引用动态读该 dict，
  原地 clear+update 后全部即时生效，无需重启；
- 训练/评估数据生成（scenario.py）在 import 时取进程快照：离线数据生成
  要求单进程内绝对确定性，运行中换配置不追溯生效（重跑进程即用新档）。

修订流程（变更管理）：新版本入库（version 递增，不覆盖旧文件历史）→
边界参数化测试重跑 → 评估重放 + 重建基线（run_eval 的 config 记录
threshold_version，基线组合一致性检查据此拦截跨版本的门禁比较）。
"""
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "thresholds.json"
ENV_CONFIG_PATH = "WATERAGENTS_THRESHOLDS_FILE"

# 未知站点的兜底档案（与旧 mock 默认值一致：基准 500m / 警戒 +2.0 / 保证 +3.5）
_DEFAULT_STATION = {
    "base_level_m": 500.0,
    "base_flow_m3_s": 1000.0,
    "warning_level_m": 502.0,
    "guaranteed_level_m": 503.5,
}

_LEVELS = ("I", "II", "III", "IV")


@dataclass(frozen=True)
class ThresholdProfile:
    """一份阈值档案（不可变快照）。"""

    version: str
    effective_date: str
    source_note: str = ""
    flow_level1: float = 5000.0
    flow_level2: float = 3000.0
    flow_level3: float = 2000.0
    rain_level1: float = 100.0
    rain_level2: float = 50.0
    stations: dict = field(default_factory=dict)
    scenario: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    def station(self, name: str) -> dict:
        """站点档案（未知站点回退默认值）。"""
        return dict(self.stations.get(name) or _DEFAULT_STATION)

    def flow_range_for(self, level: str) -> tuple[float, float]:
        """造数据档位区间（训练/评估场景生成用，由阈值推导消灭复制）。

        I 档上限 = f1 × tier1_cap_ratio（合理性封顶，防极值样本）；
        II/III 档上限 = 下一档阈值 - 1（防随机扰动跨档改变等级真值，
        与 _make_overrides 的 peak 钳制同一纪律）；IV 档下限 = 基流下限。
        """
        floor = float(self.scenario.get("flow_floor_m3_s", 500.0))
        cap_ratio = float(self.scenario.get("tier1_cap_ratio", 1.3))
        ranges = {
            "I": (self.flow_level1, round(self.flow_level1 * cap_ratio, 1)),
            "II": (self.flow_level2, self.flow_level1 - 1.0),
            "III": (self.flow_level3, self.flow_level2 - 1.0),
            "IV": (floor, self.flow_level3 - 1.0),
        }
        if level not in ranges:
            raise KeyError(f"未知等级: {level}（可选 {_LEVELS}）")
        return ranges[level]

    def rain_for_level(self, level: str) -> float:
        """造数据代表降雨（每档取安全落在区间内的代表值，远离边界）。"""
        rain_map = self.scenario.get("rain_by_level_mm", {})
        if level not in rain_map:
            raise KeyError(f"scenario.rain_by_level_mm 缺少等级 {level}")
        return float(rain_map[level])

    def to_legacy(self) -> dict[str, float]:
        """旧 utils.WARNING_THRESHOLDS 形状（向后兼容视图）。"""
        return {
            "flow_level1": self.flow_level1,
            "flow_level2": self.flow_level2,
            "flow_level3": self.flow_level3,
            "rain_level1": self.rain_level1,
            "rain_level2": self.rain_level2,
        }


def _validate(raw: dict) -> None:
    """配置合法性（防手改出错：阈值倒挂 / 站点警戒超保证直接拒绝加载）。"""
    flow = raw.get("flow_thresholds_m3_s", {})
    rain = raw.get("rain_thresholds_mm", {})
    f1, f2, f3 = (float(flow[k]) for k in ("level1", "level2", "level3"))
    r1, r2 = (float(rain[k]) for k in ("level1", "level2"))
    if not (f1 > f2 > f3 > 0):
        raise ValueError(f"流量阈值必须严格递减且为正：{f1}/{f2}/{f3}")
    if not (r1 > r2 > 0):
        raise ValueError(f"降雨阈值必须严格递减且为正：{r1}/{r2}")
    for name, st in raw.get("stations", {}).items():
        if float(st["warning_level_m"]) >= float(st["guaranteed_level_m"]):
            raise ValueError(f"站点 {name} 警戒水位须低于保证水位")
        if float(st["base_level_m"]) >= float(st["warning_level_m"]):
            raise ValueError(f"站点 {name} 基准水位须低于警戒水位")


def _build_profile(raw: dict) -> ThresholdProfile:
    flow = raw["flow_thresholds_m3_s"]
    rain = raw["rain_thresholds_mm"]
    # 保留 JSON 原生数值类型（int 5000 不转 float）——提示词渲染与旧硬编码
    # 逐字节一致（"5000m³/s" 而非 "5000.0m³/s"）；合法性校验见 _validate
    return ThresholdProfile(
        version=str(raw["version"]),
        effective_date=str(raw.get("effective_date", "")),
        source_note=str(raw.get("source_note", "")),
        flow_level1=flow["level1"],
        flow_level2=flow["level2"],
        flow_level3=flow["level3"],
        rain_level1=rain["level1"],
        rain_level2=rain["level2"],
        stations={k: dict(v) for k, v in raw.get("stations", {}).items()},
        scenario=dict(raw.get("scenario", {})),
        raw=raw,
    )


def load_thresholds(path: str | Path | None = None) -> ThresholdProfile:
    """加载并校验一份阈值档案（不写缓存；缓存走 get_thresholds）。"""
    p = Path(path or os.environ.get(ENV_CONFIG_PATH) or DEFAULT_CONFIG_PATH)
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not raw.get("version"):
        raise ValueError(f"{p} 缺少 version 字段（版本追溯必需）")
    _validate(raw)
    profile = _build_profile(raw)
    logger.info("[thresholds] 加载 %s（version=%s, 生效 %s）",
                p.name, profile.version, profile.effective_date)
    return profile


_lock = threading.Lock()
_cache: ThresholdProfile | None = None


def get_thresholds() -> ThresholdProfile:
    """当前档案（进程内缓存，首次访问加载）。"""
    global _cache
    if _cache is None:
        with _lock:
            if _cache is None:
                _cache = load_thresholds()
    return _cache


def reload_thresholds(path: str | Path | None = None) -> ThresholdProfile:
    """换档案后热生效：刷新缓存并原地同步 utils.WARNING_THRESHOLDS。

    引擎/事实门/默认系统提示均以 from agent.utils import WARNING_THRESHOLDS
    持有同一 dict 引用且调用时读值——原地 clear+update 让它们即时跟随，
    无需重启进程。提示词渲染走 render_synthesizer_prompt 的版本缓存，
    新版本自动重渲染。
    """
    global _cache
    profile = load_thresholds(path)
    with _lock:
        _cache = profile
        from agent.utils import WARNING_THRESHOLDS  # noqa: PLC0415 —— 延迟导入避免环
        WARNING_THRESHOLDS.clear()
        WARNING_THRESHOLDS.update(profile.to_legacy())
    logger.info("[thresholds] 已切换到 version=%s 并同步旧常量视图", profile.version)
    return profile


def threshold_version() -> str:
    """当前档案版本号（评估 config / 基线可比性检查用）。"""
    return get_thresholds().version


if __name__ == "__main__":
    p = get_thresholds()
    print(f"版本: {p.version}（生效 {p.effective_date}）")
    print(f"流量阈值: Ⅰ≥{p.flow_level1} / Ⅱ≥{p.flow_level2} / Ⅲ≥{p.flow_level3} m³/s")
    print(f"降雨阈值: Ⅰ>{p.rain_level1} / Ⅱ≥{p.rain_level2} mm/24h")
    for name, st in p.stations.items():
        print(f"  {name}: 基准 {st['base_level_m']}m 警戒 {st['warning_level_m']}m "
              f"保证 {st['guaranteed_level_m']}m")
    for lv in _LEVELS:
        print(f"  造数据档位 {lv}: 流量 {p.flow_range_for(lv)} m³/s，"
              f"降雨 {p.rain_for_level(lv)} mm")
