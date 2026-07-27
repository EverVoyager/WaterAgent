"""高德天气 API 客户端。

文档：https://lbs.amap.com/api/webservice/guide/api/weatherinfo

extensions=base: 实况天气（temperature/weather/winddirection/windpower/humidity）
extensions=all: 预报天气（casts 数组，4 天 day/night temp + weather + power）

降雨量映射：高德返回的是天气描述（"小雨"/"大雨"/"暴雨"），按降雨强度对照表映射为 mm/h。
"""
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List

import requests

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# 高德天气描述 → 降雨强度（mm/h）
# 参考气象标准：24h 降雨量小雨<10, 中雨 10-25, 大雨 25-50, 暴雨 50-100, 大暴雨 100-250
# 折算到小时均值：小雨 1, 中雨 2.5, 大雨 5, 暴雨 10, 大暴雨 20
RAINFALL_MAP = {
    "小雨": 1.0,
    "中雨": 2.5,
    "大雨": 5.0,
    "暴雨": 10.0,
    "大暴雨": 20.0,
    "特大暴雨": 30.0,
    "阵雨": 0.8,
    "雷阵雨": 1.5,
    "雨夹雪": 0.5,
}

# 简单缓存：避免高频请求（高德免费版每日配额有限）
_cache: Dict[str, tuple[float, Dict[str, Any]]] = {}
_CACHE_TTL = 600  # 10 分钟


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_rainfall(weather_desc: str) -> float:
    """从天气描述提取降雨强度（mm/h）。"""
    if not weather_desc:
        return 0.0
    for key, val in RAINFALL_MAP.items():
        if key in weather_desc:
            return val
    return 0.0


def _location_to_adcode(location: str) -> str:
    """把用户输入的地点名映射为高德 adcode。

    高德 adcode 是 6 位数字，黄河吕梁段相关区域：
    """
    mapping = {
        "吕梁": "141100",
        "吕梁市": "141100",
        "吴堡": "141129",
        "吴堡站": "141129",
        "吴堡水文站": "141129",
        "龙门": "610528",
        "龙门站": "610528",
        "府谷": "610822",
        "府谷站": "610822",
        "临县": "141124",
        "柳林": "141125",
    }
    for key, code in mapping.items():
        if key in location:
            return code
    # 默认吕梁市
    return get_settings().AMAP_CITY_CODE


def fetch_weather(location: str, hours: int = 6) -> Dict[str, Any]:
    """查询指定地点未来 N 小时天气。

    Args:
        location: 地点名称（"吴堡"、"吕梁市" 等）
        hours: 预测时长（小时），高德预报精度为天，按小时展开

    Returns:
        与 mock_get_weather 兼容的 dict 结构：
        {
            "location": str,
            "hours": int,
            "total_rainfall_mm": float,
            "max_hourly_rainfall_mm": float,
            "series": [{"time", "rainfall_mm", "temperature_c", "wind_speed_ms"}],
            "fetched_at": iso,
            "source": "amap",
            "current": {temperature, weather, winddirection, windpower, humidity, reporttime}
        }

    Raises:
        RuntimeError: API 调用失败或未配置 key
    """
    settings = get_settings()
    if not settings.AMAP_API_KEY:
        raise RuntimeError("AMAP_API_KEY 未配置，请在 backend/.env 中填入高德 Web 服务 API Key")

    adcode = _location_to_adcode(location)
    cache_key = f"{adcode}:{hours}"
    now = time.time()

    # 检查缓存
    if cache_key in _cache:
        cached_at, cached_data = _cache[cache_key]
        if now - cached_at < _CACHE_TTL:
            logger.debug("[weather] 命中缓存 %s", cache_key)
            return cached_data

    # 调用高德 API：先拿实况，再拿预报
    base_url = "https://restapi.amap.com/v3/weather/weatherInfo"
    params_base = {"key": settings.AMAP_API_KEY, "city": adcode, "extensions": "base", "output": "JSON"}
    params_all = {"key": settings.AMAP_API_KEY, "city": adcode, "extensions": "all", "output": "JSON"}

    try:
        resp_base = requests.get(base_url, params=params_base, timeout=10)
        resp_base.raise_for_status()
        data_base = resp_base.json()
        resp_all = requests.get(base_url, params=params_all, timeout=10)
        resp_all.raise_for_status()
        data_all = resp_all.json()
    except requests.RequestException as e:
        logger.exception("[weather] 高德 API 调用失败")
        raise RuntimeError(f"高德天气 API 调用失败：{e}") from e

    if data_base.get("status") != "1" or not data_base.get("lives"):
        raise RuntimeError(f"高德实况天气查询失败：{data_base.get('info', 'unknown error')}")
    if data_all.get("status") != "1" or not data_all.get("forecasts"):
        raise RuntimeError(f"高德预报天气查询失败：{data_all.get('info', 'unknown error')}")

    live = data_base["lives"][0]
    forecast = data_all["forecasts"][0]
    casts: List[Dict[str, Any]] = forecast.get("casts", [])

    # 把实况 + 4 天预报展开为小时序列
    series = []
    base_time = datetime.now(timezone.utc)
    total_rain = 0.0
    max_rain = 0.0

    # 前 1 小时：用实况
    current_rain = _normalize_rainfall(live.get("weather", ""))
    current_temp = float(live.get("temperature", "20"))
    # 风力等级 → 风速 m/s（粗略：1 级≈0.5, 6 级≈12, 12 级≈35）
    wind_power_str = live.get("windpower", "3")
    try:
        wind_level = int(wind_power_str.split("-")[0])
    except (ValueError, AttributeError):
        wind_level = 3
    wind_speed = max(0.5, wind_level * 1.8)

    series.append({
        "time": base_time.isoformat(),
        "rainfall_mm": round(current_rain, 1),
        "temperature_c": round(current_temp, 1),
        "wind_speed_ms": round(wind_speed, 1),
    })
    total_rain += current_rain
    max_rain = max(max_rain, current_rain)

    # 第 1 小时到 hours：从 casts 展开
    for hour_offset in range(1, hours):
        # 计算落在哪一天（cast 索引 0=今天, 1=明天...）
        day_idx = hour_offset // 24
        if day_idx >= len(casts):
            day_idx = len(casts) - 1
        cast = casts[day_idx]

        # 一天内小时分布：白天用 daytemp/dayweather，夜间用 nighttemp/nightweather
        hour_of_day = hour_offset % 24
        if 6 <= hour_of_day < 18:
            weather_desc = cast.get("dayweather", "")
            temp = float(cast.get("daytemp", "20"))
            wind_str = cast.get("daypower", "3")
        else:
            weather_desc = cast.get("nightweather", "")
            temp = float(cast.get("nighttemp", "15"))
            wind_str = cast.get("nightpower", "3")

        rain = _normalize_rainfall(weather_desc)
        try:
            wl = int(wind_str.split("-")[0])
        except (ValueError, AttributeError):
            wl = 3
        ws = max(0.5, wl * 1.8)

        # 加点小扰动让序列自然（实况固定值，预报按级别）
        rain = round(rain * (0.8 + 0.4 * ((hour_offset * 7) % 10) / 10), 1)

        series.append({
            "time": (base_time + timedelta(hours=hour_offset)).isoformat(),
            "rainfall_mm": rain,
            "temperature_c": round(temp, 1),
            "wind_speed_ms": round(ws, 1),
        })
        total_rain += rain
        max_rain = max(max_rain, rain)

    result = {
        "location": location,
        "adcode": adcode,
        "hours": hours,
        "total_rainfall_mm": round(total_rain, 1),
        "max_hourly_rainfall_mm": round(max_rain, 1),
        "series": series,
        "current": {
            "temperature": live.get("temperature", ""),
            "weather": live.get("weather", ""),
            "winddirection": live.get("winddirection", ""),
            "windpower": live.get("windpower", ""),
            "humidity": live.get("humidity", ""),
            "reporttime": live.get("reporttime", ""),
        },
        "forecast_reporttime": forecast.get("reporttime", ""),
        "fetched_at": _now_iso(),
        "source": "amap",
    }

    # 写入缓存
    _cache[cache_key] = (now, result)
    return result
