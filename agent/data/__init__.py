"""阶段 F 数据源模块：实时天气与水文数据接入。

子模块：
- weather: 高德天气 API 客户端
- hydrology: 水文站实时数据爬虫（qqjjsj.com）
"""
from agent.data.weather import fetch_weather
from agent.data.hydrology import fetch_hydrology

__all__ = ["fetch_weather", "fetch_hydrology"]
