"""阶段 F 工具独立验证脚本。

测试 3 个新接入的真实工具：
1. get_hydrology：qqjjsj.com 实时水文数据
2. predict_runoff：SCS-CN 降雨-径流模型
3. get_weather：高德天气 API（若未配置 key 应回 RuntimeError 降级到 mock）
"""
import os
import sys
import json
from pathlib import Path

# 将项目根目录（backend 的父目录）+ backend 目录加入 sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
_BACKEND_ROOT = str(Path(__file__).resolve().parent)
for p in [_PROJECT_ROOT, _BACKEND_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)


def test_hydrology():
    """测试水文爬虫。"""
    print("\n" + "=" * 60)
    print("[1] get_hydrology 真实实现测试")
    print("=" * 60)
    from agent.data.hydrology import fetch_hydrology

    for station in ["吴堡", "龙门"]:
        try:
            result = fetch_hydrology(station=station, metric="both")
            print(f"\n--- {station} ---")
            print(f"  水位: {result.get('water_level_m')} m")
            print(f"  流量: {result.get('flow_m3_s')} m³/s")
            print(f"  警戒水位: {result.get('warning_level_m')} m")
            print(f"  保证水位: {result.get('guaranteed_level_m')} m")
            print(f"  超警幅度: {result.get('above_warning_m')} m")
            print(f"  观测时间: {result.get('observed_time')}")
            print(f"  数据来源: {result.get('source')}")
        except Exception as e:
            print(f"  {station} 失败: {e}")


def test_runoff():
    """测试 SCS-CN 径流模型。"""
    print("\n" + "=" * 60)
    print("[2] predict_runoff SCS-CN 模型测试")
    print("=" * 60)
    from agent.hydrology.scs_cn import predict_runoff_scs, compute_runoff_depth

    # 单元测试：SCS-CN 公式
    print("\n--- SCS-CN 公式验证（CN=75, λ=0.2）---")
    for p in [0, 5, 10, 20, 50, 100, 200]:
        q = compute_runoff_depth(p, 75)
        print(f"  P={p:>4}mm → Q={q:>6.2f}mm  (径流系数={q/p if p > 0 else 0:.2f})")

    # 完整预测
    print("\n--- 吴堡站 24h 预测（降雨 50mm）---")
    result = predict_runoff_scs(station="吴堡", rainfall_mm=50, lead_time_hours=24)
    print(f"  总降雨: {result['total_rainfall_mm']} mm")
    print(f"  径流深: {result['runoff_depth_mm']} mm")
    print(f"  CN 值: {result['curve_number']}")
    print(f"  流域面积: {result['basin_area_km2']} km²")
    print(f"  汇流时间: {result['tc_hours']} h")
    print(f"  基流: {result['base_flow_m3_s']} m³/s")
    print(f"  洪峰流量: {result['peak_flow_m3_s']} m³/s")
    print(f"  洪峰时间: {result['peak_time']}")
    print(f"  过程线点数: {len(result['series'])}")
    print(f"  数据来源: {result['source']}")
    print(f"  前 3 点流量: {[s['predicted_flow_m3_s'] for s in result['series'][:3]]}")
    print(f"  洪峰附近: {[s['predicted_flow_m3_s'] for s in result['series'][3:6]]}")


def test_weather():
    """测试高德天气 API（无 key 时应抛 RuntimeError）。"""
    print("\n" + "=" * 60)
    print("[3] get_weather 高德 API 测试")
    print("=" * 60)
    from app.core.config import get_settings

    settings = get_settings()
    print(f"\n  AMAP_API_KEY: {'已配置' if settings.AMAP_API_KEY else '未配置（将降级到 mock）'}")
    print(f"  AMAP_CITY_CODE: {settings.AMAP_CITY_CODE}")

    from agent.data.weather import fetch_weather, _location_to_adcode

    print(f"\n--- adcode 映射测试 ---")
    for loc in ["吕梁市", "吴堡", "龙门", "北京", "未知地点"]:
        code = _location_to_adcode(loc)
        print(f"  {loc} → {code}")

    print("\n--- fetch_weather 调用 ---")
    try:
        result = fetch_weather(location="吴堡", hours=6)
        print(f"  定位: {result.get('location')} (adcode={result.get('adcode')})")
        print(f"  总降雨: {result.get('total_rainfall_mm')} mm")
        print(f"  最大小时降雨: {result.get('max_hourly_rainfall_mm')} mm")
        print(f"  当前天气: {result.get('current', {}).get('weather')}")
        print(f"  当前温度: {result.get('current', {}).get('temperature')} °C")
        print(f"  数据来源: {result.get('source')}")
        print(f"  序列点数: {len(result.get('series', []))}")
    except RuntimeError as e:
        print(f"  预期失败（未配置 key）: {e}")
        print("  → 上层会降级到 mock_get_weather，符合设计")


def test_executor_integration():
    """测试 execute_tool 入口是否正确路由到真实实现。"""
    print("\n" + "=" * 60)
    print("[4] execute_tool 集成测试（统一入口）")
    print("=" * 60)
    from agent.tools.mock_executor import execute_tool
    from agent.tools.real_executor import is_tool_real_implemented

    print("\n--- 工具真实实现注册情况 ---")
    for name in ["get_weather", "get_hydrology", "predict_runoff", "search_regulation", "query_gis_terrain", "generate_plan"]:
        print(f"  {name}: {'real' if is_tool_real_implemented(name) else 'mock'}")

    print("\n--- get_hydrology 通过 execute_tool 入口 ---")
    try:
        result = execute_tool("get_hydrology", {"station": "吴堡", "metric": "both"})
        print(f"  source: {result.get('source')}")
        print(f"  water_level_m: {result.get('water_level_m')}")
        print(f"  flow_m3_s: {result.get('flow_m3_s')}")
    except Exception as e:
        print(f"  失败: {e}")

    print("\n--- predict_runoff 通过 execute_tool 入口（默认降雨）---")
    try:
        result = execute_tool("predict_runoff", {"station": "吴堡", "lead_time_hours": 24})
        print(f"  source: {result.get('source')}")
        print(f"  peak_flow_m3_s: {result.get('peak_flow_m3_s')}")
        print(f"  runoff_depth_mm: {result.get('runoff_depth_mm')}")
    except Exception as e:
        print(f"  失败: {e}")

    print("\n--- get_weather 通过 execute_tool 入口（未配 key 应回退 mock）---")
    try:
        result = execute_tool("get_weather", {"location": "吴堡", "hours": 6})
        print(f"  source: {result.get('source')} (预期 mock)")
        print(f"  total_rainfall_mm: {result.get('total_rainfall_mm')}")
    except Exception as e:
        print(f"  失败: {e}")


if __name__ == "__main__":
    test_hydrology()
    test_runoff()
    test_weather()
    test_executor_integration()
    print("\n" + "=" * 60)
    print("阶段 F 工具独立测试完成")
    print("=" * 60)
