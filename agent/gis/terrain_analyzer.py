"""地形分析器。

实现三种 GIS 分析算法（基于 SRTM DEM + rasterio + numpy）：
1. slope（坡度）：用 Horn 算法计算坡度，统计坡度分布与高风险区面积
2. channel_cross_section（河床断面）：沿河道横断面提取高程序列，计算河宽/最大水深/平均水深
3. inundation（淹没范围）：给定水位高程，计算淹没区面积与受影响村庄数

所有结果带真实地理参考（米、平方千米、经纬度）。
"""
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agent.gis.dem_loader import DEMDataset

logger = logging.getLogger(__name__)

# 黄河吕梁段河道近似经纬度（吴堡水文站附近，从北向南流）
# 实际工程中应从 HydroSHEDS 等数据源获取，这里用关键点连线近似
RIVER_CENTERLINE = [
    (110.74, 37.75),  # 北端
    (110.72, 37.65),
    (110.75, 37.55),  # 吴堡水文站附近
    (110.76, 37.45),  # 南端
]


@dataclass
class TerrainAnalysisResult:
    """地形分析结果，可直接序列化为 Agent 工具返回。"""

    slope: dict[str, Any] = field(default_factory=dict)
    channel_cross_section: dict[str, Any] = field(default_factory=dict)
    inundation: dict[str, Any] = field(default_factory=dict)
    bbox: tuple[float, float, float, float] = (0, 0, 0, 0)
    crs: str = "EPSG:4326"
    analyzed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "slope": self.slope,
            "channel_cross_section": self.channel_cross_section,
            "inundation": self.inundation,
            "bbox": list(self.bbox),
            "crs": self.crs,
            "analyzed_at": self.analyzed_at,
        }


class TerrainAnalyzer:
    """地形分析器：基于 DEM 计算坡度/断面/淹没范围。"""

    # 坡度阈值（度），>15° 视为高地质灾害风险区（黄土高原标准）
    HIGH_RISK_SLOPE_THRESHOLD = 15.0

    # 黄河吴堡段多年平均水位高程（米，大沽高程系近似值）
    RIVER_BASE_LEVEL_M = 640.0

    def __init__(self, dem: DEMDataset):
        self.dem = dem
        self._slope_cache: np.ndarray = None

    # ====== 坡度分析 ======

    def compute_slope(self) -> np.ndarray:
        """用 Horn 算法计算坡度（度）。

        Horn 算法是 ArcGIS 标准的坡度计算方法，使用 3x3 窗口加权差分。
        """
        if self._slope_cache is not None:
            return self._slope_cache

        elev = self.dem.elevation
        # 处理 NaN：用均值填充便于差分计算
        valid_mask = ~np.isnan(elev)
        if not valid_mask.any():
            self._slope_cache = np.zeros_like(elev)
            return self._slope_cache
        filled = np.where(valid_mask, elev, np.nanmean(elev))

        res_x_m, res_y_m = self.dem.resolution_m
        # 8 邻域加权差分（Horn 算法）
        # dz/dx = ((c+2f+i) - (a+2d+g)) / (8 * cellsize_x)
        # dz/dy = ((g+2h+i) - (a+2b+c)) / (8 * cellsize_y)
        # slope = atan(sqrt(dz/dx^2 + dz/dy^2))
        dz_dx = (
            (filled[2:, :-2] + 2 * filled[2:, 1:-1] + filled[2:, 2:])
            - (filled[:-2, :-2] + 2 * filled[:-2, 1:-1] + filled[:-2, 2:])
        ) / (8 * res_x_m)
        dz_dy = (
            (filled[:-2, 2:] + 2 * filled[1:-1, 2:] + filled[2:, 2:])
            - (filled[:-2, :-2] + 2 * filled[1:-1, :-2] + filled[2:, :-2])
        ) / (8 * res_y_m)

        slope_inner = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
        # 边缘像素用 0 填充
        slope = np.zeros_like(elev, dtype=np.float32)
        slope[1:-1, 1:-1] = slope_inner
        # 边缘设为 NaN（无效）
        slope[0, :] = np.nan
        slope[-1, :] = np.nan
        slope[:, 0] = np.nan
        slope[:, -1] = np.nan
        # 原始 DEM 的 NaN 区域坡度也设为 NaN
        slope[~valid_mask] = np.nan

        self._slope_cache = slope
        return slope

    def analyze_slope(self) -> dict[str, Any]:
        """坡度统计分析：均值/最大值/高风险区面积。"""
        slope = self.compute_slope()
        valid_slope = slope[~np.isnan(slope)]
        if valid_slope.size == 0:
            return {"error": "无有效坡度数据"}

        res_x_m, res_y_m = self.dem.resolution_m
        pixel_area_km2 = (res_x_m * res_y_m) / 1e6

        high_risk_mask = slope > self.HIGH_RISK_SLOPE_THRESHOLD
        high_risk_pixels = int(np.nansum(high_risk_mask))

        # 坡度分级统计（百分比）
        levels = {
            "0-5° (平缓)": float(np.nanmean((slope >= 0) & (slope < 5))),
            "5-15° (缓坡)": float(np.nanmean((slope >= 5) & (slope < 15))),
            "15-25° (斜坡)": float(np.nanmean((slope >= 15) & (slope < 25))),
            ">25° (陡坡)": float(np.nanmean(slope >= 25)),
        }

        return {
            "mean_degree": round(float(np.nanmean(valid_slope)), 2),
            "max_degree": round(float(np.nanmax(valid_slope)), 2),
            "median_degree": round(float(np.nanmedian(valid_slope)), 2),
            "high_risk_area_km2": round(high_risk_pixels * pixel_area_km2, 2),
            "high_risk_threshold_degree": self.HIGH_RISK_SLOPE_THRESHOLD,
            "slope_distribution": {k: round(v * 100, 1) for k, v in levels.items()},
        }

    # ====== 河床断面分析 ======

    def analyze_channel_cross_section(
        self,
        river_level_m: float = None,
    ) -> dict[str, Any]:
        """沿河道横断面提取高程，计算河宽、最大水深、平均水深。

        横断面定义：在河道中心点附近，沿东西方向（垂直于南北流向）取一条切线。
        """
        if river_level_m is None:
            river_level_m = self.RIVER_BASE_LEVEL_M

        # 取河道中心点附近的横断面（吴堡水文站附近）
        center_lon, center_lat = RIVER_CENTERLINE[2]  # (110.75, 37.55)

        # 在 DEM 中找到最接近中心点的行
        transform = self.dem.transform
        inv_transform = ~transform
        # 中心点的像素坐标
        col_center, row_center = inv_transform * (center_lon, center_lat)
        row_center = int(round(row_center))
        col_center = int(round(col_center))

        if not (0 <= row_center < self.dem.shape[0] and 0 <= col_center < self.dem.shape[1]):
            return {"error": f"河道中心点 ({center_lon},{center_lat}) 不在 DEM 范围内"}

        # 沿东西方向取横断面（整行）
        profile = self.dem.elevation[row_center, :].copy()
        # 找到河道位置：高程最接近 river_level_m 的连续区域
        # 简化：用阈值 river_level_m + 5m 作为河道边界
        channel_threshold = river_level_m + 5.0
        channel_mask = profile < channel_threshold

        if not channel_mask.any():
            return {
                "river_level_m": river_level_m,
                "note": "未检测到明显河道，可能 DEM 数据或水位高程不匹配",
            }

        # 找到连续的河道区域
        channel_indices = np.where(channel_mask)[0]
        # 取最长的连续段
        gaps = np.diff(channel_indices)
        breaks = np.where(gaps > 5)[0]  # 允许 5 像素间隙
        if len(breaks) > 0:
            # 取最长段
            segments = np.split(channel_indices, breaks + 1)
            channel_indices = max(segments, key=len)

        if len(channel_indices) < 3:
            return {
                "river_level_m": river_level_m,
                "note": "河道区域过窄，无法分析断面",
            }

        channel_profile = profile[channel_indices]
        # 河宽：用像素数 × 分辨率（经度方向米）
        res_x_m, _ = self.dem.resolution_m
        width_m = (len(channel_indices) - 1) * res_x_m

        # 水深：river_level_m - 河床高程
        depths = river_level_m - channel_profile
        depths[depths < 0] = 0  # 高于水面的不算水深

        # 经纬度范围
        lon_start = transform * (channel_indices[0], row_center)
        lon_end = transform * (channel_indices[-1], row_center)

        return {
            "river_level_m": round(float(river_level_m), 2),
            "center_lon": center_lon,
            "center_lat": center_lat,
            "width_m": round(float(width_m), 1),
            "max_depth_m": round(float(np.max(depths)), 2),
            "avg_depth_m": round(float(np.mean(depths)), 2),
            "min_bed_elevation_m": round(float(np.min(channel_profile)), 2),
            "max_bed_elevation_m": round(float(np.max(channel_profile)), 2),
            "sample_points": int(len(channel_indices)),
            "lon_range": [round(float(lon_start[0]), 4), round(float(lon_end[0]), 4)],
        }

    # ====== 淹没范围分析 ======

    def analyze_inundation(
        self,
        flood_level_m: float = None,
    ) -> dict[str, Any]:
        """给定洪水位高程，计算淹没范围。

        简化模型：所有高程 < flood_level_m 的区域视为淹没区。
        实际工程应考虑水动力模型，这里用 DEM 阈值法做初步评估。

        Args:
            flood_level_m: 洪水位高程（米）。默认比河道基准水位高 3m（相当于Ⅰ级预警）
        """
        if flood_level_m is None:
            flood_level_m = self.RIVER_BASE_LEVEL_M + 3.0

        elev = self.dem.elevation
        valid_mask = ~np.isnan(elev)
        # 淹没区：高程低于洪水位
        inundation_mask = (elev < flood_level_m) & valid_mask

        res_x_m, res_y_m = self.dem.resolution_m
        pixel_area_km2 = (res_x_m * res_y_m) / 1e6
        inundated_pixels = int(inundation_mask.sum())
        inundated_area_km2 = inundated_pixels * pixel_area_km2

        # 估算受影响村庄数：每 5 km² 一个村庄（黄土高原乡镇密度近似）
        affected_villages = int(inundated_area_km2 / 5)

        # 淹没区中心经纬度（用于后续在地图上标注）
        if inundated_pixels > 0:
            rows, cols = np.where(inundation_mask)
            inv_transform = ~self.dem.transform
            # 取若干采样点
            sample_indices = np.linspace(0, len(rows) - 1, min(5, len(rows)), dtype=int)
            sample_points = []
            for idx in sample_indices:
                lon, lat = inv_transform * (cols[idx], rows[idx])
                sample_points.append({
                    "lon": round(float(lon), 4),
                    "lat": round(float(lat), 4),
                    "elevation_m": round(float(elev[rows[idx], cols[idx]]), 2),
                })
        else:
            sample_points = []

        return {
            "flood_level_m": round(float(flood_level_m), 2),
            "inundated_area_km2": round(inundated_area_km2, 2),
            "affected_villages": affected_villages,
            "inundated_pixels": inundated_pixels,
            "total_valid_pixels": int(valid_mask.sum()),
            "inundation_ratio": round(inundated_pixels / max(int(valid_mask.sum()), 1) * 100, 2),
            "sample_points": sample_points,
        }

    # ====== 综合分析 ======

    def analyze_all(
        self,
        analysis_type: str = "all",
        river_level_m: float = None,
        flood_level_m: float = None,
    ) -> TerrainAnalysisResult:
        """根据 analysis_type 调用相应分析。"""
        from datetime import datetime, timezone
        result = TerrainAnalysisResult(
            bbox=(
                self.dem.bounds.left,
                self.dem.bounds.bottom,
                self.dem.bounds.right,
                self.dem.bounds.top,
            ),
            crs=self.dem.crs,
            analyzed_at=datetime.now(timezone.utc).isoformat(),
        )

        if analysis_type in ("slope", "all"):
            logger.info("[terrain] 计算坡度 ...")
            result.slope = self.analyze_slope()

        if analysis_type in ("channel_cross_section", "all"):
            logger.info("[terrain] 提取河床断面 ...")
            result.channel_cross_section = self.analyze_channel_cross_section(
                river_level_m=river_level_m
            )

        if analysis_type in ("inundation", "all"):
            logger.info("[terrain] 计算淹没范围 ...")
            result.inundation = self.analyze_inundation(flood_level_m=flood_level_m)

        return result
