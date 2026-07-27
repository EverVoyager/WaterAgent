"""DEM 数据加载器。

加载 GeoTIFF 格式的 SRTM DEM 数据，支持按 bbox 裁剪。
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.coords import BoundingBox
from rasterio.windows import from_bounds

logger = logging.getLogger(__name__)

# 研究区 GeoTIFF 默认路径（由 build_terrain_data.py 生成）
DEFAULT_DEM_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "gis" / "study_area.tif"

# 默认研究区 bbox：黄河吕梁段吴堡水文站附近
DEFAULT_BBOX = (110.7, 37.4, 111.2, 37.8)  # minx, miny, maxx, maxy


@dataclass
class DEMDataset:
    """加载到内存的 DEM 数据集。"""

    elevation: np.ndarray  # 高程数组（米），shape=(height, width)
    transform: rasterio.Affine  # 仿射变换（像素 ↔ 地理坐标）
    crs: str  # 坐标参考系统
    bounds: BoundingBox  # 地理范围 (left, bottom, right, top)
    nodata: float  # 无效值

    @property
    def shape(self) -> Tuple[int, int]:
        return self.elevation.shape

    @property
    def resolution_m(self) -> Tuple[float, float]:
        """返回像素分辨率（米）。"""
        # SRTM 在赤道附近约 30m，纬度越高经度方向分辨率越低
        # 这里直接用 transform 的像素大小（单位：度）转米
        res_x_deg = abs(self.transform.a)
        res_y_deg = abs(self.transform.e)
        # 1° 纬度 ≈ 111.32 km；1° 经度 ≈ 111.32 * cos(lat) km
        lat_center = (self.bounds.bottom + self.bounds.top) / 2
        lat_rad = np.radians(lat_center)
        res_x_m = res_x_deg * 111320 * np.cos(lat_rad)
        res_y_m = res_y_deg * 111320
        return (res_x_m, res_y_m)


def load_study_dem(
    dem_path: Optional[Path] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> DEMDataset:
    """加载研究区 DEM 数据。

    Args:
        dem_path: GeoTIFF 文件路径，默认 data/processed/gis/study_area.tif
        bbox: 可选，按 (minx, miny, maxx, maxy) 裁剪；None 表示加载全部

    Returns:
        DEMDataset 数据集

    Raises:
        FileNotFoundError: DEM 文件不存在
        RuntimeError: DEM 文件无法打开
    """
    dem_path = Path(dem_path) if dem_path else DEFAULT_DEM_PATH
    if not dem_path.exists():
        raise FileNotFoundError(
            f"DEM 文件不存在: {dem_path}\n"
            f"请先运行 backend/build_terrain_data.py 生成研究区 DEM"
        )

    try:
        with rasterio.open(dem_path) as src:
            if bbox is not None:
                # 按 bbox 裁剪
                window = from_bounds(*bbox, src.transform)
                elevation = src.read(1, window=window)
                transform = src.window_transform(window)
                bounds = rasterio.windows.bounds(window, src.transform)
            else:
                elevation = src.read(1)
                transform = src.transform
                bounds = src.bounds

            nodata = src.nodata if src.nodata is not None else -32768

            # 处理 nodata 值：替换为 NaN 便于后续计算
            elevation = elevation.astype(np.float32)
            elevation[elevation == nodata] = np.nan
            # SRTM 有时用 0 表示海洋/无效，黄河段不应有 0 以下的高程
            elevation[elevation < -1000] = np.nan

            dataset = DEMDataset(
                elevation=elevation,
                transform=transform,
                crs=str(src.crs) if src.crs else "EPSG:4326",
                bounds=BoundingBox(*bounds),
                nodata=float(nodata),
            )
            logger.info(
                "[dem_loader] 加载 %s: shape=%s, bounds=%s, res=%.2fm",
                dem_path.name, dataset.shape, dataset.bounds, dataset.resolution_m[0],
            )
            return dataset
    except rasterio.RasterioIOError as e:
        raise RuntimeError(f"无法打开 DEM 文件: {dem_path}: {e}") from e


def is_dem_ready(dem_path: Optional[Path] = None) -> bool:
    """检查 DEM 数据是否已就绪。"""
    dem_path = Path(dem_path) if dem_path else DEFAULT_DEM_PATH
    return dem_path.exists()
