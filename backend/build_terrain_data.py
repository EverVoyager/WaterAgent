"""生成研究区 DEM（GeoTIFF）。

数据源说明：
    原计划从 AWS Open Data 下载 SRTM v3 公开 tiles，但该公开桶已于 2024 年
    逐步下线（404）。NASA Earthdata 需要账号、OpenTopography REST API 需要
    API key。为保证项目可在 Windows 无账号环境下直接跑通，这里采用**算法合成
    DEM 方案**：基于黄河吕梁段真实经纬度边界 + 黄土高原地形特征，用 Perlin 噪声
    + 河道下切函数合成符合实际地形特征的 GeoTIFF。

    所有 GIS 分析算法（坡度 Horn 算法、河床断面、淹没分析）都是真实的，
    后续若拿到真实 SRTM DEM，只需替换 study_area.tif 一个文件即可。

合成 DEM 的地形特征：
- bbox: 110.7°E-111.2°E, 37.4°N-37.8°N（吴堡水文站附近）
- 分辨率：约 30m（与 SRTM 一致）
- 高程范围：~640m（黄河河道）~ ~1500m（黄土高原山顶）
- 黄河河道：沿 110.74°E 南北走向下切，基准水位 640m
- 沟壑地形：Perlin 噪声模拟黄土高原沟壑纵横特征

用法：
    cd backend
    python build_terrain_data.py
"""
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds

# 路径设置
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
for p in (str(PROJECT_ROOT), str(BACKEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from agent.gis.dem_loader import DEFAULT_BBOX  # noqa: E402

# 输出路径
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "gis"
OUTPUT_TIF = OUTPUT_DIR / "study_area.tif"

# 河道参数
RIVER_LON = 110.74  # 黄河河道中心经度
RIVER_BASE_LEVEL = 640.0  # 河道基准水位（米）


def perlin_noise_2d(shape, scale=0.05, seed=42):
    """生成 2D Perlin 风格噪声（用于地形起伏）。

    简化版：用多频段高斯噪声叠加模拟 Perlin 噪声的分形特征。
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    noise = np.zeros(shape, dtype=np.float32)

    # 多频段叠加（fractal brownian motion）
    octaves = [
        (scale, 1.0),       # 大尺度起伏
        (scale * 2, 0.5),   # 中尺度
        (scale * 4, 0.25),  # 小尺度
        (scale * 8, 0.125), # 细节
    ]

    for freq, amp in octaves:
        # 生成低分辨率噪声 + 双线性插值上采样
        lh = max(2, int(h * freq))
        lw = max(2, int(w * freq))
        low_noise = rng.standard_normal((lh, lw)).astype(np.float32)
        # 简单的上采样：用最近邻 + 高斯平滑近似
        from scipy.ndimage import zoom
        upsampled = zoom(low_noise, (h / lh, w / lw), order=3)
        noise += amp * upsampled

    # 归一化到 [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-9)
    return noise


def synthesize_dem(bbox, target_resolution_m=30):
    """合成符合黄河吕梁段地形特征的 DEM。

    Args:
        bbox: (minx, miny, maxx, maxy)
        target_resolution_m: 目标分辨率（米），约 30m 对应 SRTM

    Returns:
        dem: np.ndarray (height, width)，float32，单位米
        transform: rasterio.Affine
    """
    minx, miny, maxx, maxy = bbox

    # 计算像素数（按目标分辨率）
    lat_center = (miny + maxy) / 2
    lat_rad = np.radians(lat_center)
    # 1° 纬度 ≈ 111.32 km；1° 经度 ≈ 111.32 * cos(lat) km
    width_m = (maxx - minx) * 111320 * np.cos(lat_rad)
    height_m = (maxy - miny) * 111320

    width_px = int(width_m / target_resolution_m)
    height_px = int(height_m / target_resolution_m)

    # 限制大小，避免过大
    width_px = min(width_px, 1500)
    height_px = min(height_px, 1500)

    print(f"  DEM 尺寸: {width_px} x {height_px} 像素")
    print(f"  分辨率: ~{target_resolution_m}m")

    # 1. 基础高程：800m + Perlin 噪声 * 600m（200m ~ 1400m 范围）
    base_noise = perlin_noise_2d((height_px, width_px), scale=0.03, seed=42)
    elevation = 800.0 + base_noise * 600.0

    # 2. 黄河河道下切：沿 RIVER_LON 经度南北走向
    # 计算每列的经度
    lons = np.linspace(minx, maxx, width_px)
    # 距河道的水平距离（度）
    dist_to_river_deg = np.abs(lons - RIVER_LON)
    # 转米
    dist_to_river_m = dist_to_river_deg * 111320 * np.cos(lat_rad)
    # 河道影响范围：±2km
    river_influence_m = 2000.0
    # 河道下切函数：在河道处高程降到 640m，向外快速抬升
    river_profile = RIVER_BASE_LEVEL + (elevation.mean() - RIVER_BASE_LEVEL) * (
        1 - np.exp(-(dist_to_river_m / river_influence_m) ** 2)
    )
    # 广播到所有行
    elevation = np.minimum(elevation, river_profile[np.newaxis, :])

    # 3. 加入黄土高原沟壑特征（高频噪声）
    gully_noise = perlin_noise_2d((height_px, width_px), scale=0.15, seed=99)
    elevation += (gully_noise - 0.5) * 80.0  # ±40m 沟壑起伏

    # 4. 全局西高东低趋势（吕梁山区在西，黄河在东）
    west_east_gradient = np.linspace(0, 1, width_px)  # 西=0, 东=1
    elevation -= 100 * west_east_gradient[np.newaxis, :]  # 东侧低 100m

    # 5. 限制高程范围
    elevation = np.clip(elevation, 600.0, 1800.0).astype(np.float32)

    # 仿射变换（左上角原点，向东向北为正）
    transform = from_bounds(minx, miny, maxx, maxy, width_px, height_px)

    return elevation, transform


def main():
    print("=" * 60)
    print("生成黄河吕梁段研究区 DEM（算法合成）")
    print("=" * 60)
    print(f"研究区 bbox: {DEFAULT_BBOX}")
    print(f"输出 GeoTIFF: {OUTPUT_TIF}")
    print(f"河道中心经度: {RIVER_LON}°E")
    print(f"河道基准水位: {RIVER_BASE_LEVEL}m")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 合成 DEM
    print("[1/2] 合成 DEM（Perlin 噪声 + 河道下切 + 沟壑特征）...")
    elevation, transform = synthesize_dem(DEFAULT_BBOX, target_resolution_m=30)
    print(f"  高程范围: {elevation.min():.1f}m ~ {elevation.max():.1f}m (均值 {elevation.mean():.1f}m)")
    print()

    # 2. 写入 GeoTIFF
    print("[2/2] 写入 GeoTIFF ...")
    minx, miny, maxx, maxy = DEFAULT_BBOX
    meta = {
        "driver": "GTiff",
        "height": elevation.shape[0],
        "width": elevation.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": -32768.0,
    }
    with rasterio.open(OUTPUT_TIF, "w", **meta) as dst:
        dst.write(elevation, 1)
    print(f"  [OK] 已写入: {OUTPUT_TIF}")
    print(f"       大小: {OUTPUT_TIF.stat().st_size / 1024 / 1024:.1f} MB")
    print()

    # 验证
    print("=" * 60)
    print("DEM 元数据：")
    print("=" * 60)
    with rasterio.open(OUTPUT_TIF) as src:
        print(f"shape: {src.shape}")
        print(f"crs: {src.crs}")
        print(f"bounds: {src.bounds}")
        print(f"resolution: {src.res}")
        data = src.read(1)
        valid = data[data > -1000]
        if valid.size > 0:
            print(f"高程范围: {valid.min():.1f}m ~ {valid.max():.1f}m (均值 {valid.mean():.1f}m)")

    # 检查河道位置高程
    inv_transform = ~transform
    col_river, _ = inv_transform * (RIVER_LON, (miny + maxy) / 2)
    col_river = int(col_river)
    river_elev = elevation[:, col_river]
    print(f"河道位置 (lon={RIVER_LON}, col={col_river}) 高程: {river_elev.min():.1f}m ~ {river_elev.max():.1f}m")
    print()
    print("Agent 的 query_gis_terrain 工具现已走真实 GIS 分析（rasterio + DEM）。")
    print("注：DEM 数据为算法合成（基于真实经纬度 + 黄土高原地形特征），")
    print("    后续可替换为真实 SRTM DEM，仅需覆盖 study_area.tif 文件即可。")


if __name__ == "__main__":
    main()
