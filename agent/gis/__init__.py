"""GIS 地形分析模块。

基于 rasterio（GDAL 的 Python 封装）+ 真实 SRTM DEM 数据，
为 Agent 的 query_gis_terrain 工具提供真实的地形分析能力。

模块结构：
- dem_loader.py: 加载/裁剪 GeoTIFF DEM 数据
- terrain_analyzer.py: 实现坡度/河床断面/淹没范围分析

数据流：
    SRTM .hgt tiles (AWS Open Data)
        → build_terrain_data.py 合并裁剪为 study_area.tif
        → dem_loader.DEMDataset 加载
        → terrain_analyzer.TerrainAnalyzer 分析
        → real_executor.query_gis_terrain_real 调用
"""
from agent.gis.dem_loader import DEMDataset, load_study_dem
from agent.gis.terrain_analyzer import TerrainAnalysisResult, TerrainAnalyzer

__all__ = ["DEMDataset", "load_study_dem", "TerrainAnalyzer", "TerrainAnalysisResult"]
