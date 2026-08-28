"""速率限制配置（slowapi）。

独立模块存放 Limiter 实例，避免 main.py ↔ api 模块循环导入。
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

# 默认每分钟每 IP 30 次（覆盖所有未显式标注的端点）；
# 健康检查等高频探测端点可在路由上用 @limiter.limit 放宽
limiter = Limiter(key_func=get_remote_address, default_limits=["30/minute"])
