"""共享访问钥匙；本地 development 可留空，production 必须配置。"""
import secrets

from fastapi import Header, HTTPException, Request

from app.core.config import get_settings


def require_access_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    if request.url.path == "/api/health":
        return
    settings = get_settings()
    expected = settings.API_ACCESS_KEY
    if request.url.path.startswith("/api/memories/") and settings.MEMORY_ADMIN_API_KEY:
        expected = settings.MEMORY_ADMIN_API_KEY
    if expected and not secrets.compare_digest((x_api_key or "").encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="缺少或无效的访问钥匙")
