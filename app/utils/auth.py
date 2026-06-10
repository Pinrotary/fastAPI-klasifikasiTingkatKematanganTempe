from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

from app.config.settings import settings

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """
    Dependency — validasi API Key dari header X-API-Key.
    Tambahkan ke endpoint dengan: Depends(verify_api_key)
    """
    if not settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API Key belum dikonfigurasi di server.",
        )
    if api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key tidak valid atau tidak ditemukan.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return api_key
