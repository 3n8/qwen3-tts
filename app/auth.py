import os
from fastapi import Header, HTTPException, status
from typing import Optional


async def verify_api_key(
    x_tts_api_key: Optional[str] = Header(None, alias="x-tts-api-key"),
    xi_api_key: Optional[str] = Header(None, alias="xi-api-key"),
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> str:
    api_key = x_tts_api_key or xi_api_key or x_api_key

    if authorization and authorization.startswith("Bearer "):
        api_key = api_key or authorization[7:]

    expected_key = os.getenv("TTS_API_KEY")

    if not expected_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="TTS_API_KEY not configured on server",
        )

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_UNAUTHORIZED,
            detail="API key required. Pass via x-tts-api-key, xi-api-key, x-api-key, or Authorization header",
        )

    if api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_UNAUTHORIZED, detail="Invalid API key"
        )

    return api_key
