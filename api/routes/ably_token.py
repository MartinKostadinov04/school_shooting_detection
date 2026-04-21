import os
import time
from fastapi import APIRouter, HTTPException

router = APIRouter()

ABLY_CHANNEL  = "gunshot-detection"
TOKEN_TTL_MS  = 3_600_000   # 1 hour


@router.get("/ably-token")
def get_ably_token():
    """
    Generate a short-lived Ably capability token for the frontend.
    The frontend calls this instead of embedding the bare API key in its bundle.
    Requires ABLY_API_KEY (full key: app_id.key_name:key_secret).
    """
    api_key = os.getenv("ABLY_API_KEY", "")
    if not api_key or ":" not in api_key:
        raise HTTPException(
            status_code=503,
            detail="ABLY_API_KEY not configured on server",
        )

    try:
        from ably import AblyRest
        client = AblyRest(api_key)
        token_request = client.auth.create_token_request(
            token_params={
                "capability": {ABLY_CHANNEL: ["subscribe", "publish"]},
                "ttl": TOKEN_TTL_MS,
            }
        )
        # Return the signed token request — Ably JS SDK can use it directly
        return token_request
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ably token error: {exc}")
