import os
from fastapi import APIRouter, HTTPException

router = APIRouter()

ABLY_CHANNEL = "gunshot-detection"
TOKEN_TTL_MS = 3_600_000   # 1 hour


@router.get("/ably-token")
async def get_ably_token():
    """
    Generate a short-lived Ably capability token request for the frontend.
    The Ably JS SDK uses the returned TokenRequest to authenticate without
    ever seeing the raw API key.
    Requires ABLY_API_KEY (full root key: app_id.key_name:key_secret).
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
        token_request = await client.auth.create_token_request(
            token_params={
                "capability": {ABLY_CHANNEL: ["subscribe", "publish"]},
                "ttl": TOKEN_TTL_MS,
            }
        )
        # TokenRequest is a dataclass — serialise to dict for JSON response
        return {
            "keyName":   token_request.key_name,
            "ttl":       token_request.ttl,
            "nonce":     token_request.nonce,
            "timestamp": token_request.timestamp,
            "capability": token_request.capability,
            "mac":       token_request.mac,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ably token error: {exc}")
