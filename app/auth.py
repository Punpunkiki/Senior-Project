"""app/auth.py -- LIFF ID-token verification.

A diagnosis id is a UUID, which is unguessable but not secret: anyone who
obtains the link could otherwise read someone else's result. The LIFF result
page therefore sends the viewer's LINE ID token, and we ask LINE to verify it
and tell us who it belongs to.

If `liff_channel_id` is not configured the check cannot run. Rather than fail
closed (which would break local development before a LIFF channel exists) or
fail open silently (which would be a quiet privacy hole), the service logs a
loud warning at startup and the endpoint stays open. Configure the channel id
before anything real is served.
"""
from __future__ import annotations

from typing import Optional

import httpx

from src.utils import get_logger

log = get_logger("auth")

LINE_VERIFY_URL = "https://api.line.me/oauth2/v2.1/verify"
_TIMEOUT_SECONDS = 5.0


def verify_id_token(id_token: str, channel_id: str) -> Optional[str]:
    """Return the LINE user id (`sub`) the token belongs to, or None.

    LINE validates the signature, expiry and audience for us, so we do not
    parse the JWT ourselves.
    """
    if not id_token or not channel_id:
        return None
    try:
        response = httpx.post(
            LINE_VERIFY_URL,
            data={"id_token": id_token, "client_id": channel_id},
            timeout=_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as exc:
        log.warning("ID token verification request failed: %s", exc)
        return None

    if response.status_code != 200:
        log.info("ID token rejected by LINE (status %s)", response.status_code)
        return None

    payload = response.json()
    if payload.get("aud") != channel_id:
        log.warning("ID token audience mismatch")
        return None
    return payload.get("sub")


def bearer_token(authorization: Optional[str]) -> Optional[str]:
    """Pull the token out of an `Authorization: Bearer <token>` header."""
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()
