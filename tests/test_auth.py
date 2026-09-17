"""LIFF ID-token handling and result-page ownership enforcement."""
from __future__ import annotations

import pytest

from app.auth import bearer_token, verify_id_token


# --- header parsing ----------------------------------------------------------
@pytest.mark.parametrize("header,expected", [
    ("Bearer abc123", "abc123"),
    ("bearer abc123", "abc123"),
    ("Bearer   spaced  ", "spaced"),
    (None, None),
    ("", None),
    ("abc123", None),          # no scheme
    ("Basic abc123", None),    # wrong scheme
    ("Bearer", None),          # no token
    ("Bearer    ", None),      # whitespace only
])
def test_bearer_token_parsing(header, expected):
    assert bearer_token(header) == expected


# --- token verification ------------------------------------------------------
class FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def test_missing_inputs_short_circuit():
    assert verify_id_token("", "channel") is None
    assert verify_id_token("token", "") is None


def test_valid_token_returns_subject(monkeypatch):
    monkeypatch.setattr("httpx.post", lambda *a, **k: FakeResponse(
        200, {"sub": "U_owner", "aud": "channel-1"}))
    assert verify_id_token("tok", "channel-1") == "U_owner"


def test_rejected_token_returns_none(monkeypatch):
    monkeypatch.setattr("httpx.post", lambda *a, **k: FakeResponse(
        400, {"error": "invalid_request"}))
    assert verify_id_token("tok", "channel-1") is None


def test_audience_mismatch_is_refused(monkeypatch):
    """A token minted for a different channel must never be accepted."""
    monkeypatch.setattr("httpx.post", lambda *a, **k: FakeResponse(
        200, {"sub": "U_owner", "aud": "someone-elses-channel"}))
    assert verify_id_token("tok", "channel-1") is None


def test_network_failure_fails_closed(monkeypatch):
    import httpx

    def boom(*a, **k):
        raise httpx.ConnectError("no route")

    monkeypatch.setattr("httpx.post", boom)
    assert verify_id_token("tok", "channel-1") is None
