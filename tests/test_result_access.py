"""End-to-end ownership enforcement on GET /api/diagnoses/{id}.

A diagnosis id is unguessable but not secret, so possession of the link must
not be enough to read someone else's result.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

OWNER = "U_owner_1111"
STRANGER = "U_stranger_2222"
CHANNEL = "liff-channel-1"


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


@pytest.fixture
def client_and_id(tmp_path, monkeypatch):
    from app.config import settings

    settings.line_channel_secret = "secret"
    settings.line_channel_access_token = "token"
    settings.database_url = f"sqlite:///{tmp_path / 'access.db'}"
    settings.upload_dir = str(tmp_path / "uploads")
    settings.liff_channel_id = CHANNEL

    from app.main import app

    with TestClient(app) as client:
        from app.db import get_session, record_diagnosis, upsert_user

        with get_session() as session:
            upsert_user(session, OWNER)
            record = record_diagnosis(
                session, OWNER,
                [{"class": "Anthracnose", "confidence": 0.9},
                 {"class": "Canker", "confidence": 0.06},
                 {"class": "Healthy", "confidence": 0.04}],
                "confident", "efficientnet_b0")
            diagnosis_id = record.id
        yield client, diagnosis_id

    settings.liff_channel_id = ""


def _token_for(monkeypatch, user_id: str, aud: str = CHANNEL):
    monkeypatch.setattr("app.auth.httpx.post", lambda *a, **k: FakeResponse(
        200, {"sub": user_id, "aud": aud}))


def test_owner_can_read_their_diagnosis(client_and_id, monkeypatch):
    client, diagnosis_id = client_and_id
    _token_for(monkeypatch, OWNER)

    res = client.get(f"/api/diagnoses/{diagnosis_id}",
                     headers={"Authorization": "Bearer good-token"})

    assert res.status_code == 200
    body = res.json()
    assert body["top3"][0]["class"] == "Anthracnose"
    assert body["disease"]["name_th"] == "โรคแอนแทรคโนส"
    # The response must never leak who the farmer is.
    assert "line_user_id" not in body


def test_stranger_with_valid_token_is_forbidden(client_and_id, monkeypatch):
    client, diagnosis_id = client_and_id
    _token_for(monkeypatch, STRANGER)

    res = client.get(f"/api/diagnoses/{diagnosis_id}",
                     headers={"Authorization": "Bearer someone-elses-token"})

    assert res.status_code == 403


def test_no_token_is_unauthorized(client_and_id):
    client, diagnosis_id = client_and_id
    assert client.get(f"/api/diagnoses/{diagnosis_id}").status_code == 401


def test_invalid_token_is_unauthorized(client_and_id, monkeypatch):
    client, diagnosis_id = client_and_id
    monkeypatch.setattr("app.auth.httpx.post",
                        lambda *a, **k: FakeResponse(400, {}))

    res = client.get(f"/api/diagnoses/{diagnosis_id}",
                     headers={"Authorization": "Bearer bad-token"})

    assert res.status_code == 401


def test_token_for_another_channel_is_rejected(client_and_id, monkeypatch):
    client, diagnosis_id = client_and_id
    _token_for(monkeypatch, OWNER, aud="a-different-channel")

    res = client.get(f"/api/diagnoses/{diagnosis_id}",
                     headers={"Authorization": "Bearer wrong-audience"})

    assert res.status_code == 401


def test_unknown_id_is_404_even_with_valid_token(client_and_id, monkeypatch):
    client, _ = client_and_id
    _token_for(monkeypatch, OWNER)

    res = client.get("/api/diagnoses/00000000-0000-0000-0000-000000000000",
                     headers={"Authorization": "Bearer good-token"})

    assert res.status_code == 404
