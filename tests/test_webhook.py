"""Webhook security and API surface.

The signature check is the only thing standing between the public internet
and this service, so it gets tested first and hardest.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

SECRET = "test-channel-secret"


def _sign(body: str, secret: str = SECRET) -> str:
    digest = hmac.new(secret.encode(), body.encode(), hashlib.sha256).digest()
    return base64.b64encode(digest).decode()


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bot")
    from app.config import settings

    settings.line_channel_secret = SECRET
    settings.line_channel_access_token = "test-token"
    settings.database_url = f"sqlite:///{tmp / 'test.db'}"
    settings.upload_dir = str(tmp / "uploads")

    from app.main import app

    with TestClient(app) as c:
        yield c


def test_health_reports_model_state(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    # No checkpoint is committed to the repo, so this is False here -- and
    # that is exactly what the bot must surface rather than fake a result.
    assert body["model_loaded"] is False


def test_webhook_rejects_missing_signature(client):
    resp = client.post("/webhook", content=json.dumps({"events": []}))
    assert resp.status_code == 400


def test_webhook_rejects_wrong_signature(client):
    body = json.dumps({"events": []})
    resp = client.post("/webhook", content=body,
                       headers={"X-Line-Signature": _sign(body, "wrong-secret")})
    assert resp.status_code == 400


def test_webhook_rejects_tampered_body(client):
    signed = json.dumps({"events": []})
    tampered = json.dumps({"events": [{"type": "message"}]})
    resp = client.post("/webhook", content=tampered,
                       headers={"X-Line-Signature": _sign(signed)})
    assert resp.status_code == 400


def test_webhook_accepts_valid_signature(client):
    body = json.dumps({"events": [], "destination": "x"})
    resp = client.post("/webhook", content=body,
                       headers={"X-Line-Signature": _sign(body)})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


# --- read-only API used by the website / LIFF page -------------------------
def test_list_diseases_covers_every_model_class(client):
    from app.diagnosis import NOT_DURIAN_CLASS
    from app.ml.predictor import get_predictor
    from app.config import settings

    predictor = get_predictor(settings.config_yaml_path, settings.model_name)
    returned = {d["class_name"] for d in client.get("/api/diseases").json()["diseases"]}
    expected = {c for c in predictor.classes if c != NOT_DURIAN_CLASS}
    assert expected <= returned


def test_get_disease_by_slug(client):
    body = client.get("/api/diseases/anthracnose").json()
    assert body["class_name"] == "Anthracnose"
    assert body["name_th"] == "โรคแอนแทรคโนส"


def test_unknown_disease_slug_is_404(client):
    assert client.get("/api/diseases/not-a-real-disease").status_code == 404


def test_unknown_diagnosis_is_404(client):
    assert client.get("/api/diagnoses/00000000-0000-0000-0000-000000000000"
                      ).status_code == 404
