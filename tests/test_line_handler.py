"""Routing and state-machine tests for LineService.

Events are built by running realistic LINE webhook payloads through the real
WebhookParser, so the isinstance dispatch in handle_event() is genuinely
exercised rather than mocked away. Only the network transport (reply / push /
download / loading animation) is stubbed.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
from typing import Any, List

import pytest
from linebot.v3.webhook import WebhookParser
from PIL import Image

from app import messages as M
from app.config import Settings
from app.db import init_db
from app.knowledge import load_knowledge_base
from app.line_handler import LineService, OutgoingFlex
from app.ml.predictor import ClassScore
from app.state import ConversationState, StateStore

SECRET = "handler-test-secret"
USER = "U4af4980629b1f6f1f1f1f1f1f1f1f1f1"


class FakePredictor:
    """Stands in for DurianPredictor without loading any weights."""

    def __init__(self, ready: bool = True, top3: List[ClassScore] | None = None):
        self.ready = ready
        self.model_name = "efficientnet_b0"
        self.temperature = 1.0
        self.classes = ["Anthracnose", "Healthy", "not_durian"]
        self._top3 = top3 or [
            ClassScore("Anthracnose", 0.93),
            ClassScore("Pink_disease", 0.04),
            ClassScore("Healthy", 0.03),
        ]

    def predict(self, image, top_k: int = 3):
        return self._top3[:top_k]


class RecordingService(LineService):
    """LineService with the network calls replaced by recorders."""

    def __init__(self, *args, image_bytes: bytes = b"", **kwargs):
        super().__init__(*args, **kwargs)
        self.replies: List[List[Any]] = []
        self.pushes: List[List[Any]] = []
        self.loading_shown: List[str] = []
        self._image_bytes = image_bytes

    def reply(self, reply_token: str, payloads: List[Any]) -> None:
        self.replies.append(payloads)

    def push(self, user_id: str, payloads: List[Any]) -> None:
        self.pushes.append(payloads)

    def show_loading(self, user_id: str) -> None:
        self.loading_shown.append(user_id)

    def download_image(self, message_id: str) -> bytes:
        return self._image_bytes


def _jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (320, 240), (100, 150, 80)).save(buf, format="JPEG")
    return buf.getvalue()


def _parse(payload: dict):
    body = json.dumps(payload)
    sig = base64.b64encode(
        hmac.new(SECRET.encode(), body.encode(), hashlib.sha256).digest()
    ).decode()
    return WebhookParser(SECRET).parse(body, sig)


def _text_event(text: str) -> Any:
    return _parse({
        "destination": "U0000",
        "events": [{
            "type": "message",
            "message": {"type": "text", "id": "1", "text": text,
                        "quoteToken": "quote-token-1"},
            "webhookEventId": "01FZ74A0TDDPYRVKNK77XKC3ZR",
            "deliveryContext": {"isRedelivery": False},
            "timestamp": 1625665242211,
            "source": {"type": "user", "userId": USER},
            "replyToken": "reply-token-1",
            "mode": "active",
        }],
    })[0]


def _image_event() -> Any:
    return _parse({
        "destination": "U0000",
        "events": [{
            "type": "message",
            "message": {"type": "image", "id": "42",
                        "contentProvider": {"type": "line"},
                        "quoteToken": "quote-token-2"},
            "webhookEventId": "01FZ74A0TDDPYRVKNK77XKC3ZS",
            "deliveryContext": {"isRedelivery": False},
            "timestamp": 1625665242211,
            "source": {"type": "user", "userId": USER},
            "replyToken": "reply-token-2",
            "mode": "active",
        }],
    })[0]


def _follow_event() -> Any:
    return _parse({
        "destination": "U0000",
        "events": [{
            "type": "follow",
            "webhookEventId": "01FZ74A0TDDPYRVKNK77XKC3ZT",
            "deliveryContext": {"isRedelivery": False},
            "timestamp": 1625665242211,
            "source": {"type": "user", "userId": USER},
            "replyToken": "reply-token-3",
            "mode": "active",
        }],
    })[0]


@pytest.fixture
def make_service(tmp_path):
    init_db(f"sqlite:///{tmp_path / 'handler.db'}")

    def _make(predictor=None, image_bytes: bytes | None = None):
        settings = Settings(
            line_channel_secret=SECRET,
            line_channel_access_token="token",
            upload_dir=str(tmp_path / "uploads"),
            web_base_url="https://example.org",
        )
        return RecordingService(
            settings,
            predictor or FakePredictor(),
            load_knowledge_base(),
            StateStore(ttl_seconds=600),
            image_bytes=image_bytes if image_bytes is not None else _jpeg_bytes(),
        )

    return _make


def _flex(payload) -> OutgoingFlex:
    assert isinstance(payload, OutgoingFlex)
    return payload


def _flex_text(payload) -> str:
    return json.dumps(_flex(payload).container, ensure_ascii=False)


# --- text routing ------------------------------------------------------------
def test_diagnose_keyword_prompts_for_photo_and_sets_state(make_service):
    svc = make_service()
    svc.handle_event(_text_event(M.KEYWORD_DIAGNOSE))

    assert len(svc.replies) == 1
    out = _flex(svc.replies[0][0])
    assert M.HOW_TO_PHOTO_TIPS[0] in json.dumps(out.container, ensure_ascii=False)
    assert out.quick_reply is not None
    assert svc.state.get(USER) is ConversationState.WAITING_IMAGE


def test_how_to_photo_keyword_does_not_change_state(make_service):
    svc = make_service()
    svc.handle_event(_text_event(M.KEYWORD_HOW_TO_PHOTO))
    assert svc.state.get(USER) is ConversationState.IDLE
    assert M.HOW_TO_PHOTO_TITLE in _flex_text(svc.replies[0][0])


def test_knowledge_keyword_returns_carousel(make_service):
    svc = make_service()
    svc.handle_event(_text_event(M.KEYWORD_KNOWLEDGE))
    container = _flex(svc.replies[0][0]).container
    assert container["type"] == "carousel"
    # The healthy entry is not a "disease" to browse.
    assert "ต้นแข็งแรงดี" not in json.dumps(container, ensure_ascii=False)


def test_contact_keyword_returns_configured_text(make_service):
    svc = make_service()
    svc.handle_event(_text_event(M.KEYWORD_CONTACT))
    assert svc.replies[0][0] == svc.settings.admin_contact_text


def test_unknown_text_gets_nudge_with_quick_reply(make_service):
    svc = make_service()
    svc.handle_event(_text_event("สวัสดีครับหมอ"))
    msg = svc.replies[0][0]
    assert msg.text == M.FALLBACK_TEXT
    assert msg.quick_reply is not None


# --- follow ------------------------------------------------------------------
def test_follow_records_user_without_replying(make_service):
    """OA Manager already sends the greeting; replying here would double up."""
    svc = make_service()
    svc.handle_event(_follow_event())
    assert svc.replies == []
    assert svc.pushes == []


# --- image / diagnosis -------------------------------------------------------
def test_image_shows_loading_then_pushes_result(make_service):
    svc = make_service()
    svc.handle_event(_image_event())

    assert svc.loading_shown == [USER]
    assert len(svc.pushes) == 1
    text = _flex_text(svc.pushes[0][0])
    assert "โรคแอนแทรคโนส" in text
    assert M.DISCLAIMER in text


def test_image_clears_waiting_state(make_service):
    svc = make_service()
    svc.state.set(USER, ConversationState.WAITING_IMAGE)
    svc.handle_event(_image_event())
    assert svc.state.get(USER) is ConversationState.IDLE


def test_image_accepted_even_when_idle(make_service):
    svc = make_service()
    assert svc.state.get(USER) is ConversationState.IDLE
    svc.handle_event(_image_event())
    assert len(svc.pushes) == 1


def test_uncertain_confidence_yields_top2_card(make_service):
    svc = make_service(predictor=FakePredictor(top3=[
        ClassScore("Anthracnose", 0.58),
        ClassScore("Pink_disease", 0.31),
        ClassScore("Healthy", 0.11),
    ]))
    svc.handle_event(_image_event())
    text = _flex_text(svc.pushes[0][0])
    assert M.RESULT_UNCERTAIN_HINT in text
    assert "โรคราสีชมพู" in text


def test_low_confidence_yields_not_confident_card(make_service):
    svc = make_service(predictor=FakePredictor(top3=[
        ClassScore("Anthracnose", 0.30),
        ClassScore("Pink_disease", 0.28),
        ClassScore("Healthy", 0.20),
    ]))
    svc.handle_event(_image_event())
    assert M.RESULT_NOT_CONFIDENT_BODY in _flex_text(svc.pushes[0][0])


def test_not_durian_yields_reject_card(make_service):
    svc = make_service(predictor=FakePredictor(top3=[
        ClassScore("not_durian", 0.97),
        ClassScore("Healthy", 0.02),
        ClassScore("Anthracnose", 0.01),
    ]))
    svc.handle_event(_image_event())
    assert M.RESULT_NOT_DURIAN_BODY in _flex_text(svc.pushes[0][0])


def test_healthy_prediction_yields_green_card(make_service):
    svc = make_service(predictor=FakePredictor(top3=[
        ClassScore("Healthy", 0.95),
        ClassScore("Anthracnose", 0.03),
        ClassScore("Pink_disease", 0.02),
    ]))
    svc.handle_event(_image_event())
    text = _flex_text(svc.pushes[0][0])
    assert "#2E8B57" in text
    assert M.RESULT_HEALTHY_BODY in text


def test_unready_model_says_so_instead_of_guessing(make_service):
    svc = make_service(predictor=FakePredictor(ready=False))
    svc.handle_event(_image_event())
    assert M.ERROR_MODEL_NOT_READY in _flex_text(svc.pushes[0][0])


def test_oversized_image_is_rejected_politely(make_service):
    svc = make_service(image_bytes=b"x" * (11 * 1024 * 1024))
    svc.handle_event(_image_event())
    assert svc.pushes[0][0] == M.ERROR_IMAGE_TOO_LARGE


def test_download_failure_surfaces_polite_error(make_service):
    svc = make_service()

    def boom(message_id: str) -> bytes:
        raise RuntimeError("LINE content API down")

    svc.download_image = boom
    svc.handle_event(_image_event())
    assert svc.pushes[0][0] == M.ERROR_BUSY


def test_diagnosis_is_persisted(make_service):
    from app.db import get_session
    from app.db import Diagnosis
    from sqlalchemy import select

    svc = make_service()
    svc.handle_event(_image_event())

    with get_session() as session:
        rows = session.execute(select(Diagnosis)).scalars().all()
    assert len(rows) == 1
    assert rows[0].top1_class == "Anthracnose"
    assert rows[0].tier == "confident"


def test_logs_never_contain_the_raw_user_id(make_service, caplog):
    import logging

    svc = make_service()
    with caplog.at_level(logging.INFO):
        svc.handle_event(_image_event())
    assert USER not in caplog.text
