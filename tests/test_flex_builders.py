"""Flex builder tests: valid JSON, within LINE's platform limits, and --
most importantly -- no invented medical content for classes whose knowledge
base entry is still awaiting expert review.
"""
from __future__ import annotations

import json

import pytest

from app import messages as M
from app.flex import builders as B
from app.knowledge import Disease, load_knowledge_base

WEB = "https://example.org"


@pytest.fixture(scope="module")
def kb():
    return load_knowledge_base()


def _is_json_serialisable(container) -> bool:
    json.dumps(container, ensure_ascii=False)
    return True


def _all_text(container) -> str:
    return json.dumps(container, ensure_ascii=False)


# --- structural validity -----------------------------------------------------
def test_how_to_photo_bubble_is_valid(kb):
    bubble = B.build_how_to_photo_bubble()
    assert bubble["type"] == "bubble"
    assert _is_json_serialisable(bubble)
    B.validate_container(bubble)
    for tip in M.HOW_TO_PHOTO_TIPS:
        assert tip in _all_text(bubble)


def test_result_bubble_is_valid_and_within_limits(kb):
    disease = kb.by_class("Anthracnose")
    bubble = B.build_result_bubble(disease, 0.91, detail_url=f"{WEB}/result/1")
    assert bubble["type"] == "bubble"
    assert bubble["size"] == "mega"
    B.validate_container(bubble)
    assert B.container_size_bytes(bubble) < B.MAX_BUBBLE_BYTES


def test_result_bubble_carries_disclaimer(kb):
    bubble = B.build_result_bubble(kb.by_class("Anthracnose"), 0.91)
    assert M.DISCLAIMER in _all_text(bubble)


def test_healthy_bubble_uses_green_and_care_tips(kb):
    bubble = B.build_result_bubble(kb.by_class("Healthy"), 0.96)
    text = _all_text(bubble)
    assert "#2E8B57" in text  # LEAF_500, the healthy colour
    assert M.HEADING_CARE_TIPS in text
    # A healthy result must not show treatment sections.
    assert M.HEADING_ACTIONS not in text


def test_severity_drives_header_colour(kb):
    severe = B.build_result_bubble(kb.by_class("Pink_disease"), 0.9)
    assert severe["header"]["backgroundColor"] == "#C0392B"  # DANGER_600


def test_uncertain_bubble_shows_both_candidates(kb):
    bubble = B.build_uncertain_bubble(
        kb.by_class("Anthracnose"), 0.60, kb.by_class("Pink_disease"), 0.25)
    text = _all_text(bubble)
    assert "โรคแอนแทรคโนส" in text
    assert "โรคราสีชมพู" in text
    assert M.RESULT_UNCERTAIN_HINT in text
    B.validate_container(bubble)


def test_uncertain_bubble_without_secondary(kb):
    bubble = B.build_uncertain_bubble(kb.by_class("Anthracnose"), 0.6, None, None)
    B.validate_container(bubble)


def test_notice_bubbles_are_valid():
    for bubble in (B.build_not_confident_bubble(), B.build_not_durian_bubble(),
                   B.build_model_not_ready_bubble()):
        assert bubble["type"] == "bubble"
        B.validate_container(bubble)


def test_knowledge_carousel_within_limits(kb):
    carousel = B.build_knowledge_carousel(
        [d for d in kb.all() if not d.is_healthy], WEB)
    assert carousel["type"] == "carousel"
    assert len(carousel["contents"]) <= B.MAX_CAROUSEL_BUBBLES
    assert B.container_size_bytes(carousel) < B.MAX_CAROUSEL_BYTES
    B.validate_container(carousel)


def test_carousel_truncates_past_the_bubble_cap(kb):
    many = kb.all() * 3
    carousel = B.build_knowledge_carousel(many, WEB)
    assert len(carousel["contents"]) == B.MAX_CAROUSEL_BUBBLES
    B.validate_container(carousel)


def test_validate_container_rejects_oversized_carousel():
    fat = {"type": "carousel", "contents": [{"type": "bubble"}] * 20}
    with pytest.raises(ValueError, match="bubbles"):
        B.validate_container(fat)


# --- quick replies -----------------------------------------------------------
def test_quick_replies_within_item_limit():
    for qr in (B.photo_quick_reply(), B.default_quick_reply()):
        assert len(qr["items"]) <= B.MAX_QUICK_REPLY_ITEMS


def test_photo_quick_reply_offers_camera_and_album():
    actions = {i["action"]["type"] for i in B.photo_quick_reply()["items"]}
    assert actions == {"camera", "cameraRoll"}


# --- the honesty guarantee ---------------------------------------------------
def test_pending_disease_shows_notice_not_invented_advice():
    """A class still awaiting expert content must render the 'being prepared'
    notice -- never a fabricated symptom or treatment list."""
    pending = Disease(
        class_name="Canker", slug="canker", name_th="โรคแคงเกอร์",
        name_en="Canker", pathogen=None, type=None, severity="watch",
        pending_expert_input=True,
    )
    bubble = B.build_result_bubble(pending, 0.88)
    text = _all_text(bubble)
    assert M.PENDING_DISEASE_NOTICE in text
    assert M.HEADING_ACTIONS not in text
    assert M.HEADING_SYMPTOMS not in text


def test_all_pending_kb_entries_have_no_content(kb):
    """Guards the KB itself: a pending entry must stay empty until a human
    fills it in, so nothing unreviewed can reach a farmer."""
    for disease in kb.pending():
        assert disease.symptoms == []
        assert disease.immediate_actions == []
        assert disease.chemical_options == []


def test_no_chemical_option_carries_a_numeric_rate(kb):
    """Dosage numbers must never appear -- always 'ใช้ตามอัตราบนฉลาก'."""
    for disease in kb.all():
        for option in disease.chemical_options:
            assert any(ch.isdigit() for ch in option["active_ingredient"]) is False
            assert "อัตราบนฉลาก" in option["note"]


# --- confidence bar ----------------------------------------------------------
@pytest.mark.parametrize("conf,expected", [(0.0, "2%"), (0.5, "50%"), (1.0, "100%")])
def test_confidence_bar_width_tracks_confidence(kb, conf, expected):
    bubble = B.build_result_bubble(kb.by_class("Anthracnose"), conf)
    assert expected in _all_text(bubble)
