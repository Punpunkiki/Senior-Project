"""app/flex/builders.py -- Flex Message builders.

Every builder returns a plain dict (not an SDK object) for three reasons:
  1. unit-testable without the LINE SDK or network,
  2. exportable straight into the Flex Message Simulator
     (scripts/export_flex_samples.py -> docs/flex-samples/),
  3. the SDK wrapping happens once, in app/line_handler.py.

LINE platform limits enforced here (verified against LINE's docs):
  - a bubble's JSON must stay under 30 KB, a carousel's under 50 KB
  - a carousel holds at most 12 bubbles
  - a message carries at most 13 quick-reply items
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from app import messages as M
from app.flex import tokens as T
from app.knowledge import Disease

MAX_BUBBLE_BYTES = 30_000
MAX_CAROUSEL_BYTES = 50_000
MAX_CAROUSEL_BUBBLES = 12
MAX_QUICK_REPLY_ITEMS = 13

# Spec caps on how much of the knowledge base a single card may show, so the
# card stays readable on a small phone.
MAX_SYMPTOMS = 3
MAX_CAUSES = 2
MAX_ACTIONS = 3
MAX_PREVENTION = 2


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def _text(
    text: str,
    size: str = T.SIZE_BODY,
    color: str = T.INK_900,
    weight: Optional[str] = None,
    wrap: bool = True,
    **extra: Any,
) -> Dict[str, Any]:
    node: Dict[str, Any] = {"type": "text", "text": text, "size": size,
                            "color": color, "wrap": wrap}
    if weight:
        node["weight"] = weight
    node.update(extra)
    return node


def _bullet(text: str) -> Dict[str, Any]:
    """A bullet row: a dot column plus wrapping text."""
    return {
        "type": "box",
        "layout": "baseline",
        "spacing": "sm",
        "contents": [
            {"type": "text", "text": "•", "size": T.SIZE_BODY,
             "color": T.HUSK_600, "flex": 0},
            {"type": "text", "text": text, "size": T.SIZE_BODY,
             "color": T.INK_900, "wrap": True, "flex": 1},
        ],
    }


def _section(heading: str, items: List[str], limit: int) -> List[Dict[str, Any]]:
    """A heading plus up to `limit` bullets. Empty list -> section omitted."""
    shown = [i for i in items if i][:limit]
    if not shown:
        return []
    return [
        {"type": "separator", "margin": "lg", "color": T.GREY_300},
        _text(heading, size=T.SIZE_HEADING, weight="bold", color=T.LEAF_700,
              margin="lg"),
        {"type": "box", "layout": "vertical", "margin": "sm", "spacing": "sm",
         "contents": [_bullet(i) for i in shown]},
    ]


def _confidence_bar(confidence: float, color: str) -> Dict[str, Any]:
    """A progress bar drawn from nested boxes (Flex has no native bar)."""
    pct = max(0, min(100, round(confidence * 100)))
    return {
        "type": "box",
        "layout": "vertical",
        "margin": "md",
        "spacing": "xs",
        "contents": [
            {
                "type": "box",
                "layout": "horizontal",
                "contents": [
                    _text("ความมั่นใจ", size=T.SIZE_CAPTION, color=T.GREY_600,
                          flex=0),
                    _text(f"{pct}%", size=T.SIZE_CAPTION, color=color,
                          weight="bold", align="end"),
                ],
            },
            {
                "type": "box",
                "layout": "vertical",
                "height": "12px",
                "backgroundColor": T.GREY_300,
                "cornerRadius": "6px",
                "contents": [
                    {
                        "type": "box",
                        "layout": "vertical",
                        "width": f"{max(pct, 2)}%",
                        "height": "12px",
                        "backgroundColor": color,
                        "cornerRadius": "6px",
                        "contents": [{"type": "filler"}],
                    }
                ],
            },
        ],
    }


def _disclaimer() -> Dict[str, Any]:
    return _text(M.DISCLAIMER, size=T.SIZE_CAPTION, color=T.GREY_600,
                 margin="lg")


def _header(title: str, subtitle: Optional[str], color: str,
            chip: Optional[str] = None) -> Dict[str, Any]:
    contents: List[Dict[str, Any]] = []
    if chip:
        contents.append({
            "type": "box",
            "layout": "vertical",
            "backgroundColor": "#FFFFFF33",
            "cornerRadius": "12px",
            "paddingAll": "4px",
            "paddingStart": "10px",
            "paddingEnd": "10px",
            "flex": 0,
            "contents": [_text(chip, size=T.SIZE_CAPTION, color=T.WHITE,
                               weight="bold")],
        })
    contents.append(_text(title, size=T.SIZE_TITLE, weight="bold",
                          color=T.WHITE, margin="sm"))
    if subtitle:
        contents.append(_text(subtitle, size=T.SIZE_SUBTITLE, color="#FFFFFFCC"))
    return {
        "type": "box",
        "layout": "vertical",
        "backgroundColor": color,
        "paddingAll": "16px",
        "spacing": "xs",
        "contents": contents,
    }


def _button(label: str, action: Dict[str, Any], style: str = "primary",
            color: Optional[str] = None) -> Dict[str, Any]:
    btn: Dict[str, Any] = {"type": "button", "style": style, "height": "sm",
                           "action": {**action, "label": label}}
    if color:
        btn["color"] = color
    return btn


def _footer_buttons(detail_url: Optional[str]) -> Dict[str, Any]:
    """Primary 'full detail' (only when a URL exists), then the two message
    actions that keep the farmer inside the chat."""
    contents: List[Dict[str, Any]] = []
    if detail_url:
        contents.append(_button(
            M.BTN_FULL_DETAIL, {"type": "uri", "uri": detail_url},
            style="primary", color=T.LEAF_700))
    contents.append(_button(
        M.BTN_NEW_PHOTO,
        {"type": "message", "text": M.KEYWORD_DIAGNOSE},
        style="secondary"))
    contents.append(_button(
        M.BTN_CONTACT_STAFF,
        {"type": "message", "text": M.KEYWORD_CONTACT},
        style="secondary"))
    return {"type": "box", "layout": "vertical", "spacing": "sm",
            "paddingAll": "12px", "contents": contents}


def _hero(image_url: Optional[str]) -> Optional[Dict[str, Any]]:
    if not image_url:
        return None
    return {
        "type": "image",
        "url": image_url,
        "size": "full",
        "aspectRatio": "4:3",
        "aspectMode": "cover",
    }


def _bubble(header: Dict[str, Any], body_contents: List[Dict[str, Any]],
            footer: Optional[Dict[str, Any]] = None,
            hero: Optional[Dict[str, Any]] = None,
            size: str = "mega") -> Dict[str, Any]:
    bubble: Dict[str, Any] = {
        "type": "bubble",
        "size": size,
        "header": header,
        "body": {"type": "box", "layout": "vertical", "paddingAll": "16px",
                 "backgroundColor": T.WHITE, "contents": body_contents},
    }
    if hero:
        bubble["hero"] = hero
    if footer:
        bubble["footer"] = footer
    return bubble


# --------------------------------------------------------------------------- #
# Quick replies
# --------------------------------------------------------------------------- #
def photo_quick_reply() -> Dict[str, Any]:
    return {
        "items": [
            {"type": "action",
             "action": {"type": "camera", "label": M.BTN_TAKE_PHOTO}},
            {"type": "action",
             "action": {"type": "cameraRoll", "label": M.BTN_PICK_PHOTO}},
        ]
    }


def default_quick_reply() -> Dict[str, Any]:
    return {
        "items": [
            {"type": "action",
             "action": {"type": "message", "label": "📷 ตรวจโรค",
                        "text": M.KEYWORD_DIAGNOSE}},
            {"type": "action",
             "action": {"type": "message", "label": "📚 คลังความรู้",
                        "text": M.KEYWORD_KNOWLEDGE}},
            {"type": "action",
             "action": {"type": "message", "label": "📸 วิธีถ่ายรูป",
                        "text": M.KEYWORD_HOW_TO_PHOTO}},
        ]
    }


# --------------------------------------------------------------------------- #
# Bubbles
# --------------------------------------------------------------------------- #
def build_how_to_photo_bubble(image_url: Optional[str] = None) -> Dict[str, Any]:
    body = [_text(M.HOW_TO_PHOTO_TITLE, size=T.SIZE_HEADING, weight="bold",
                  color=T.LEAF_700)]
    body.append({
        "type": "box", "layout": "vertical", "margin": "md", "spacing": "md",
        "contents": [
            {
                "type": "box", "layout": "baseline", "spacing": "sm",
                "contents": [
                    {"type": "text", "text": f"{i}.", "size": T.SIZE_BODY,
                     "color": T.DURIAN_500, "weight": "bold", "flex": 0},
                    {"type": "text", "text": tip, "size": T.SIZE_BODY,
                     "color": T.INK_900, "wrap": True, "flex": 1},
                ],
            }
            for i, tip in enumerate(M.HOW_TO_PHOTO_TIPS, start=1)
        ],
    })
    body.append(_text(M.HOW_TO_PHOTO_FOOTER, size=T.SIZE_BODY,
                      color=T.GREY_600, margin="lg"))
    return _bubble(
        header=_header("ถ่ายรูปยังไงให้หมอดูออก", None, T.LEAF_700),
        body_contents=body,
        hero=_hero(image_url),
    )


def build_result_bubble(disease: Disease, confidence: float,
                        image_url: Optional[str] = None,
                        detail_url: Optional[str] = None) -> Dict[str, Any]:
    """The full diagnosis card for a confident prediction."""
    color = T.severity_color(disease.severity)
    subtitle = " · ".join(x for x in (disease.name_en, disease.pathogen) if x)
    header = _header(disease.name_th, subtitle or None, color,
                     chip=T.severity_label_th(disease.severity))

    body: List[Dict[str, Any]] = [_confidence_bar(confidence, color)]

    if disease.pending_expert_input:
        body.append(_text(M.PENDING_DISEASE_NOTICE, margin="lg",
                          color=T.INK_900))
    elif disease.is_healthy:
        body.append(_text(M.RESULT_HEALTHY_BODY, margin="lg"))
        body += _section(M.HEADING_CARE_TIPS, disease.prevention, MAX_PREVENTION)
    else:
        body += _section(M.HEADING_SYMPTOMS, disease.symptoms, MAX_SYMPTOMS)
        body += _section(M.HEADING_CAUSES, disease.causes_conditions, MAX_CAUSES)
        body += _section(M.HEADING_ACTIONS, disease.immediate_actions,
                         MAX_ACTIONS)
        body += _section(M.HEADING_PREVENTION, disease.prevention,
                         MAX_PREVENTION)

    body.append(_disclaimer())
    return _bubble(header, body, _footer_buttons(detail_url), _hero(image_url))


def build_uncertain_bubble(primary: Disease, primary_confidence: float,
                           secondary: Optional[Disease],
                           secondary_confidence: Optional[float],
                           image_url: Optional[str] = None,
                           detail_url: Optional[str] = None) -> Dict[str, Any]:
    """The 'อาจเป็น...' card: shows top-2 side by side for comparison."""
    color = T.DURIAN_500
    header = _header(M.RESULT_UNCERTAIN_TITLE, None, color, chip="ยังไม่ชัด")

    body: List[Dict[str, Any]] = [
        _text(primary.name_th, size=T.SIZE_HEADING, weight="bold",
              color=T.LEAF_700),
        _confidence_bar(primary_confidence, color),
    ]
    if secondary and secondary_confidence is not None:
        body += [
            {"type": "separator", "margin": "lg", "color": T.GREY_300},
            _text("หรืออาจเป็น", size=T.SIZE_CAPTION, color=T.GREY_600,
                  margin="lg"),
            _text(secondary.name_th, size=T.SIZE_HEADING, weight="bold",
                  color=T.LEAF_700),
            _confidence_bar(secondary_confidence, T.GREY_600),
        ]

    body += _section(M.HEADING_SYMPTOMS, primary.symptoms, MAX_SYMPTOMS)
    body.append(_text(M.RESULT_UNCERTAIN_HINT, margin="lg", color=T.INK_900))
    body.append(_disclaimer())
    return _bubble(header, body, _footer_buttons(detail_url), _hero(image_url))


def build_notice_bubble(title: str, body_text: str, color: str,
                        image_url: Optional[str] = None,
                        detail_url: Optional[str] = None) -> Dict[str, Any]:
    """Used for 'ยังไม่แน่ใจ' and 'ไม่ใช่รูปทุเรียน'."""
    body = [_text(body_text), _disclaimer()]
    return _bubble(_header(title, None, color), body,
                   _footer_buttons(detail_url), _hero(image_url))


def build_not_confident_bubble(image_url: Optional[str] = None) -> Dict[str, Any]:
    return build_notice_bubble(M.RESULT_NOT_CONFIDENT_TITLE,
                               M.RESULT_NOT_CONFIDENT_BODY, T.GREY_600,
                               image_url)


def build_not_durian_bubble(image_url: Optional[str] = None) -> Dict[str, Any]:
    return build_notice_bubble(M.RESULT_NOT_DURIAN_TITLE,
                               M.RESULT_NOT_DURIAN_BODY, T.HUSK_600, image_url)


def build_model_not_ready_bubble() -> Dict[str, Any]:
    return build_notice_bubble("ระบบยังไม่พร้อมครับ",
                               M.ERROR_MODEL_NOT_READY, T.GREY_600)


def _disease_card(disease: Disease, web_base_url: str) -> Dict[str, Any]:
    color = T.severity_color(disease.severity)
    subtitle = disease.name_en or ""
    if disease.pending_expert_input:
        subtitle = "ข้อมูลกำลังจัดทำ"
    return _bubble(
        header=_header(disease.name_th, subtitle or None, color),
        body_contents=[
            _text(
                disease.symptoms[0] if disease.symptoms
                else M.PENDING_DISEASE_NOTICE,
                color=T.INK_900,
            )
        ],
        footer={
            "type": "box", "layout": "vertical", "paddingAll": "12px",
            "contents": [_button(
                M.BTN_READ_DISEASE,
                {"type": "uri",
                 "uri": f"{web_base_url.rstrip('/')}/diseases/{disease.slug}"},
                style="primary", color=T.LEAF_700)],
        },
        size="kilo",
    )


def build_knowledge_carousel(diseases: List[Disease],
                             web_base_url: str) -> Dict[str, Any]:
    """Carousel of disease cards. Truncated to the platform's bubble cap."""
    cards = [_disease_card(d, web_base_url)
             for d in diseases[:MAX_CAROUSEL_BUBBLES]]
    return {"type": "carousel", "contents": cards}


# --------------------------------------------------------------------------- #
# Validation helpers (used by tests and by the sample exporter)
# --------------------------------------------------------------------------- #
def container_size_bytes(container: Dict[str, Any]) -> int:
    return len(json.dumps(container, ensure_ascii=False).encode("utf-8"))


def validate_container(container: Dict[str, Any]) -> None:
    """Raise if a container would be rejected by the LINE platform."""
    kind = container.get("type")
    size = container_size_bytes(container)
    if kind == "bubble" and size > MAX_BUBBLE_BYTES:
        raise ValueError(f"bubble is {size} bytes (limit {MAX_BUBBLE_BYTES})")
    if kind == "carousel":
        if size > MAX_CAROUSEL_BYTES:
            raise ValueError(
                f"carousel is {size} bytes (limit {MAX_CAROUSEL_BYTES})")
        n = len(container.get("contents", []))
        if n > MAX_CAROUSEL_BUBBLES:
            raise ValueError(
                f"carousel has {n} bubbles (limit {MAX_CAROUSEL_BUBBLES})")
