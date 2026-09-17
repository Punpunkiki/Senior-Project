"""app/line_handler.py -- LINE event routing and the conversation state machine.

Flow for a photo (the main path):
  webhook returns 200 immediately  ->  background task runs this module
  -> show a loading animation      (immediate visible feedback)
  -> download the image, run inference, store the diagnosis
  -> push the result Flex card     (push, not reply: the reply token may
                                    already be stale after inference)

Logging never records message content or raw LINE user ids -- only a short
salted-looking prefix hash, enough to correlate a session in the logs.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from linebot.v3.messaging import (ApiClient, Configuration, FlexMessage,
                                  FlexContainer, MessagingApi,
                                  MessagingApiBlob, PushMessageRequest,
                                  ReplyMessageRequest,
                                  ShowLoadingAnimationRequest, TextMessage)
from linebot.v3.webhooks import (FollowEvent, ImageMessageContent, MessageEvent,
                                 TextMessageContent, UnfollowEvent)
from PIL import Image

from app import messages as M
from app.config import Settings
from app.db import get_session, record_diagnosis, upsert_user
from app.diagnosis import DiagnosisTier, classify
from app.flex import builders as B
from app.knowledge import KnowledgeBase
from app.ml.predictor import DurianPredictor
from app.state import ConversationState, StateStore
from src.utils import get_logger

log = get_logger("line_handler")

# LINE accepts 5-60 seconds, in multiples of 5.
LOADING_SECONDS = 10


def anon(user_id: str) -> str:
    """Short non-reversible tag for logs -- never log the raw user id."""
    return hashlib.sha256(user_id.encode()).hexdigest()[:8]


@dataclass
class OutgoingFlex:
    alt_text: str
    container: Dict[str, Any]
    quick_reply: Optional[Dict[str, Any]] = None


class LineService:
    """Thin wrapper over the v3 Messaging API + the bot's routing rules."""

    def __init__(self, settings: Settings, predictor: DurianPredictor,
                 kb: KnowledgeBase, state: StateStore):
        self.settings = settings
        self.predictor = predictor
        self.kb = kb
        self.state = state
        self._config = Configuration(
            access_token=settings.line_channel_access_token)

    # --- transport -------------------------------------------------------
    def _flex_message(self, out: OutgoingFlex) -> FlexMessage:
        msg = FlexMessage(
            alt_text=out.alt_text,
            contents=FlexContainer.from_dict(out.container),
        )
        if out.quick_reply:
            msg.quick_reply = out.quick_reply
        return msg

    def _to_sdk(self, payload: Any) -> Any:
        if isinstance(payload, OutgoingFlex):
            return self._flex_message(payload)
        if isinstance(payload, str):
            return TextMessage(text=payload)
        return payload

    def reply(self, reply_token: str, payloads: List[Any]) -> None:
        with ApiClient(self._config) as client:
            MessagingApi(client).reply_message(
                ReplyMessageRequest(
                    reply_token=reply_token,
                    messages=[self._to_sdk(p) for p in payloads],
                )
            )

    def push(self, user_id: str, payloads: List[Any]) -> None:
        with ApiClient(self._config) as client:
            MessagingApi(client).push_message(
                PushMessageRequest(
                    to=user_id,
                    messages=[self._to_sdk(p) for p in payloads],
                )
            )

    def show_loading(self, user_id: str) -> None:
        try:
            with ApiClient(self._config) as client:
                MessagingApi(client).show_loading_animation(
                    ShowLoadingAnimationRequest(
                        chat_id=user_id, loading_seconds=LOADING_SECONDS)
                )
        except Exception as exc:  # cosmetic only -- never block the diagnosis
            log.warning("loading animation failed for %s: %s",
                        anon(user_id), exc)

    def download_image(self, message_id: str) -> bytes:
        with ApiClient(self._config) as client:
            return MessagingApiBlob(client).get_message_content(message_id)

    # --- routing ---------------------------------------------------------
    def handle_event(self, event: Any) -> None:
        if isinstance(event, FollowEvent):
            self.handle_follow(event)
        elif isinstance(event, UnfollowEvent):
            self.handle_unfollow(event)
        elif isinstance(event, MessageEvent):
            if isinstance(event.message, TextMessageContent):
                self.handle_text(event)
            elif isinstance(event.message, ImageMessageContent):
                self.handle_image(event)

    def handle_follow(self, event: FollowEvent) -> None:
        """Record the user but stay silent: the greeting message is configured
        in LINE OA Manager, and replying here would double up."""
        user_id = event.source.user_id
        with get_session() as session:
            upsert_user(session, user_id)
        log.info("follow: %s", anon(user_id))

    def handle_unfollow(self, event: UnfollowEvent) -> None:
        user_id = event.source.user_id
        with get_session() as session:
            user = upsert_user(session, user_id)
            user.blocked = True
            session.commit()
        self.state.clear(user_id)
        log.info("unfollow: %s", anon(user_id))

    def handle_text(self, event: MessageEvent) -> None:
        user_id = event.source.user_id
        text = (event.message.text or "").strip()
        with get_session() as session:
            upsert_user(session, user_id)

        if text == M.KEYWORD_DIAGNOSE:
            self.state.set(user_id, ConversationState.WAITING_IMAGE)
            self.reply(event.reply_token, [OutgoingFlex(
                alt_text=M.HOW_TO_PHOTO_TITLE,
                container=B.build_how_to_photo_bubble(),
                quick_reply=B.photo_quick_reply(),
            )])
        elif text == M.KEYWORD_HOW_TO_PHOTO:
            self.reply(event.reply_token, [OutgoingFlex(
                alt_text=M.HOW_TO_PHOTO_TITLE,
                container=B.build_how_to_photo_bubble(),
                quick_reply=B.photo_quick_reply(),
            )])
        elif text == M.KEYWORD_KNOWLEDGE:
            self.reply(event.reply_token, [OutgoingFlex(
                alt_text=M.KNOWLEDGE_TITLE,
                container=B.build_knowledge_carousel(
                    [d for d in self.kb.all() if not d.is_healthy],
                    self.settings.web_base_url),
            )])
        elif text == M.KEYWORD_CONTACT:
            self.reply(event.reply_token, [self.settings.admin_contact_text])
        else:
            msg = TextMessage(text=M.FALLBACK_TEXT)
            msg.quick_reply = B.default_quick_reply()
            self.reply(event.reply_token, [msg])

    def handle_image(self, event: MessageEvent) -> None:
        """Accepted in any state, per spec -- a farmer who just sends a photo
        without pressing the menu should still get a diagnosis."""
        user_id = event.source.user_id
        self.show_loading(user_id)
        try:
            self.process_image(user_id, event.message.id)
        except Exception:
            log.exception("diagnosis failed for %s", anon(user_id))
            self.push(user_id, [M.ERROR_BUSY])
        finally:
            self.state.clear(user_id)

    # --- the diagnosis itself --------------------------------------------
    def process_image(self, user_id: str, message_id: str) -> None:
        if not self.predictor.ready:
            log.error("predictor not ready -- no checkpoint loaded")
            self.push(user_id, [OutgoingFlex(
                alt_text=M.ERROR_MODEL_NOT_READY,
                container=B.build_model_not_ready_bubble(),
            )])
            return

        raw = self.download_image(message_id)
        max_bytes = int(self.settings.max_image_mb * 1024 * 1024)
        if len(raw) > max_bytes:
            self.push(user_id, [M.ERROR_IMAGE_TOO_LARGE])
            return

        image = Image.open(io.BytesIO(raw)).convert("RGB")
        image_path = self._save_image(user_id, message_id, image)

        top3 = self.predictor.predict(image, top_k=3)
        diagnosis = classify(top3, self.settings.confidence_high,
                             self.settings.confidence_low)

        with get_session() as session:
            upsert_user(session, user_id)
            record = record_diagnosis(
                session, user_id,
                [{"class": s.label, "confidence": s.confidence} for s in top3],
                diagnosis.tier.value, self.predictor.model_name,
                str(image_path) if image_path else None,
            )

        log.info("diagnosis %s user=%s tier=%s conf=%.3f",
                 record.id, anon(user_id), diagnosis.tier.value,
                 diagnosis.top1.confidence)
        self.push(user_id, [self._build_result(diagnosis, record.id)])

    def _save_image(self, user_id: str, message_id: str,
                    image: Image.Image) -> Optional[Path]:
        try:
            out_dir = Path(self.settings.upload_dir) / anon(user_id)
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{message_id}.jpg"
            image.save(path, format="JPEG", quality=88)
            return path
        except Exception as exc:
            # Storage is a nice-to-have (it powers the LIFF page); a failure
            # here must not cost the farmer their diagnosis.
            log.warning("could not store image for %s: %s", anon(user_id), exc)
            return None

    def _detail_url(self, diagnosis_id: str) -> str:
        """LIFF forwards its query string to the endpoint, so both forms land
        on the same static /result page with ?dg=<id>."""
        if self.settings.liff_id:
            return (f"https://liff.line.me/{self.settings.liff_id}"
                    f"?dg={diagnosis_id}")
        return (f"{self.settings.web_base_url.rstrip('/')}"
                f"/result/?dg={diagnosis_id}")

    def _build_result(self, diagnosis, diagnosis_id: str) -> OutgoingFlex:
        detail_url = self._detail_url(diagnosis_id)

        if diagnosis.tier is DiagnosisTier.NOT_DURIAN:
            return OutgoingFlex(M.RESULT_NOT_DURIAN_TITLE,
                                B.build_not_durian_bubble())
        if diagnosis.tier is DiagnosisTier.NOT_CONFIDENT:
            return OutgoingFlex(M.RESULT_NOT_CONFIDENT_TITLE,
                                B.build_not_confident_bubble())

        primary = self.kb.by_class(diagnosis.top1.label)
        if primary is None:
            # A class the knowledge base does not cover: never invent content.
            log.error("no KB entry for class %s", diagnosis.top1.label)
            return OutgoingFlex(M.RESULT_NOT_CONFIDENT_TITLE,
                                B.build_not_confident_bubble())

        if diagnosis.tier is DiagnosisTier.UNCERTAIN_TOP2:
            secondary = (self.kb.by_class(diagnosis.top2.label)
                         if diagnosis.top2 else None)
            return OutgoingFlex(
                f"{M.RESULT_UNCERTAIN_TITLE} {primary.name_th}",
                B.build_uncertain_bubble(
                    primary, diagnosis.top1.confidence,
                    secondary,
                    diagnosis.top2.confidence if (secondary and diagnosis.top2)
                    else None,
                    detail_url=detail_url),
            )

        alt = (M.RESULT_HEALTHY_TITLE if primary.is_healthy
               else f"{M.RESULT_CONFIDENT_PREFIX}{primary.name_th}")
        return OutgoingFlex(
            alt,
            B.build_result_bubble(primary, diagnosis.top1.confidence,
                                  detail_url=detail_url),
        )
