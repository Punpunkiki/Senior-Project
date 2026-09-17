"""app/main.py -- FastAPI service for the "หมอทุเรียน" LINE OA.

Endpoints:
  POST /webhook              LINE events. Verifies X-Line-Signature, then
                             returns 200 immediately and does the slow work
                             (image download + inference) in a background
                             task, because LINE retries slow webhooks.
  GET  /api/diseases         knowledge base index (for the website)
  GET  /api/diseases/{slug}  one disease
  GET  /api/diagnoses/{id}   one diagnosis (for the LIFF result page)
  GET  /health               liveness + whether the model is actually loaded
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhook import WebhookParser

from app.auth import bearer_token, verify_id_token
from app.config import settings
from app.db import get_diagnosis, get_session, init_db
from app.knowledge import load_knowledge_base
from app.line_handler import LineService
from app.ml.predictor import get_predictor
from app.state import get_state_store
from src.utils import get_logger

log = get_logger("main")

_service: LineService | None = None
_parser: WebhookParser | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service, _parser

    init_db(settings.database_url)
    kb = load_knowledge_base()
    predictor = get_predictor(settings.config_yaml_path, settings.model_name)

    missing = kb.missing_for(predictor.classes)
    if missing:
        # Loud, but not fatal: the bot degrades to "ยังไม่แน่ใจ" for an
        # uncovered class rather than inventing advice for it.
        log.error("knowledge base is missing entries for: %s", missing)
    pending = [d.class_name for d in kb.pending()]
    if pending:
        log.warning("classes still awaiting expert content: %s", pending)
    if not predictor.ready:
        log.error("MODEL NOT LOADED -- the bot will reply 'ระบบยังไม่พร้อม' "
                  "to every photo until outputs/%s/best.pt exists",
                  settings.model_name)
    if not settings.liff_channel_id:
        log.warning("LIFF_CHANNEL_ID is not set -- /api/diagnoses/{id} cannot "
                    "verify who is asking, so anyone holding a result link can "
                    "read it. Set it before serving real users.")

    _parser = WebhookParser(settings.line_channel_secret)
    _service = LineService(settings, predictor, kb, get_state_store())
    yield


app = FastAPI(title="หมอทุเรียน API", lifespan=lifespan)


def service() -> LineService:
    if _service is None:
        raise HTTPException(status_code=503, detail="service not ready")
    return _service


@app.get("/health")
def health() -> Dict[str, Any]:
    predictor = get_predictor(settings.config_yaml_path, settings.model_name)
    return {
        "status": "ok",
        "model_name": settings.model_name,
        "model_loaded": predictor.ready,
        "temperature": predictor.temperature,
    }


@app.post("/webhook")
async def webhook(request: Request, background: BackgroundTasks) -> JSONResponse:
    signature = request.headers.get("X-Line-Signature", "")
    body = (await request.body()).decode("utf-8")

    if _parser is None:
        raise HTTPException(status_code=503, detail="service not ready")
    try:
        events = _parser.parse(body, signature)
    except InvalidSignatureError:
        # Someone POSTing without the channel secret. Do not leak detail.
        log.warning("rejected webhook with invalid signature")
        raise HTTPException(status_code=400, detail="invalid signature")

    # Return 200 now; do the slow work after the response is sent.
    for event in events:
        background.add_task(_safe_handle, event)
    return JSONResponse({"ok": True})


def _safe_handle(event: Any) -> None:
    """A crash in one event must not take down the worker or the others."""
    try:
        service().handle_event(event)
    except Exception:
        log.exception("unhandled error while processing an event")


@app.get("/api/diseases")
def list_diseases() -> Dict[str, List[Dict[str, Any]]]:
    kb = load_knowledge_base()
    return {"diseases": [asdict(d) for d in kb.all()]}


@app.get("/api/diseases/{slug}")
def get_disease(slug: str) -> Dict[str, Any]:
    kb = load_knowledge_base()
    disease = kb.by_slug(slug)
    if disease is None:
        raise HTTPException(status_code=404, detail="disease not found")
    return asdict(disease)


@app.get("/api/diagnoses/{diagnosis_id}")
def read_diagnosis(diagnosis_id: str,
                   authorization: str | None = Header(default=None)
                   ) -> Dict[str, Any]:
    """A diagnosis is readable only by the LINE user who created it.

    The LIFF page sends that user's ID token; we ask LINE who it belongs to and
    compare. Without `liff_channel_id` configured the check cannot run -- the
    service warns loudly at startup in that case.
    """
    with get_session() as session:
        record = get_diagnosis(session, diagnosis_id)
        if record is None:
            raise HTTPException(status_code=404, detail="diagnosis not found")

        if settings.liff_channel_id:
            token = bearer_token(authorization)
            if not token:
                raise HTTPException(status_code=401,
                                    detail="LINE ID token required")
            viewer = verify_id_token(token, settings.liff_channel_id)
            if viewer is None:
                raise HTTPException(status_code=401, detail="invalid ID token")
            if viewer != record.line_user_id:
                log.warning("blocked cross-user read of diagnosis %s",
                            diagnosis_id)
                raise HTTPException(status_code=403, detail="not your diagnosis")

        payload = record.to_public_dict()

    kb = load_knowledge_base()
    top1 = payload["top3"][0]["class"] if payload["top3"] else None
    disease = kb.by_class(top1) if top1 else None
    payload["disease"] = asdict(disease) if disease else None
    return payload


# --------------------------------------------------------------------------- #
# Static site (must be mounted LAST so it cannot shadow /api or /webhook).
# `npm run build` in web/ produces web/out; serving it from this same origin
# means the LIFF page needs no CORS configuration and there is one deployment.
# --------------------------------------------------------------------------- #
_WEB_DIR = Path(__file__).resolve().parent.parent / "web" / "out"
if _WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=_WEB_DIR, html=True), name="web")
else:
    log.warning("No static site at %s -- API only. Run `npm run build` in web/ "
                "to generate it.", _WEB_DIR)
