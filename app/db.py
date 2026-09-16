"""app/db.py -- SQLite storage for users and diagnoses.

Kept to plain SQLAlchemy Core/ORM with a URL from settings so swapping to
Postgres is a connection-string change, not a rewrite.

Privacy: we store the LINE user id (needed to push results back) and the
diagnosis, nothing else. No display name, no profile picture, no location.
Images are written to disk and reaped after `image_retention_days`
(scripts/reap_images.py).
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import (JSON, Boolean, DateTime, Float, ForeignKey, String,
                        create_engine, delete, select)
from sqlalchemy.orm import (DeclarativeBase, Mapped, Session, mapped_column,
                            sessionmaker)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    line_user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    blocked: Mapped[bool] = mapped_column(Boolean, default=False)


class Diagnosis(Base):
    __tablename__ = "diagnoses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    line_user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.line_user_id"), index=True)
    image_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    top3: Mapped[Any] = mapped_column(JSON)
    tier: Mapped[str] = mapped_column(String(32))
    top1_class: Mapped[str] = mapped_column(String(64))
    top1_confidence: Mapped[float] = mapped_column(Float)
    model_name: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, index=True)

    def to_public_dict(self) -> Dict[str, Any]:
        """Shape returned by GET /api/diagnoses/{id} (LIFF result page).
        Deliberately omits line_user_id and the on-disk image path."""
        top3 = self.top3 if isinstance(self.top3, list) else json.loads(self.top3)
        return {
            "id": self.id,
            "tier": self.tier,
            "top3": top3,
            "model_name": self.model_name,
            "created_at": self.created_at.isoformat(),
        }


_engine = None
_SessionLocal = None


def init_db(database_url: str) -> None:
    global _engine, _SessionLocal
    connect_args = {"check_same_thread": False} if database_url.startswith(
        "sqlite") else {}
    _engine = create_engine(database_url, connect_args=connect_args,
                            future=True)
    Base.metadata.create_all(_engine)
    _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False,
                                 future=True)


def get_session() -> Session:
    if _SessionLocal is None:
        raise RuntimeError("init_db() must be called before get_session()")
    return _SessionLocal()


# --------------------------------------------------------------------------- #
# Operations
# --------------------------------------------------------------------------- #
def upsert_user(session: Session, line_user_id: str) -> User:
    user = session.get(User, line_user_id)
    if user is None:
        user = User(line_user_id=line_user_id)
        session.add(user)
    else:
        user.last_seen_at = _utcnow()
    session.commit()
    return user


def record_diagnosis(session: Session, line_user_id: str, top3: List[dict],
                     tier: str, model_name: str,
                     image_path: Optional[str] = None) -> Diagnosis:
    diagnosis = Diagnosis(
        id=str(uuid.uuid4()),
        line_user_id=line_user_id,
        image_path=image_path,
        top3=top3,
        tier=tier,
        top1_class=top3[0]["class"],
        top1_confidence=top3[0]["confidence"],
        model_name=model_name,
    )
    session.add(diagnosis)
    session.commit()
    return diagnosis


def get_diagnosis(session: Session, diagnosis_id: str) -> Optional[Diagnosis]:
    return session.get(Diagnosis, diagnosis_id)


def purge_old_images(session: Session, retention_days: int) -> int:
    """Delete image files older than the retention window and null out their
    paths. Diagnosis rows are kept (they carry no personal data beyond the
    user id) so the LIFF result link does not 404."""
    cutoff = _utcnow() - timedelta(days=retention_days)
    rows = session.execute(
        select(Diagnosis).where(Diagnosis.created_at < cutoff,
                                Diagnosis.image_path.is_not(None))
    ).scalars().all()
    removed = 0
    for row in rows:
        path = Path(row.image_path)
        if path.exists():
            path.unlink()
            removed += 1
        row.image_path = None
    session.commit()
    return removed
