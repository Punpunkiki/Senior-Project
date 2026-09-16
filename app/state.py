"""app/state.py -- per-user conversation state with a TTL.

Deliberately tiny: the bot only needs to know whether it is waiting for a
photo. State is in-process and expires after `ttl_seconds` (10 min by
spec), so a farmer who wanders off mid-flow is not stuck in `waiting_image`
forever.

Note for deployment: an in-memory store means state is per-process and is
lost on restart. That is acceptable because an expired/absent state simply
falls back to `IDLE`, and images are accepted in any state anyway. If the
service is ever scaled to multiple instances and state starts mattering
more, move this behind the same interface backed by the DB or Redis.
"""
from __future__ import annotations

import threading
import time
from enum import Enum
from typing import Dict, Optional, Tuple

DEFAULT_TTL_SECONDS = 10 * 60


class ConversationState(str, Enum):
    IDLE = "idle"
    WAITING_IMAGE = "waiting_image"


class StateStore:
    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS,
                 clock=time.monotonic):
        self._ttl = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._states: Dict[str, Tuple[ConversationState, float]] = {}

    def set(self, user_id: str, state: ConversationState) -> None:
        with self._lock:
            self._states[user_id] = (state, self._clock())

    def get(self, user_id: str) -> ConversationState:
        with self._lock:
            entry = self._states.get(user_id)
            if entry is None:
                return ConversationState.IDLE
            state, stamp = entry
            if self._clock() - stamp > self._ttl:
                del self._states[user_id]
                return ConversationState.IDLE
            return state

    def clear(self, user_id: str) -> None:
        with self._lock:
            self._states.pop(user_id, None)

    def purge_expired(self) -> int:
        """Drop expired entries so an idle process doesn't grow forever."""
        now = self._clock()
        with self._lock:
            stale = [k for k, (_, s) in self._states.items()
                     if now - s > self._ttl]
            for k in stale:
                del self._states[k]
        return len(stale)


_store: Optional[StateStore] = None


def get_state_store(ttl_seconds: int = DEFAULT_TTL_SECONDS) -> StateStore:
    global _store
    if _store is None:
        _store = StateStore(ttl_seconds)
    return _store
