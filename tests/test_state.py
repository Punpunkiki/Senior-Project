"""Conversation state machine: defaults, transitions, and TTL expiry."""
from __future__ import annotations

from app.state import ConversationState, StateStore

USER = "U123"


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_unknown_user_defaults_to_idle():
    store = StateStore(ttl_seconds=600)
    assert store.get(USER) is ConversationState.IDLE


def test_set_and_get():
    store = StateStore(ttl_seconds=600)
    store.set(USER, ConversationState.WAITING_IMAGE)
    assert store.get(USER) is ConversationState.WAITING_IMAGE


def test_state_expires_after_ttl():
    clock = FakeClock()
    store = StateStore(ttl_seconds=600, clock=clock)
    store.set(USER, ConversationState.WAITING_IMAGE)

    clock.advance(599)
    assert store.get(USER) is ConversationState.WAITING_IMAGE

    clock.advance(2)  # now past the 10-minute window
    assert store.get(USER) is ConversationState.IDLE


def test_clear_resets_to_idle():
    store = StateStore(ttl_seconds=600)
    store.set(USER, ConversationState.WAITING_IMAGE)
    store.clear(USER)
    assert store.get(USER) is ConversationState.IDLE


def test_purge_expired_drops_only_stale_entries():
    clock = FakeClock()
    store = StateStore(ttl_seconds=600, clock=clock)
    store.set("old", ConversationState.WAITING_IMAGE)
    clock.advance(601)
    store.set("fresh", ConversationState.WAITING_IMAGE)

    assert store.purge_expired() == 1
    assert store.get("fresh") is ConversationState.WAITING_IMAGE


def test_users_are_independent():
    store = StateStore(ttl_seconds=600)
    store.set("a", ConversationState.WAITING_IMAGE)
    assert store.get("b") is ConversationState.IDLE
