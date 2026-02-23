"""Unit tests for the notification system: router, events, config."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from apollobot.notifications.config import ChannelConfig, NotificationsConfig
from apollobot.notifications.events import (
    EventSeverity,
    EventType,
    NotificationEvent,
)
from apollobot.notifications.channel import NotificationChannel
from apollobot.notifications.router import NotificationRouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockChannel(NotificationChannel):
    """In-memory mock channel for testing."""

    def __init__(self, name: str = "mock", supports_responses: bool = False):
        self.name = name
        self.supports_responses = supports_responses
        self.sent: list[NotificationEvent] = []
        self._response: bool = True

    async def send(self, event: NotificationEvent) -> None:
        self.sent.append(event)

    async def wait_for_response(self, event: NotificationEvent, timeout: float = 3600) -> bool:
        return self._response


def _make_event(
    event_type: EventType = EventType.PHASE_COMPLETED,
    **kwargs,
) -> NotificationEvent:
    defaults = {
        "session_id": "test-session",
        "title": "Test event",
        "summary": "Something happened",
    }
    defaults.update(kwargs)
    return NotificationEvent(event_type=event_type, **defaults)


# ---------------------------------------------------------------------------
# Event model tests
# ---------------------------------------------------------------------------


class TestNotificationEvent:
    def test_create_event(self):
        event = _make_event()
        assert event.event_type == EventType.PHASE_COMPLETED
        assert event.session_id == "test-session"
        assert event.severity == EventSeverity.INFO
        assert event.requires_response is False
        assert event.timestamp  # auto-generated

    def test_event_with_details(self):
        event = _make_event(
            details={"cost": 1.23, "papers": 42},
            phase="literature_review",
        )
        assert event.details["cost"] == 1.23
        assert event.phase == "literature_review"

    def test_checkpoint_event(self):
        event = _make_event(
            event_type=EventType.CHECKPOINT_APPROVAL,
            severity=EventSeverity.WARNING,
            requires_response=True,
        )
        assert event.requires_response is True
        assert event.severity == EventSeverity.WARNING


# ---------------------------------------------------------------------------
# Config model tests
# ---------------------------------------------------------------------------


class TestNotificationsConfig:
    def test_defaults(self):
        cfg = NotificationsConfig()
        assert cfg.enabled is False
        assert cfg.heartbeat_interval == 300
        assert cfg.channels == []

    def test_channel_config_extras(self):
        cfg = ChannelConfig(
            type="telegram",
            token="bot123:ABC",
            chat_id="12345",
            events=["session_completed", "finding"],
        )
        assert cfg.type == "telegram"
        assert cfg.token == "bot123:ABC"  # type: ignore[attr-defined]
        assert cfg.chat_id == "12345"  # type: ignore[attr-defined]
        assert "session_completed" in cfg.events

    def test_wildcard_events(self):
        cfg = ChannelConfig(type="webhook", url="https://example.com")
        assert cfg.events == ["*"]


# ---------------------------------------------------------------------------
# Router tests
# ---------------------------------------------------------------------------


class TestNotificationRouter:
    @pytest.mark.asyncio
    async def test_dispatch_to_all_channels(self):
        router = NotificationRouter()
        ch1 = MockChannel("ch1")
        ch2 = MockChannel("ch2")
        router.register(ch1)
        router.register(ch2)

        event = _make_event()
        await router.dispatch(event)

        assert len(ch1.sent) == 1
        assert len(ch2.sent) == 1

    @pytest.mark.asyncio
    async def test_dispatch_with_filter(self):
        router = NotificationRouter()
        ch1 = MockChannel("ch1")
        ch2 = MockChannel("ch2")
        router.register(ch1, events=["session_completed"])
        router.register(ch2, events=["*"])

        event = _make_event(event_type=EventType.PHASE_COMPLETED)
        await router.dispatch(event)

        assert len(ch1.sent) == 0  # filtered out
        assert len(ch2.sent) == 1  # wildcard matches

    @pytest.mark.asyncio
    async def test_dispatch_matching_filter(self):
        router = NotificationRouter()
        ch = MockChannel("ch")
        router.register(ch, events=["phase_completed", "session_started"])

        await router.dispatch(_make_event(event_type=EventType.PHASE_COMPLETED))
        await router.dispatch(_make_event(event_type=EventType.SESSION_STARTED))
        await router.dispatch(_make_event(event_type=EventType.HEARTBEAT))

        assert len(ch.sent) == 2

    @pytest.mark.asyncio
    async def test_approval_with_bidirectional(self):
        router = NotificationRouter()
        ch = MockChannel("bidir", supports_responses=True)
        ch._response = True
        router.register(ch)

        event = _make_event(
            event_type=EventType.CHECKPOINT_APPROVAL,
            requires_response=True,
        )
        result = await router.request_approval(event)
        assert result is True

    @pytest.mark.asyncio
    async def test_approval_deny(self):
        router = NotificationRouter()
        ch = MockChannel("bidir", supports_responses=True)
        ch._response = False
        router.register(ch)

        event = _make_event(
            event_type=EventType.CHECKPOINT_APPROVAL,
            requires_response=True,
        )
        result = await router.request_approval(event)
        assert result is False

    @pytest.mark.asyncio
    async def test_approval_no_bidirectional_auto_approves(self):
        router = NotificationRouter()
        ch = MockChannel("notify_only", supports_responses=False)
        router.register(ch)

        event = _make_event(
            event_type=EventType.CHECKPOINT_APPROVAL,
            requires_response=True,
        )
        result = await router.request_approval(event)
        assert result is True

    @pytest.mark.asyncio
    async def test_channel_error_doesnt_break_others(self):
        router = NotificationRouter()

        class FailingChannel(NotificationChannel):
            name = "failing"
            supports_responses = False

            async def send(self, event):
                raise RuntimeError("boom")

        ch_ok = MockChannel("ok")
        router.register(FailingChannel())
        router.register(ch_ok)

        event = _make_event()
        await router.dispatch(event)  # should not raise

        assert len(ch_ok.sent) == 1

    @pytest.mark.asyncio
    async def test_connect_disconnect(self):
        router = NotificationRouter()
        ch = MockChannel("ch")
        ch.connect = AsyncMock()
        ch.disconnect = AsyncMock()
        router.register(ch)

        await router.connect_all()
        ch.connect.assert_awaited_once()

        await router.disconnect_all()
        ch.disconnect.assert_awaited_once()
