"""Integration tests for ChannelCheckpointHandler."""

import pytest

from apollobot.notifications.checkpoint import ChannelCheckpointHandler
from apollobot.notifications.channel import NotificationChannel
from apollobot.notifications.events import EventType, NotificationEvent
from apollobot.notifications.router import NotificationRouter


class RecordingChannel(NotificationChannel):
    """Records all events for assertion."""

    def __init__(self, name: str = "recorder", approve: bool = True):
        self.name = name
        self.supports_responses = True
        self.sent: list[NotificationEvent] = []
        self._approve = approve

    async def send(self, event: NotificationEvent) -> None:
        self.sent.append(event)

    async def wait_for_response(self, event: NotificationEvent, timeout: float = 3600) -> bool:
        return self._approve


class TestChannelCheckpointHandler:
    @pytest.mark.asyncio
    async def test_notify_dispatches_event(self):
        router = NotificationRouter()
        ch = RecordingChannel()
        router.register(ch)

        handler = ChannelCheckpointHandler(router, session_id="sess-001")
        await handler.notify("analysis", "Completed 5 analysis steps")

        assert len(ch.sent) == 1
        event = ch.sent[0]
        assert event.event_type == EventType.PHASE_COMPLETED
        assert event.session_id == "sess-001"
        assert event.phase == "analysis"
        assert "Completed 5 analysis steps" in event.summary

    @pytest.mark.asyncio
    async def test_request_approval_approved(self):
        router = NotificationRouter()
        ch = RecordingChannel(approve=True)
        router.register(ch)

        handler = ChannelCheckpointHandler(router, session_id="sess-002")
        result = await handler.request_approval("plan", "Proceed with research?")

        assert result is True
        assert len(ch.sent) == 1
        event = ch.sent[0]
        assert event.event_type == EventType.CHECKPOINT_APPROVAL
        assert event.requires_response is True

    @pytest.mark.asyncio
    async def test_request_approval_denied(self):
        router = NotificationRouter()
        ch = RecordingChannel(approve=False)
        router.register(ch)

        handler = ChannelCheckpointHandler(router, session_id="sess-003")
        result = await handler.request_approval("plan", "Proceed?")

        assert result is False

    @pytest.mark.asyncio
    async def test_multiple_channels(self):
        router = NotificationRouter()
        ch1 = RecordingChannel("ch1", approve=True)
        ch2 = RecordingChannel("ch2", approve=False)
        router.register(ch1)
        router.register(ch2)

        handler = ChannelCheckpointHandler(router, session_id="sess-004")
        await handler.notify("literature_review", "Found 42 papers")

        # Both channels should receive the notification
        assert len(ch1.sent) == 1
        assert len(ch2.sent) == 1

    @pytest.mark.asyncio
    async def test_session_id_in_events(self):
        router = NotificationRouter()
        ch = RecordingChannel()
        router.register(ch)

        handler = ChannelCheckpointHandler(router, session_id="my-session")
        await handler.notify("data_acquisition", "Got 3 datasets")
        await handler.request_approval("analysis", "Ready to analyze?")

        assert all(e.session_id == "my-session" for e in ch.sent)


class TestCheckpointHandlerFallback:
    @pytest.mark.asyncio
    async def test_no_channels_auto_approves(self):
        """When no channels are configured, approval auto-approves."""
        router = NotificationRouter()
        handler = ChannelCheckpointHandler(router, session_id="sess-empty")
        result = await handler.request_approval("plan", "No channels configured")
        assert result is True
