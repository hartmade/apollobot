"""Channel implementation tests with mocked HTTP."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from apollobot.notifications.events import (
    EventSeverity,
    EventType,
    NotificationEvent,
)


def _make_event(**kwargs) -> NotificationEvent:
    defaults = {
        "event_type": EventType.PHASE_COMPLETED,
        "session_id": "test-123",
        "title": "Test",
        "summary": "A test event",
    }
    defaults.update(kwargs)
    return NotificationEvent(**defaults)


# ---------------------------------------------------------------------------
# Console channel
# ---------------------------------------------------------------------------


class TestConsoleChannel:
    @pytest.mark.asyncio
    async def test_send_info(self):
        from apollobot.notifications.channels.console import ConsoleChannel

        mock_console = MagicMock()
        ch = ConsoleChannel(console=mock_console)
        await ch.send(_make_event())
        assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_send_error_uses_panel(self):
        from apollobot.notifications.channels.console import ConsoleChannel

        mock_console = MagicMock()
        ch = ConsoleChannel(console=mock_console)
        await ch.send(_make_event(severity=EventSeverity.ERROR))
        assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_heartbeat_is_dim(self):
        from apollobot.notifications.channels.console import ConsoleChannel

        mock_console = MagicMock()
        ch = ConsoleChannel(console=mock_console)
        await ch.send(_make_event(event_type=EventType.HEARTBEAT))
        call_args = mock_console.print.call_args[0][0]
        assert "[dim]" in call_args


# ---------------------------------------------------------------------------
# Webhook channel
# ---------------------------------------------------------------------------


class TestWebhookChannel:
    @pytest.mark.asyncio
    async def test_send_posts_json(self):
        from apollobot.notifications.channels.webhook import WebhookChannel

        ch = WebhookChannel(url="https://example.com/hook")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.aclose = AsyncMock()

        with patch("apollobot.notifications.channels.webhook.httpx.AsyncClient", return_value=mock_client):
            await ch.send(_make_event())

        mock_client.post.assert_awaited_once()
        call_kwargs = mock_client.post.call_args
        assert call_kwargs[1]["headers"]["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_send_with_hmac(self):
        from apollobot.notifications.channels.webhook import WebhookChannel

        ch = WebhookChannel(url="https://example.com/hook", secret="mysecret")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.aclose = AsyncMock()

        with patch("apollobot.notifications.channels.webhook.httpx.AsyncClient", return_value=mock_client):
            await ch.send(_make_event())

        headers = mock_client.post.call_args[1]["headers"]
        assert "X-Apollo-Signature" in headers
        assert headers["X-Apollo-Signature"].startswith("sha256=")


# ---------------------------------------------------------------------------
# Slack channel
# ---------------------------------------------------------------------------


class TestSlackChannel:
    @pytest.mark.asyncio
    async def test_send_blocks(self):
        from apollobot.notifications.channels.slack import SlackChannel

        ch = SlackChannel(webhook_url="https://hooks.slack.com/services/T/B/X")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.aclose = AsyncMock()

        with patch("apollobot.notifications.channels.slack.httpx.AsyncClient", return_value=mock_client):
            await ch.send(_make_event(phase="analysis"))

        payload = mock_client.post.call_args[1]["json"]
        assert "blocks" in payload
        assert any(b["type"] == "header" for b in payload["blocks"])


# ---------------------------------------------------------------------------
# Google Chat channel
# ---------------------------------------------------------------------------


class TestGoogleChatChannel:
    @pytest.mark.asyncio
    async def test_send_card(self):
        from apollobot.notifications.channels.google_chat import GoogleChatChannel

        ch = GoogleChatChannel(webhook_url="https://chat.googleapis.com/v1/spaces/X/messages?key=Y")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.aclose = AsyncMock()

        with patch("apollobot.notifications.channels.google_chat.httpx.AsyncClient", return_value=mock_client):
            await ch.send(_make_event())

        payload = mock_client.post.call_args[1]["json"]
        assert "cardsV2" in payload


# ---------------------------------------------------------------------------
# Telegram channel
# ---------------------------------------------------------------------------


class TestTelegramChannel:
    @pytest.mark.asyncio
    async def test_send_message(self):
        from apollobot.notifications.channels.telegram import TelegramChannel

        ch = TelegramChannel(token="123:ABC", chat_id="456")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.aclose = AsyncMock()

        with patch("apollobot.notifications.channels.telegram.httpx.AsyncClient", return_value=mock_client):
            await ch.send(_make_event())

        payload = mock_client.post.call_args[1]["json"]
        assert payload["chat_id"] == "456"
        assert payload["parse_mode"] == "HTML"

    @pytest.mark.asyncio
    async def test_approval_sends_inline_keyboard(self):
        from apollobot.notifications.channels.telegram import TelegramChannel

        ch = TelegramChannel(token="123:ABC", chat_id="456")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.aclose = AsyncMock()

        event = _make_event(
            event_type=EventType.CHECKPOINT_APPROVAL,
            requires_response=True,
            phase="plan",
        )

        with patch("apollobot.notifications.channels.telegram.httpx.AsyncClient", return_value=mock_client):
            await ch.send(event)

        payload = mock_client.post.call_args[1]["json"]
        assert "reply_markup" in payload
        keyboard = payload["reply_markup"]["inline_keyboard"]
        assert len(keyboard[0]) == 2  # Approve + Deny buttons


# ---------------------------------------------------------------------------
# Discord channel
# ---------------------------------------------------------------------------


class TestDiscordChannel:
    @pytest.mark.asyncio
    async def test_send_embed(self):
        from apollobot.notifications.channels.discord import DiscordChannel

        ch = DiscordChannel(webhook_url="https://discord.com/api/webhooks/X/Y")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.aclose = AsyncMock()

        with patch("apollobot.notifications.channels.discord.httpx.AsyncClient", return_value=mock_client):
            await ch.send(_make_event())

        payload = mock_client.post.call_args[1]["json"]
        assert "embeds" in payload
        assert payload["embeds"][0]["title"] == "Test"


# ---------------------------------------------------------------------------
# Email channel
# ---------------------------------------------------------------------------


class TestEmailChannel:
    @pytest.mark.asyncio
    async def test_send_critical_immediately(self):
        from apollobot.notifications.channels.email import EmailChannel

        ch = EmailChannel(
            smtp_host="localhost",
            from_addr="test@example.com",
            to_addrs=["user@example.com"],
        )

        with patch.object(ch, "_smtp_send") as mock_smtp:
            await ch.send(_make_event(
                event_type=EventType.SESSION_FAILED,
                severity=EventSeverity.CRITICAL,
            ))
            mock_smtp.assert_called_once()

    @pytest.mark.asyncio
    async def test_batches_non_critical(self):
        from apollobot.notifications.channels.email import EmailChannel

        ch = EmailChannel(
            smtp_host="localhost",
            from_addr="test@example.com",
            to_addrs=["user@example.com"],
            min_interval=9999,  # very long interval
        )

        with patch.object(ch, "_smtp_send") as mock_smtp:
            await ch.send(_make_event())
            # First send goes through (last_sent is 0)
            assert mock_smtp.call_count == 1

            await ch.send(_make_event())
            # Second should be batched (within min_interval)
            assert mock_smtp.call_count == 1
            assert len(ch._pending) == 1
