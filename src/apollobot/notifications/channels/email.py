"""
Email channel — SMTP notifications with HTML summaries.

Batches notifications to avoid spam (configurable minimum interval).
Sends a rich HTML summary on session completion.
"""

from __future__ import annotations

import asyncio
import logging
import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from apollobot.notifications.channel import NotificationChannel
from apollobot.notifications.events import (
    EventSeverity,
    EventType,
    NotificationEvent,
)

logger = logging.getLogger(__name__)


class EmailChannel(NotificationChannel):
    """SMTP email notification channel."""

    name: str = "email"
    supports_responses: bool = False

    def __init__(
        self,
        smtp_host: str = "localhost",
        smtp_port: int = 587,
        username: str = "",
        password: str = "",
        from_addr: str = "",
        to_addrs: list[str] | None = None,
        use_tls: bool = True,
        min_interval: float = 60.0,
    ) -> None:
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs or []
        self.use_tls = use_tls
        self.min_interval = min_interval
        self._last_sent: float = 0.0
        self._pending: list[NotificationEvent] = []

    async def send(self, event: NotificationEvent) -> None:
        now = time.monotonic()

        # Always send immediately for critical events and session summaries
        if event.event_type in (
            EventType.SESSION_COMPLETED,
            EventType.SESSION_FAILED,
        ) or event.severity == EventSeverity.CRITICAL:
            # Flush any pending events along with this one
            events_to_send = self._pending + [event]
            self._pending = []
            await self._send_email(events_to_send)
            self._last_sent = now
            return

        # Batch non-critical events
        self._pending.append(event)
        if now - self._last_sent >= self.min_interval:
            events_to_send = self._pending
            self._pending = []
            await self._send_email(events_to_send)
            self._last_sent = now

    async def _send_email(self, events: list[NotificationEvent]) -> None:
        if not events or not self.to_addrs:
            return

        # Build subject from most important event
        primary = events[-1]
        subject = f"[ApolloBot] {primary.title}"
        if len(events) > 1:
            subject += f" (+{len(events) - 1} more)"

        html = self._build_html(events)
        plain = self._build_plain(events)

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.from_addr
        msg["To"] = ", ".join(self.to_addrs)
        msg.attach(MIMEText(plain, "plain"))
        msg.attach(MIMEText(html, "html"))

        # Run SMTP in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, self._smtp_send, msg)
        except Exception:
            logger.exception("Failed to send email notification")

    def _smtp_send(self, msg: MIMEMultipart) -> None:
        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            if self.use_tls:
                server.starttls()
            if self.username:
                server.login(self.username, self.password)
            server.send_message(msg)

    def _build_html(self, events: list[NotificationEvent]) -> str:
        rows = []
        for ev in events:
            color = {
                EventSeverity.INFO: "#3498db",
                EventSeverity.WARNING: "#f39c12",
                EventSeverity.ERROR: "#e74c3c",
                EventSeverity.CRITICAL: "#8e44ad",
            }.get(ev.severity, "#3498db")

            rows.append(
                f'<tr><td style="border-left: 4px solid {color}; padding: 8px;">'
                f"<strong>{ev.title}</strong><br/>"
                f"{ev.summary}"
                f"{'<br/><em>Phase: ' + ev.phase + '</em>' if ev.phase else ''}"
                f"</td></tr>"
            )

        return (
            "<html><body>"
            '<table style="font-family: sans-serif; border-collapse: collapse; width: 100%;">'
            f"{''.join(rows)}"
            "</table>"
            f'<p style="color: #999; font-size: 12px;">Session: {events[0].session_id}</p>'
            "</body></html>"
        )

    def _build_plain(self, events: list[NotificationEvent]) -> str:
        lines = []
        for ev in events:
            lines.append(f"[{ev.severity.value.upper()}] {ev.title}")
            lines.append(f"  {ev.summary}")
            if ev.phase:
                lines.append(f"  Phase: {ev.phase}")
            lines.append("")
        lines.append(f"Session: {events[0].session_id}")
        return "\n".join(lines)
