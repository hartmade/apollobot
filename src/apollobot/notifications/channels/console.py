"""
Console channel — Rich terminal output for notifications.

Wraps the existing terminal UI behavior as a notification channel,
serving as the default when no external channels are configured.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel

from apollobot.notifications.channel import NotificationChannel
from apollobot.notifications.events import (
    EventSeverity,
    EventType,
    NotificationEvent,
)

_SEVERITY_STYLE = {
    EventSeverity.INFO: "blue",
    EventSeverity.WARNING: "yellow",
    EventSeverity.ERROR: "red",
    EventSeverity.CRITICAL: "bold red",
}

_EVENT_EMOJI = {
    EventType.SESSION_STARTED: "\U0001f52c",   # microscope
    EventType.PHASE_STARTED: "\u25b6\ufe0f",   # play
    EventType.PHASE_COMPLETED: "\u2705",        # check
    EventType.PHASE_FAILED: "\u274c",           # cross
    EventType.CHECKPOINT_APPROVAL: "\u26a0\ufe0f",  # warning
    EventType.FINDING: "\U0001f4a1",            # bulb
    EventType.BUDGET_WARNING: "\U0001f4b0",     # money bag
    EventType.SESSION_COMPLETED: "\U0001f389",  # party
    EventType.SESSION_FAILED: "\U0001f6a8",     # rotating light
    EventType.HEARTBEAT: "\U0001f493",          # heartbeat
}


class ConsoleChannel(NotificationChannel):
    """Rich terminal output channel."""

    name: str = "console"
    supports_responses: bool = False

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()

    async def send(self, event: NotificationEvent) -> None:
        emoji = _EVENT_EMOJI.get(event.event_type, "\u2139\ufe0f")
        style = _SEVERITY_STYLE.get(event.severity, "blue")

        if event.event_type == EventType.HEARTBEAT:
            self._console.print(f"[dim]{emoji} {event.summary}[/dim]")
            return

        if event.severity in (EventSeverity.ERROR, EventSeverity.CRITICAL):
            self._console.print(
                Panel(
                    f"{event.summary}",
                    title=f"{emoji} {event.title}",
                    border_style=style,
                )
            )
        else:
            self._console.print(
                f"\n[bold {style}]{emoji} {event.title}[/bold {style}]"
            )
            if event.summary:
                self._console.print(f"  {event.summary}")
