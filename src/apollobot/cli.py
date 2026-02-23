"""
CLI — the primary user interface for ApolloBot.

Commands:
    apollo init              — Interactive setup
    apollo research          — Start a research session (alias for discover)
    apollo discover          — Start a Discover mode session
    apollo translate         — Translate findings into implementation spec
    apollo implement         — Build from translation spec
    apollo commercialize     — Market analysis and GTM planning
    apollo pipeline          — Full Discover → Translate → Implement → Commercialize
    apollo checkpoint        — Manage pipeline checkpoints
    apollo provenance        — View provenance chain
    apollo status            — Check running session status
    apollo submit            — Submit to Frontier Science Journal
    apollo list              — List past sessions
    apollo servers           — Manage MCP server connections
    apollo calls             — View Compute Fund calls
    apollo apply-grant       — Apply for compute grants
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.prompt import Prompt

from apollobot import __version__

console = Console()


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """ApolloBot — Autonomous research engine by Frontier Science."""
    pass


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


@main.command()
def init() -> None:
    """Interactive setup — configure identity, API keys, and domain."""
    from apollobot.core import ApolloConfig, UserIdentity, APIConfig, ComputeConfig, save_config, APOLLO_HOME

    console.print("\n[bold green]ApolloBot Setup[/bold green]\n")

    name = Prompt.ask("  Name", default="")
    affiliation = Prompt.ask("  Affiliation", default="")
    email = Prompt.ask("  Email", default="")
    orcid = Prompt.ask("  ORCID (optional)", default="")

    provider = Prompt.ask("  Default AI provider", choices=["anthropic", "openai"], default="anthropic")
    api_key = Prompt.ask(f"  {provider.title()} API key", password=True, default="")

    domain = Prompt.ask(
        "  Primary domain",
        choices=["bioinformatics", "physics", "cs_ml", "comp_chem", "economics"],
        default="bioinformatics",
    )

    compute_mode = Prompt.ask("  Compute mode", choices=["local", "cloud", "hybrid"], default="local")
    max_budget = float(Prompt.ask("  Max budget per session (USD)", default="50"))

    config = ApolloConfig(
        identity=UserIdentity(name=name, affiliation=affiliation, email=email, orcid=orcid),
        api=APIConfig(
            default_provider=provider,
            anthropic_api_key=api_key if provider == "anthropic" else "",
            openai_api_key=api_key if provider == "openai" else "",
        ),
        compute=ComputeConfig(mode=compute_mode, max_budget_usd=max_budget),
        default_domain=domain,
    )
    save_config(config)

    console.print(f"\n[green]>[/green] Config saved to {APOLLO_HOME / 'config.yaml'}")
    console.print("[green]>[/green] Ready! Try: [bold]apollo discover \"your question here\"[/bold]\n")


# ---------------------------------------------------------------------------
# Discover mode (also aliased as 'research')
# ---------------------------------------------------------------------------


def _run_discover(objective, mission_file, mode, domain, paper, dataset, non_interactive):
    """Shared logic for discover/research commands."""
    from apollobot.agents.orchestrator import run_research
    from apollobot.core import load_config

    if not objective and not mission_file:
        console.print("[red]Error: Provide an objective or --from mission.yaml[/red]")
        sys.exit(1)

    config = load_config()
    if not config.api.get_key():
        console.print("[red]Error: No API key. Run 'apollo init' first.[/red]")
        sys.exit(1)

    asyncio.run(run_research(
        objective=objective or "",
        mode=mode or config.default_mode,
        domain=domain or config.default_domain,
        mission_file=mission_file or "",
        interactive=not non_interactive,
    ))


@main.command()
@click.argument("objective", required=False)
@click.option("--from", "mission_file", type=click.Path(exists=True), help="Mission YAML file")
@click.option("--mode", "-m", type=click.Choice(["hypothesis", "exploratory", "meta-analysis", "replication", "simulation"]), default=None)
@click.option("--domain", "-d", default=None)
@click.option("--paper", default="", help="Paper ID for replication mode")
@click.option("--dataset", default="", help="Dataset ID for exploratory mode")
@click.option("--non-interactive", is_flag=True)
def discover(objective, mission_file, mode, domain, paper, dataset, non_interactive):
    """Start a Discover mode research session."""
    _run_discover(objective, mission_file, mode, domain, paper, dataset, non_interactive)


@main.command()
@click.argument("objective", required=False)
@click.option("--from", "mission_file", type=click.Path(exists=True), help="Mission YAML file")
@click.option("--mode", "-m", type=click.Choice(["hypothesis", "exploratory", "meta-analysis", "replication", "simulation"]), default=None)
@click.option("--domain", "-d", default=None)
@click.option("--paper", default="", help="Paper ID for replication mode")
@click.option("--dataset", default="", help="Dataset ID for exploratory mode")
@click.option("--non-interactive", is_flag=True)
def research(objective, mission_file, mode, domain, paper, dataset, non_interactive):
    """Start a research session (alias for discover)."""
    _run_discover(objective, mission_file, mode, domain, paper, dataset, non_interactive)


# ---------------------------------------------------------------------------
# Translate mode
# ---------------------------------------------------------------------------


@main.command()
@click.option("--session", "session_id", default="", help="Session ID to translate")
@click.option("--paper", "paper_doi", default="", help="External paper DOI to translate")
@click.option("--domain", "-d", default=None)
@click.option("--non-interactive", is_flag=True)
def translate(session_id, paper_doi, domain, non_interactive):
    """Translate research findings into implementation specs."""
    from apollobot.agents.orchestrator import run_translate
    from apollobot.core import load_config

    if not session_id and not paper_doi:
        console.print("[red]Error: Provide --session <id> or --paper <doi>[/red]")
        sys.exit(1)

    config = load_config()
    if not config.api.get_key():
        console.print("[red]Error: No API key. Run 'apollo init' first.[/red]")
        sys.exit(1)

    asyncio.run(run_translate(
        session_id=session_id,
        paper_doi=paper_doi,
        domain=domain or config.default_domain,
        interactive=not non_interactive,
    ))


# ---------------------------------------------------------------------------
# Implement mode
# ---------------------------------------------------------------------------


@main.command()
@click.option("--spec", "session_id", required=True, help="Session ID with translation spec")
@click.option("--domain", "-d", default=None)
@click.option("--non-interactive", is_flag=True)
def implement(session_id, domain, non_interactive):
    """Build production implementation from translation spec."""
    from apollobot.agents.orchestrator import run_implement
    from apollobot.core import load_config

    config = load_config()
    if not config.api.get_key():
        console.print("[red]Error: No API key. Run 'apollo init' first.[/red]")
        sys.exit(1)

    asyncio.run(run_implement(
        session_id=session_id,
        domain=domain or config.default_domain,
        interactive=not non_interactive,
    ))


# ---------------------------------------------------------------------------
# Commercialize mode
# ---------------------------------------------------------------------------


@main.command()
@click.option("--impl", "session_id", required=True, help="Session ID with implementation")
@click.option("--domain", "-d", default=None)
@click.option("--non-interactive", is_flag=True)
def commercialize(session_id, domain, non_interactive):
    """Generate market analysis and go-to-market plan."""
    from apollobot.agents.orchestrator import run_commercialize
    from apollobot.core import load_config

    config = load_config()
    if not config.api.get_key():
        console.print("[red]Error: No API key. Run 'apollo init' first.[/red]")
        sys.exit(1)

    asyncio.run(run_commercialize(
        session_id=session_id,
        domain=domain or config.default_domain,
        interactive=not non_interactive,
    ))


# ---------------------------------------------------------------------------
# Pipeline mode
# ---------------------------------------------------------------------------


@main.command()
@click.argument("objective")
@click.option("--domain", "-d", default=None)
@click.option("--auto-translate", is_flag=True, help="Auto-translate if score >= 7")
@click.option("--non-interactive", is_flag=True)
def pipeline(objective, domain, auto_translate, non_interactive):
    """Run full pipeline: Discover -> Translate -> Implement -> Commercialize."""
    from apollobot.agents.orchestrator import run_pipeline
    from apollobot.core import load_config

    config = load_config()
    if not config.api.get_key():
        console.print("[red]Error: No API key. Run 'apollo init' first.[/red]")
        sys.exit(1)

    asyncio.run(run_pipeline(
        objective=objective,
        domain=domain or config.default_domain,
        auto_translate=auto_translate,
        interactive=not non_interactive,
    ))


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------


@main.group()
def checkpoint() -> None:
    """Manage pipeline checkpoints."""
    pass


@checkpoint.command()
@click.argument("session_id")
def approve(session_id):
    """Approve a pending checkpoint."""
    console.print(f"[green]>[/green] Checkpoint approved for {session_id}")
    console.print("[yellow]Note: Web-based checkpoint approval coming in v0.3.0[/yellow]")


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


@main.command()
@click.argument("session_id")
def provenance(session_id):
    """View the full provenance chain for a session."""
    from apollobot.core import APOLLO_SESSIONS_DIR

    session_dir = Path(APOLLO_SESSIONS_DIR) / session_id
    prov_dir = session_dir / "provenance"

    if not prov_dir.exists():
        console.print(f"[red]No provenance found for {session_id}[/red]")
        return

    # Execution log
    exec_log = prov_dir / "execution_log.json"
    if exec_log.exists():
        events = json.loads(exec_log.read_text())
        console.print(f"\n[bold]Execution Log[/bold] ({len(events)} events)")
        for e in events[-10:]:  # Show last 10
            console.print(f"  [{e.get('timestamp', '?')[:19]}] {e.get('event', '?')}")

    # Data lineage
    lineage = prov_dir / "data_lineage.json"
    if lineage.exists():
        transforms = json.loads(lineage.read_text())
        console.print(f"\n[bold]Data Lineage[/bold] ({len(transforms)} transforms)")
        for t in transforms[-5:]:
            console.print(f"  {t.get('source', '?')} -> {t.get('operation', '?')}")

    # Model calls
    calls = prov_dir / "model_calls.json"
    if calls.exists():
        model_calls = json.loads(calls.read_text())
        total_cost = sum(c.get("cost_usd", 0) for c in model_calls)
        console.print(f"\n[bold]Model Calls[/bold] ({len(model_calls)} calls, ${total_cost:.2f} total)")

    # Source provenance (cross-mode)
    source = prov_dir / "source_provenance.json"
    if source.exists():
        console.print("\n[bold]Cross-mode link:[/bold] Source provenance linked")


# ---------------------------------------------------------------------------
# Status & List
# ---------------------------------------------------------------------------


@main.command()
@click.argument("session_id", required=False)
def status(session_id):
    """Check status of sessions."""
    from apollobot.core import APOLLO_SESSIONS_DIR
    from apollobot.core.session import Session

    sessions_dir = Path(APOLLO_SESSIONS_DIR)
    if not sessions_dir.exists():
        console.print("[dim]No sessions found.[/dim]")
        return

    if session_id:
        session_path = sessions_dir / session_id
        if not session_path.exists():
            console.print(f"[red]Session {session_id} not found.[/red]")
            return
        session = Session.load_state(session_path)
        console.print(f"\n[bold]{session.mission.id}[/bold]")
        console.print(f"  Mode: {session.mission.mode.value}")
        console.print(f"  Objective: {session.mission.objective}")
        console.print(f"  Phase: {session.current_phase.value}")
        console.print(f"  Cost: ${session.cost.total_cost:.2f}")
        if session.translation_scores:
            avg = session.translation_scores.get("average", 0)
            console.print(f"  Translation potential: {avg:.1f}/10")
    else:
        for d in sorted(sessions_dir.iterdir(), reverse=True):
            if d.is_dir() and (d / "session_state.json").exists():
                try:
                    session = Session.load_state(d)
                    phase = session.current_phase.value
                    mode = session.mission.mode.value
                    emoji = {"complete": ">", "failed": "x", "cancelled": "-"}.get(phase, "~")
                    console.print(f"  {emoji} [bold]{session.mission.id}[/bold] [{mode}:{phase}] — {session.mission.title[:60]}")
                except Exception:
                    console.print(f"  ? [dim]{d.name} (corrupted)[/dim]")


@main.command(name="list")
def list_sessions():
    """List all research sessions."""
    from apollobot.core import APOLLO_SESSIONS_DIR
    sessions_dir = Path(APOLLO_SESSIONS_DIR)
    if not sessions_dir.exists():
        console.print("[dim]No sessions found.[/dim]")
        return
    for d in sorted(sessions_dir.iterdir(), reverse=True):
        if d.is_dir() and (d / "mission.yaml").exists():
            console.print(f"  - {d.name}")


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------


@main.command()
@click.option("--session", required=True, help="Session ID")
@click.option("--journal", default="frontier")
def submit(session, journal):
    """Submit a completed session to Frontier Science Journal."""
    from apollobot.core import APOLLO_SESSIONS_DIR
    session_dir = Path(APOLLO_SESSIONS_DIR) / session
    if not session_dir.exists():
        console.print(f"[red]Session {session} not found.[/red]")
        sys.exit(1)
    console.print(f"[bold]Packaging {session} for {journal}...[/bold]")
    console.print(f"[yellow]Submission API coming in v0.2.0[/yellow]")
    console.print(f"[dim]Submit manually at https://frontierscience.ai/journal/submit[/dim]")


# ---------------------------------------------------------------------------
# Compute Fund
# ---------------------------------------------------------------------------


@main.group()
def calls() -> None:
    """View Compute Fund calls."""
    pass


@calls.command(name="list")
@click.option("--track", default="", help="Filter by domain track")
def calls_list(track):
    """List open Compute Fund calls."""
    console.print("[bold]Open Compute Fund Calls[/bold]\n")
    console.print("[dim]No open calls. Check https://frontierscience.ai/compute[/dim]")
    console.print("[yellow]Compute Fund API integration coming in v0.3.0[/yellow]")


@calls.command(name="track")
@click.argument("domain")
def calls_track(domain):
    """Track calls for a specific domain."""
    console.print(f"[bold]Tracking calls for: {domain}[/bold]")
    console.print("[yellow]Coming in v0.3.0[/yellow]")


@main.command(name="apply-grant")
@click.option("--proposal", required=True, type=click.Path(exists=True), help="Proposal YAML file")
def apply_grant(proposal):
    """Apply for a Compute Fund grant."""
    console.print(f"[bold]Submitting grant proposal: {proposal}[/bold]")
    console.print("[yellow]Grant application API coming in v0.3.0[/yellow]")


# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------


@main.group()
def notify() -> None:
    """Manage notification channels."""
    pass


@notify.command(name="list")
def notify_list():
    """Show configured notification channels and their status."""
    from apollobot.core import load_config

    config = load_config()
    notif = config.notifications

    if not notif.enabled:
        console.print("[dim]Notifications are disabled.[/dim]")
        console.print("[dim]Enable in ~/.apollobot/config.yaml under notifications.enabled[/dim]")
        return

    console.print(f"\n[bold]Notification channels[/bold] (heartbeat: {notif.heartbeat_interval}s)\n")

    if not notif.channels:
        console.print("[dim]  No channels configured.[/dim]")
        return

    for ch in notif.channels:
        status = "[green]enabled[/green]" if ch.enabled else "[red]disabled[/red]"
        events = ", ".join(ch.events) if ch.events != ["*"] else "all events"
        console.print(f"  [{status}] [bold]{ch.type}[/bold] — {events}")


@notify.command(name="test")
def notify_test():
    """Send a test notification to all configured channels."""
    from apollobot.core import load_config
    from apollobot.agents.orchestrator import Orchestrator
    from apollobot.notifications.events import EventType, NotificationEvent

    config = load_config()
    if not config.notifications.enabled:
        console.print("[red]Notifications are not enabled. Edit ~/.apollobot/config.yaml[/red]")
        return

    async def _test():
        orchestrator = Orchestrator(config=config, interactive=False)
        router = orchestrator.router
        await router.connect_all()
        event = NotificationEvent(
            event_type=EventType.HEARTBEAT,
            session_id="test-notification",
            title="ApolloBot test notification",
            summary="If you see this, your notification channel is working!",
        )
        await router.dispatch(event)
        await router.disconnect_all()
        console.print("[green]>[/green] Test notification sent to all channels.")

    asyncio.run(_test())


@notify.command(name="setup")
def notify_setup():
    """Interactive channel configuration wizard."""
    from apollobot.core import load_config, save_config
    from apollobot.notifications.config import ChannelConfig

    config = load_config()

    console.print("\n[bold]Notification Setup[/bold]\n")

    ch_type = Prompt.ask(
        "Channel type",
        choices=["telegram", "discord", "slack", "google_chat", "email", "webhook", "console"],
    )

    extras: dict = {}
    if ch_type == "telegram":
        extras["token"] = Prompt.ask("Bot token")
        extras["chat_id"] = Prompt.ask("Chat ID")
    elif ch_type == "discord":
        extras["webhook_url"] = Prompt.ask("Webhook URL")
        if Prompt.ask("Enable bidirectional approvals?", choices=["y", "n"], default="n") == "y":
            extras["bot_token"] = Prompt.ask("Bot token")
            extras["channel_id"] = Prompt.ask("Channel ID")
    elif ch_type == "slack":
        extras["webhook_url"] = Prompt.ask("Incoming webhook URL")
    elif ch_type == "google_chat":
        extras["webhook_url"] = Prompt.ask("Webhook URL")
    elif ch_type == "email":
        extras["smtp_host"] = Prompt.ask("SMTP host", default="smtp.gmail.com")
        extras["smtp_port"] = int(Prompt.ask("SMTP port", default="587"))
        extras["username"] = Prompt.ask("Username/email")
        extras["password"] = Prompt.ask("Password", password=True)
        extras["from_addr"] = Prompt.ask("From address", default=extras["username"])
        extras["to_addrs"] = [Prompt.ask("To address")]
    elif ch_type == "webhook":
        extras["url"] = Prompt.ask("Webhook URL")
        secret = Prompt.ask("HMAC secret (optional)", default="")
        if secret:
            extras["secret"] = secret

    ch_config = ChannelConfig(type=ch_type, **extras)
    config.notifications.enabled = True
    config.notifications.channels.append(ch_config)
    save_config(config)

    console.print(f"\n[green]>[/green] Added {ch_type} channel. Run [bold]apollo notify test[/bold] to verify.")


# ---------------------------------------------------------------------------
# Servers
# ---------------------------------------------------------------------------


@main.command()
@click.argument("action", type=click.Choice(["list", "add", "test"]), default="list")
@click.option("--name", default="")
@click.option("--url", default="")
@click.option("--domain", default="")
def servers(action, name, url, domain):
    """Manage MCP server connections."""
    from apollobot.mcp.servers.builtin import ALL_BUILTIN_SERVERS
    from apollobot.core import load_custom_servers

    if action == "list":
        console.print("\n[bold]Built-in servers:[/bold]")
        for srv in ALL_BUILTIN_SERVERS:
            console.print(f"  - [bold]{srv.name}[/bold] [{srv.domain}] — {srv.description}")
        custom = load_custom_servers()
        if custom:
            console.print("\n[bold]Custom servers:[/bold]")
            for s in custom:
                console.print(f"  - [bold]{s['name']}[/bold] — {s.get('url', 'N/A')}")

    elif action == "add":
        if not name or not url:
            console.print("[red]--name and --url required[/red]")
            return
        from apollobot.core import APOLLO_SERVERS_FILE, APOLLO_HOME
        import yaml
        APOLLO_HOME.mkdir(parents=True, exist_ok=True)
        existing = {"custom_servers": load_custom_servers()}
        existing["custom_servers"].append({"name": name, "url": url, "domain": domain})
        APOLLO_SERVERS_FILE.write_text(yaml.dump(existing))
        console.print(f"[green]>[/green] Added {name}")

    elif action == "test":
        if not name:
            console.print("[red]--name required[/red]")
            return
        console.print(f"[dim]Testing {name}...[/dim]")
        console.print(f"[yellow]Server testing coming in v0.2.0[/yellow]")


if __name__ == "__main__":
    main()
