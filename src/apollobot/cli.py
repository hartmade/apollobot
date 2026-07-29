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
    apollo run               — Start continuous autonomous runtime
    apollo activity          — Query runtime action/decision history
    apollo checkpoint        — Manage pipeline checkpoints
    apollo provenance        — View provenance chain
    apollo status            — Check running session status
    apollo submit            — Explain managed publication workflow
    apollo list              — List past sessions
    apollo servers           — Manage MCP server connections
    apollo calls             — View Compute Fund calls
    apollo apply-grant       — Apply for compute grants
    apollo export            — Export research data as portable archive
    apollo report            — Generate research performance report
    apollo resume            — Resume or clean up crashed sessions
    apollo monitor           — Live monitoring dashboard
    apollo guardrails        — Manage runtime safety constraints
"""

from __future__ import annotations

import asyncio
import json
import os
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
    from apollobot.core import (
        ApolloConfig,
        UserIdentity,
        APIConfig,
        ComputeConfig,
        save_config,
        APOLLO_HOME,
    )

    console.print("\n[bold green]ApolloBot Setup[/bold green]\n")

    name = Prompt.ask("  Name", default="")
    affiliation = Prompt.ask("  Affiliation", default="")
    email = Prompt.ask("  Email", default="")
    orcid = Prompt.ask("  ORCID (optional)", default="")

    provider = Prompt.ask(
        "  Default AI provider", choices=["anthropic", "openai", "minimax"], default="anthropic"
    )
    api_key = Prompt.ask(f"  {provider.title()} API key", password=True, default="")

    domain = Prompt.ask(
        "  Primary domain",
        choices=["bioinformatics", "physics", "cs_ml", "comp_chem", "economics"],
        default="bioinformatics",
    )

    compute_mode = Prompt.ask(
        "  Compute mode", choices=["local", "cloud", "hybrid"], default="local"
    )
    max_budget = float(Prompt.ask("  Max budget per session (USD)", default="50"))

    config = ApolloConfig(
        identity=UserIdentity(name=name, affiliation=affiliation, email=email, orcid=orcid),
        api=APIConfig(
            default_provider=provider,
            anthropic_api_key=api_key if provider == "anthropic" else "",
            openai_api_key=api_key if provider == "openai" else "",
            minimax_api_key=api_key if provider == "minimax" else "",
        ),
        compute=ComputeConfig(mode=compute_mode, max_budget_usd=max_budget),
        default_domain=domain,
    )
    save_config(config)

    console.print(f"\n[green]>[/green] Config saved to {APOLLO_HOME / 'config.yaml'}")
    console.print(
        '[green]>[/green] Ready! Try: [bold]apollo discover "your question here"[/bold]\n'
    )


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

    asyncio.run(
        run_research(
            objective=objective or "",
            mode=mode or config.default_mode,
            domain=domain or config.default_domain,
            mission_file=mission_file or "",
            interactive=not non_interactive,
        )
    )


@main.command()
@click.argument("objective", required=False)
@click.option("--from", "mission_file", type=click.Path(exists=True), help="Mission YAML file")
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["hypothesis", "exploratory", "meta-analysis", "replication", "simulation"]),
    default=None,
)
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
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["hypothesis", "exploratory", "meta-analysis", "replication", "simulation"]),
    default=None,
)
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

    asyncio.run(
        run_translate(
            session_id=session_id,
            paper_doi=paper_doi,
            domain=domain or config.default_domain,
            interactive=not non_interactive,
        )
    )


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

    asyncio.run(
        run_implement(
            session_id=session_id,
            domain=domain or config.default_domain,
            interactive=not non_interactive,
        )
    )


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

    asyncio.run(
        run_commercialize(
            session_id=session_id,
            domain=domain or config.default_domain,
            interactive=not non_interactive,
        )
    )


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

    asyncio.run(
        run_pipeline(
            objective=objective,
            domain=domain or config.default_domain,
            auto_translate=auto_translate,
            interactive=not non_interactive,
        )
    )


# ---------------------------------------------------------------------------
# Continuous runtime
# ---------------------------------------------------------------------------


@main.command()
@click.option("--domain", "-d", default=None, help="Research domain focus")
@click.option("--daily-budget", type=float, default=100.0, help="Daily compute budget in USD")
@click.option("--max-concurrent", type=int, default=3, help="Max concurrent sessions")
@click.option("--interval", type=int, default=300, help="Seconds between ticks")
@click.option("--instructions", default="", help="Custom instructions for the brain")
@click.option("--health-port", type=int, default=8080, help="Port for health check HTTP server")
@click.option("--dry-run", is_flag=True, help="Dry run — no real API calls or sessions")
def run(domain, daily_budget, max_concurrent, interval, instructions, health_port, dry_run):
    """Start the continuous autonomous runtime.

    ApolloBot will run indefinitely, autonomously discovering research
    questions, investigating them, and producing papers. The brain decides
    what to research and when, within the safety guardrails.

    Examples:

        apollo run --domain bioinformatics

        apollo run --domain physics --daily-budget 50 --interval 600

        apollo run --instructions "Focus on aging-related epigenetic markers"

        apollo run --dry-run  # dry run
    """
    from apollobot.core import load_config
    from apollobot.runtime.runner import run_continuous

    config = load_config()
    if not config.api.get_key():
        console.print("[red]Error: No API key. Run 'apollo init' first.[/red]")
        sys.exit(1)

    asyncio.run(
        run_continuous(
            domain=domain or config.default_domain,
            daily_budget=daily_budget,
            max_concurrent=max_concurrent,
            interval=interval,
            user_instructions=instructions,
            health_port=health_port,
            dry_run=dry_run,
        )
    )


# ---------------------------------------------------------------------------
# Activity history
# ---------------------------------------------------------------------------


def _parse_duration(duration: str) -> int:
    """Convert a human duration string to seconds.

    Accepted suffixes: s (seconds), m (minutes), h (hours), d (days).
    Plain integers are treated as hours for backwards compatibility.
    """
    import re

    m = re.fullmatch(r"(\d+)\s*([smhd])?", duration.strip(), re.IGNORECASE)
    if not m:
        raise click.BadParameter(f"Cannot parse duration: {duration!r}  (try e.g. 1h, 24h, 7d)")
    value = int(m.group(1))
    unit = (m.group(2) or "h").lower()
    multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    return value * multipliers[unit]


@main.command()
@click.option("--last", "duration", default="24h", help="Time window (e.g. 1h, 24h, 7d)")
@click.option(
    "--type",
    "action_type",
    default="",
    help="Filter by action type (start_research, scan_literature, review_session, idle)",
)
@click.option("--domain", default="", help="Filter by domain (searches objective text)")
@click.option("--limit", type=int, default=50, help="Max entries to show")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def activity(duration, action_type, domain, limit, as_json):
    """Query runtime action and decision history."""
    from datetime import datetime, timedelta, timezone

    from apollobot.runtime.storage import RunnerStorage

    # Parse duration and compute cutoff timestamp
    try:
        seconds = _parse_duration(duration)
    except click.BadParameter as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()

    # Open the runtime database (read-only query)
    db_path = Path.home() / ".apollobot" / "runtime.db"
    if not db_path.exists():
        console.print(
            "[dim]No runtime database found. Start the runtime first with 'apollo run'.[/dim]"
        )
        return

    storage = RunnerStorage(db_path=str(db_path))

    try:
        actions = storage.get_actions_since(cutoff, action_type=action_type, limit=limit)
        decisions = storage.get_decisions_since(cutoff, limit=limit)
    finally:
        storage.close()

    # Apply domain filter (action_history has no domain column, so we
    # do a case-insensitive substring match against objective + details).
    if domain:
        domain_lower = domain.lower()
        actions = [a for a in actions if domain_lower in (a.objective + " " + a.details).lower()]

    if not actions and not decisions:
        console.print(f"[dim]No activity found in the last {duration}.[/dim]")
        return

    # ------------------------------------------------------------------
    # JSON output
    # ------------------------------------------------------------------
    if as_json:
        payload = {
            "window": duration,
            "actions": [a.model_dump() for a in actions],
            "decisions": [d.model_dump() for d in decisions],
        }
        click.echo(json.dumps(payload, indent=2))
        return

    # ------------------------------------------------------------------
    # Rich table output
    # ------------------------------------------------------------------
    from rich.table import Table

    if actions:
        table = Table(title=f"Actions (last {duration})", show_lines=False)
        table.add_column("Tick", style="cyan", justify="right")
        table.add_column("Timestamp", style="dim")
        table.add_column("Type", style="bold")
        table.add_column("Objective")
        table.add_column("Result", style="green")
        table.add_column("Details", style="dim")

        for a in actions:
            # Truncate long fields for readability
            objective = (a.objective[:60] + "...") if len(a.objective) > 63 else a.objective
            details = (a.details[:40] + "...") if len(a.details) > 43 else a.details
            ts_display = a.timestamp[:19].replace("T", " ")

            result_style = {
                "completed": "[green]completed[/green]",
                "failed": "[red]failed[/red]",
                "blocked": "[yellow]blocked[/yellow]",
            }.get(a.result, a.result)

            table.add_row(
                str(a.tick),
                ts_display,
                a.action_type,
                objective,
                result_style,
                details,
            )

        console.print(table)

    if decisions:
        console.print()
        dtable = Table(title=f"Decisions (last {duration})", show_lines=False)
        dtable.add_column("Tick", style="cyan", justify="right")
        dtable.add_column("Timestamp", style="dim")
        dtable.add_column("Reasoning")
        dtable.add_column("Actions", style="bold")
        dtable.add_column("Next Check-in", justify="right")

        for d in decisions:
            reasoning = (d.reasoning[:70] + "...") if len(d.reasoning) > 73 else d.reasoning
            actions_str = ", ".join(d.actions[:3])
            if len(d.actions) > 3:
                actions_str += f" (+{len(d.actions) - 3})"
            ts_display = d.timestamp[:19].replace("T", " ")
            checkin = f"{d.next_check_in}s"

            dtable.add_row(
                str(d.tick),
                ts_display,
                reasoning,
                actions_str,
                checkin,
            )

        console.print(dtable)

    # Summary line
    console.print(
        f"\n[dim]{len(actions)} action(s), {len(decisions)} decision(s) "
        f"in the last {duration}[/dim]"
    )


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
        console.print(
            f"\n[bold]Model Calls[/bold] ({len(model_calls)} calls, ${total_cost:.2f} total)"
        )

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

    if not session_id:
        # Show runtime status
        from apollobot.runtime.pidfile import PidFile

        pidfile = PidFile()
        running, pid = pidfile.is_running()
        if running:
            console.print(f"[bold green]Runtime active[/bold green] (PID: {pid})")
        else:
            console.print("[dim]Runtime not running.[/dim]")
        console.print()

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
                    console.print(
                        f"  {emoji} [bold]{session.mission.id}[/bold] [{mode}:{phase}] — {session.mission.title[:60]}"
                    )
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
# Review
# ---------------------------------------------------------------------------


@main.command()
@click.option("--session", "session_id", default="", help="Review an ApolloBot session")
@click.option(
    "--manuscript", type=click.Path(exists=True), default=None, help="Review a manuscript file"
)
@click.option("--output", "-o", type=click.Path(), default="", help="Save report to file")
@click.option("--paper-id", default="", help="Journal paper ID for posting review")
@click.option("--post-to-journal", is_flag=True, help="Post review to Frontier Science Journal API")
def review(session_id, manuscript, output, paper_id, post_to_journal):
    """Run AI review on a session or manuscript."""
    from apollobot.core import APOLLO_SESSIONS_DIR, load_config
    from apollobot.agents import create_llm
    from apollobot.review.submission import SubmissionReviewer

    if post_to_journal or paper_id:
        raise click.ClickException(
            "Direct review posting was retired in v0.2. Managed living records request "
            "automated review through the authenticated Frontier Science workflow."
        )

    if not session_id and not manuscript:
        console.print("[red]Error: Provide --session <id> or --manuscript <file>[/red]")
        sys.exit(1)

    config = load_config()
    if not config.api.get_key():
        console.print("[red]Error: No API key. Run 'apollo init' first.[/red]")
        sys.exit(1)

    llm = create_llm(config.api.default_provider, config.api.get_key())
    reviewer = SubmissionReviewer(llm)
    sessions_dir = Path(config.output_dir) if config.output_dir else Path(APOLLO_SESSIONS_DIR)

    async def _review():
        manuscript_text = ""
        provenance_path = None

        if session_id:
            session_dir = sessions_dir / session_id
            if not session_dir.exists():
                console.print(f"[red]Session {session_id} not found.[/red]")
                sys.exit(1)
            for name in ("manuscript.md", "manuscript.tex"):
                candidate = session_dir / name
                if candidate.exists():
                    manuscript_text = candidate.read_text()
                    break
            prov_dir = session_dir / "provenance"
            if prov_dir.exists():
                provenance_path = prov_dir
            if not manuscript_text:
                console.print(f"[red]No manuscript found in {session_id}[/red]")
                sys.exit(1)
        else:
            manuscript_text = Path(manuscript).read_text()

        console.print("[bold]Running AI review...[/bold]\n")
        report = await reviewer.review(
            manuscript_text,
            provenance_path=provenance_path,
            session_id=session_id,
        )

        report_md = reviewer.format_report(report)
        console.print(report_md)

        # Save report
        save_path = output
        if not save_path and session_id:
            session_dir = sessions_dir / session_id
            save_path = str(session_dir / "review" / "submission_review.md")
        if save_path:
            save_file = Path(save_path)
            save_file.parent.mkdir(parents=True, exist_ok=True)
            save_file.write_text(report_md)
            console.print(f"\n[green]>[/green] Report saved to {save_path}")

    asyncio.run(_review())


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------


@main.command()
@click.option("--session", required=True, help="Session ID")
@click.option("--title", default="", help="Paper title (defaults to session title)")
@click.option(
    "--abstract", "abstract_text", default="", help="Paper abstract (defaults to session objective)"
)
@click.option("--track", default="", help="Journal track (defaults to session domain)")
@click.option("--auto-review", is_flag=True, help="Run AI review after submission")
def submit(session, title, abstract_text, track, auto_review):
    """Explain the authenticated living-record publication workflow."""
    _ = (session, title, abstract_text, track, auto_review)
    console.print("[yellow]Direct CLI publication was retired in ApolloBot v0.2.[/yellow]")
    console.print(
        "Frontier Science now creates living records from authenticated, managed "
        "investigations so ownership, budgets, provenance, review, and DOI state remain linked."
    )
    platform_url = os.getenv("FRONTIER_PLATFORM_URL", "").rstrip("/")
    destination = platform_url or "your managed Frontier Science deployment"
    console.print(
        f"[bold]Start or open the investigation at {destination}, then use "
        "Create research record after the run completes.[/bold]"
    )
    raise click.ClickException("Legacy shared-secret journal submission is disabled")


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
    platform_url = os.getenv("FRONTIER_PLATFORM_URL", "").rstrip("/")
    if platform_url:
        console.print(f"[dim]No open calls. Check {platform_url}/compute[/dim]")
    else:
        console.print("[dim]No open calls. Configure FRONTIER_PLATFORM_URL to browse calls.[/dim]")
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

    console.print(
        f"\n[bold]Notification channels[/bold] (heartbeat: {notif.heartbeat_interval}s)\n"
    )

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

    console.print(
        f"\n[green]>[/green] Added {ch_type} channel. Run [bold]apollo notify test[/bold] to verify."
    )


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio", "streamable-http", "sse"]),
    default="stdio",
    help="MCP transport protocol",
)
@click.option("--host", default="0.0.0.0", help="Host for HTTP/SSE transport")
@click.option("--port", type=int, default=8080, help="Port for HTTP/SSE transport")
def serve(transport, host, port):
    """Start ApolloBot as an MCP server for AI-to-AI integration."""
    from apollobot.server.app import mcp as mcp_server

    console.print(f"[bold green]ApolloBot MCP Server[/bold green]")
    console.print(f"  Transport: {transport}")
    if transport != "stdio":
        console.print(f"  Endpoint:  http://{host}:{port}/mcp")
    console.print()

    if transport == "stdio":
        mcp_server.run(transport="stdio")
    else:
        mcp_server.run(transport=transport, host=host, port=port)


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


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _parse_since(since_str: str):
    """Parse a --since value into a datetime cutoff.

    Accepts ISO date strings like '2026-01-01' or duration strings like '30d'.
    Returns a datetime or None if the string is empty.
    """
    import re
    from datetime import datetime, timedelta, timezone

    if not since_str:
        return None

    # Duration pattern: digits followed by 'd' (days)
    m = re.fullmatch(r"(\d+)d", since_str.strip())
    if m:
        days = int(m.group(1))
        return datetime.now(timezone.utc) - timedelta(days=days)

    # Try ISO date
    try:
        dt = datetime.fromisoformat(since_str.strip())
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        raise click.BadParameter(
            f"Cannot parse '{since_str}'. Use a date (2026-01-01) or duration (30d).",
            param_hint="--since",
        )


def _human_size(num_bytes: int) -> str:
    """Format a byte count as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024  # type: ignore[assignment]
    return f"{num_bytes:.1f} TB"


@main.command()
@click.option(
    "--since",
    default="",
    help="Export sessions since date (e.g. 2026-01-01) or duration (e.g. 30d)",
)
@click.option(
    "--output", "-o", default="", help="Output file path (default: apollobot-export-{date}.tar.gz)"
)
@click.option("--format", "fmt", type=click.Choice(["tar.gz", "zip"]), default="tar.gz")
@click.option("--include-db", is_flag=True, help="Include runtime database")
def export(since, output, fmt, include_db):
    """Export research sessions and data as a portable archive."""
    import io
    import tarfile
    import zipfile
    from datetime import datetime, timezone

    from apollobot.core import APOLLO_HOME, APOLLO_SESSIONS_DIR, load_config

    config = load_config()
    sessions_dir = Path(config.output_dir) if config.output_dir else Path(APOLLO_SESSIONS_DIR)

    if not sessions_dir.exists():
        console.print("[red]No sessions directory found.[/red]")
        sys.exit(1)

    # Parse --since cutoff
    cutoff = _parse_since(since)

    # Collect matching session directories
    session_dirs = []
    for d in sorted(sessions_dir.iterdir()):
        if not d.is_dir():
            continue
        if cutoff is not None:
            mtime = datetime.fromtimestamp(d.stat().st_mtime, tz=timezone.utc)
            if mtime < cutoff:
                continue
        session_dirs.append(d)

    if not session_dirs:
        console.print("[yellow]No sessions matched the filter.[/yellow]")
        sys.exit(0)

    # Determine output path
    today = datetime.now().strftime("%Y-%m-%d")
    if not output:
        ext = ".tar.gz" if fmt == "tar.gz" else ".zip"
        output = f"apollobot-export-{today}{ext}"
    out_path = Path(output).resolve()

    # Build manifest
    manifest = {
        "export_date": datetime.now(timezone.utc).isoformat(),
        "apollobot_version": __version__,
        "session_count": len(session_dirs),
        "sessions": [d.name for d in session_dirs],
        "include_db": include_db,
        "format": fmt,
    }
    if since:
        manifest["since_filter"] = since

    # Runtime database path
    runtime_db = Path(APOLLO_HOME) / "runtime.db"

    if fmt == "tar.gz":
        with tarfile.open(str(out_path), "w:gz") as tar:
            # Add sessions
            for d in session_dirs:
                arcname = f"sessions/{d.name}"
                tar.add(str(d), arcname=arcname)

            # Optionally add runtime database
            if include_db and runtime_db.exists():
                tar.add(str(runtime_db), arcname="runtime.db")
            elif include_db:
                console.print(
                    f"[yellow]Warning: runtime database not found at {runtime_db}[/yellow]"
                )

            # Add manifest
            manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
            info = tarfile.TarInfo(name="manifest.json")
            info.size = len(manifest_bytes)
            tar.addfile(info, io.BytesIO(manifest_bytes))
    else:
        with zipfile.ZipFile(str(out_path), "w", zipfile.ZIP_DEFLATED) as zf:
            # Add sessions
            for d in session_dirs:
                for file_path in d.rglob("*"):
                    if file_path.is_file():
                        arcname = f"sessions/{d.name}/{file_path.relative_to(d)}"
                        zf.write(str(file_path), arcname)

            # Optionally add runtime database
            if include_db and runtime_db.exists():
                zf.write(str(runtime_db), "runtime.db")
            elif include_db:
                console.print(
                    f"[yellow]Warning: runtime database not found at {runtime_db}[/yellow]"
                )

            # Add manifest
            manifest_json = json.dumps(manifest, indent=2)
            zf.writestr("manifest.json", manifest_json)

    archive_size = out_path.stat().st_size
    console.print(
        f"Exported {len(session_dirs)} sessions to {out_path} ({_human_size(archive_size)})"
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@main.command()
@click.option("--period", default="7d", help="Report period (e.g. 1d, 7d, 30d, all)")
@click.option("--format", "fmt", type=click.Choice(["text", "json", "md"]), default="text")
@click.option("--output", "-o", "report_output", default="", help="Save report to file")
def report(period, fmt, report_output):
    """Generate a research performance report."""
    from apollobot.runtime.metrics import compute_metrics
    from apollobot.runtime.trajectory import ResearchTrajectory
    from apollobot.runtime.storage import RunnerStorage
    from apollobot.core import APOLLO_HOME

    db_path = str(APOLLO_HOME / "runtime.db")
    try:
        storage = RunnerStorage(db_path)
    except Exception:
        console.print("[red]No runtime database found. Run 'apollo run' first.[/red]")
        return

    metrics = compute_metrics(storage)
    trajectory = ResearchTrajectory(storage)
    analysis = trajectory.analyze()
    storage.close()

    if fmt == "json":
        import dataclasses

        data = {
            "metrics": dataclasses.asdict(metrics),
            "trajectory": {
                "total_papers": analysis.total_papers,
                "total_cost": analysis.total_cost,
                "overall_quality": analysis.overall_quality,
                "best_domains": analysis.best_performing_domains,
                "underexplored": analysis.underexplored_domains,
                "quality_trend": analysis.avg_quality_trend,
                "cost_trend": analysis.cost_efficiency_trend,
                "recommendations": analysis.recommendations,
            },
        }
        report_text = json.dumps(data, indent=2)
        if report_output:
            Path(report_output).write_text(report_text)
            console.print(f"[green]>[/green] Report saved to {report_output}")
        else:
            console.print(report_text)
        return

    lines = []
    if fmt == "md":
        lines.append("# ApolloBot Research Report\n")
        lines.append(f"**Period**: {period}\n")
    else:
        lines.append(f"ApolloBot Research Report (period: {period})")
        lines.append("=" * 50)

    lines.append(f"\nPapers completed: {metrics.completed_sessions}")
    lines.append(f"Papers failed:    {metrics.failed_sessions}")
    lines.append(f"Completion rate:  {metrics.completion_rate:.0%}")
    lines.append(f"Papers/day:       {metrics.papers_per_day}")
    lines.append(f"")
    lines.append(f"Total cost:       ${metrics.total_cost_usd:.2f}")
    lines.append(f"Avg cost/paper:   ${metrics.avg_cost_per_paper:.2f}")
    lines.append(f"")
    lines.append(f"Avg quality:      {metrics.avg_translation_score}/10")
    lines.append(f"High quality:     {metrics.high_quality_papers} papers (score >= 7)")
    lines.append(f"Reputation:       {metrics.reputation_score}/100")
    lines.append(f"")
    lines.append(f"Quality trend:    {analysis.avg_quality_trend}")
    lines.append(f"Cost trend:       {analysis.cost_efficiency_trend}")

    if analysis.best_performing_domains:
        lines.append(f"\nBest domains: {', '.join(analysis.best_performing_domains)}")
    if analysis.underexplored_domains:
        lines.append(f"Unexplored:   {', '.join(analysis.underexplored_domains)}")

    if analysis.recommendations:
        lines.append(f"\nRecommendations:")
        for r in analysis.recommendations:
            lines.append(f"  - {r}")

    report_text = "\n".join(lines)

    if report_output:
        Path(report_output).write_text(report_text)
        console.print(f"[green]>[/green] Report saved to {report_output}")
    else:
        console.print(report_text)


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--mark-failed", is_flag=True, help="Mark incomplete sessions as failed instead of resuming"
)
def resume(mark_failed):
    """Resume or clean up incomplete sessions from a crashed runtime."""
    from apollobot.runtime.storage import RunnerStorage
    from apollobot.core import APOLLO_HOME

    db_path = str(APOLLO_HOME / "runtime.db")
    try:
        storage = RunnerStorage(db_path)
    except Exception:
        console.print("[red]No runtime database found.[/red]")
        return

    active = storage.get_active_sessions()
    if not active:
        console.print("[dim]No incomplete sessions found.[/dim]")
        storage.close()
        return

    console.print(f"\n[bold]Found {len(active)} incomplete session(s):[/bold]\n")
    for s in active:
        console.print(f"  {s.session_id} [{s.domain}:{s.phase}] — {s.objective[:60]}")

    if mark_failed:
        for s in active:
            storage.update_session(s.session_id, phase="failed")
        console.print(f"\n[yellow]Marked {len(active)} sessions as failed.[/yellow]")
    else:
        console.print(
            "\n[dim]These sessions will be visible to the brain on next runtime start.[/dim]"
        )
        console.print(
            "[dim]Use --mark-failed to clean them up, or start the runtime to resume.[/dim]"
        )

    storage.close()


# ---------------------------------------------------------------------------
# Monitor (live dashboard)
# ---------------------------------------------------------------------------


@main.command()
@click.option("--refresh", type=float, default=2.0, help="Refresh interval in seconds")
@click.option("--health-url", default="http://localhost:8080/health", help="Health endpoint URL")
def monitor(refresh, health_url):
    """Live monitoring dashboard for the running runtime."""
    import time as _time
    import urllib.request
    from rich.live import Live
    from rich.panel import Panel

    def fetch_health():
        try:
            req = urllib.request.Request(health_url)
            with urllib.request.urlopen(req, timeout=3) as resp:
                return json.loads(resp.read())
        except Exception:
            return None

    def build_display(data):
        if data is None:
            return Panel(
                f"[red]Cannot connect to runtime[/red]\n[dim]Checking {health_url}[/dim]",
                title="ApolloBot Monitor",
            )

        status_color = "green" if data.get("status") == "healthy" else "red"
        uptime_mins = data.get("uptime_seconds", 0) // 60

        lines = [
            f"[{status_color}]{data.get('status', '?').upper()}[/{status_color}]  |  "
            f"Domain: {data.get('domain', '?')}  |  "
            f"Uptime: {uptime_mins}m  |  "
            f"Watchdog: {data.get('watchdog', '?')}",
            "",
            f"Tick: {data.get('tick_count', 0)}  |  Last: {data.get('last_tick', '?')[:19]}",
            f"Active sessions: {data.get('active_sessions', 0)}  |  "
            f"Total papers: {data.get('total_papers', 0)}",
            f"Daily cost: ${data.get('daily_cost_usd', 0):.2f}",
        ]

        return Panel("\n".join(lines), title="ApolloBot Monitor", border_style=status_color)

    console.print(f"[dim]Monitoring {health_url} (Ctrl+C to stop)[/dim]\n")

    try:
        with Live(build_display(None), refresh_per_second=1, console=console) as live:
            while True:
                data = fetch_health()
                live.update(build_display(data))
                _time.sleep(refresh)
    except KeyboardInterrupt:
        console.print("\n[dim]Monitor stopped.[/dim]")


# ---------------------------------------------------------------------------
# Guardrails management
# ---------------------------------------------------------------------------


@main.group()
def guardrails() -> None:
    """Manage runtime safety constraints."""
    pass


@guardrails.command(name="status")
@click.option("--health-url", default="http://localhost:8080/health", help="Health endpoint URL")
def guardrails_status(health_url):
    """Show current guardrails status from the running runtime."""
    import urllib.request

    try:
        req = urllib.request.Request(health_url)
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
    except Exception:
        console.print("[red]Cannot connect to runtime. Is it running?[/red]")
        return

    console.print("\n[bold]Guardrails Status[/bold]\n")
    console.print(f"  Status:          {data.get('status', '?')}")
    console.print(f"  Daily cost:      ${data.get('daily_cost_usd', 0):.2f}")
    console.print(f"  Active sessions: {data.get('active_sessions', 0)}")
    console.print(f"  Watchdog:        {data.get('watchdog', '?')}")


@guardrails.command(name="set")
@click.option("--daily-budget", type=float, default=None, help="New daily budget in USD")
@click.option("--max-concurrent", type=int, default=None, help="Max concurrent sessions")
@click.option("--emergency-stop", type=bool, default=None, help="Enable/disable emergency stop")
@click.option(
    "--health-url", default="http://localhost:8080", help="Runtime health server base URL"
)
def guardrails_set(daily_budget, max_concurrent, emergency_stop, health_url):
    """Update guardrails constraints on a running runtime."""
    import urllib.request

    updates = {}
    if daily_budget is not None:
        updates["daily_compute_budget_usd"] = daily_budget
    if max_concurrent is not None:
        updates["max_concurrent_sessions"] = max_concurrent
    if emergency_stop is not None:
        updates["emergency_stop"] = emergency_stop

    if not updates:
        console.print(
            "[red]No updates specified. Use --daily-budget, --max-concurrent, or --emergency-stop.[/red]"
        )
        return

    try:
        data = json.dumps(updates).encode()
        req = urllib.request.Request(
            f"{health_url}/guardrails",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            result = json.loads(resp.read())
            console.print(f"[green]>[/green] Guardrails updated: {result}")
    except Exception as e:
        console.print(f"[red]Failed to update guardrails: {e}[/red]")
        console.print(
            "[dim]Make sure the runtime is running and the guardrails endpoint is enabled.[/dim]"
        )


if __name__ == "__main__":
    main()
