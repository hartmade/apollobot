"""
Orchestrator — the top-level controller that ties everything together.

This is what runs when the user types `apollo research "..."`, `apollo discover`,
`apollo translate`, `apollo implement`, `apollo commercialize`, or `apollo pipeline`.

It initializes the session, creates the agent pipeline, and
manages the full research lifecycle.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from rich.console import Console

from apollobot.agents import LLMProvider, create_llm
from apollobot.agents.executor import CheckpointHandler, ResearchExecutor
from apollobot.agents.planner import ResearchPlanner
from apollobot.core import ApolloConfig, APOLLO_SESSIONS_DIR, load_config
from apollobot.core.mission import Mission, ResearchMode
from apollobot.core.provenance import ProvenanceEngine
from apollobot.core.session import Phase, Session
from apollobot.core.translation import TranslationReport
from apollobot.mcp import MCPClient, MCPServerInfo
from apollobot.mcp.servers.builtin import get_domain_pack
from apollobot.notifications import (
    EventSeverity,
    EventType,
    NotificationEvent,
    NotificationRouter,
)
from apollobot.notifications.channels.console import ConsoleChannel
from apollobot.notifications.channels.webhook import WebhookChannel
from apollobot.notifications.checkpoint import ChannelCheckpointHandler
from apollobot.notifications.config import NotificationsConfig
from apollobot.notifications.heartbeat import HeartbeatMonitor

console = Console()


class InteractiveCheckpointHandler(CheckpointHandler):
    """Checkpoint handler that prompts the user in the terminal."""

    async def request_approval(self, phase: str, summary: str) -> bool:
        console.print(f"\n[bold yellow]Checkpoint: {phase}[/bold yellow]")
        console.print(f"  {summary}")
        response = console.input("  [approve/deny/modify]: ").strip().lower()
        return response in ("approve", "a", "yes", "y", "")

    async def notify(self, phase: str, summary: str) -> None:
        console.print(f"\n[bold blue]i {phase}[/bold blue]: {summary}")


class Orchestrator:
    """
    Top-level research orchestrator.

    Supports all pipeline modes:
    - Discover (research) — original research
    - Translate — research-to-spec conversion
    - Implement — spec-to-code conversion
    - Commercialize — market analysis
    - Pipeline — full Discover → Translate → Implement → Commercialize

    Usage:
        orchestrator = Orchestrator()
        session = await orchestrator.run(mission)
        # or mode-specific:
        session = await orchestrator.run_discover(mission)
        session = await orchestrator.run_translate(mission)
        session = await orchestrator.run_pipeline(mission)
    """

    def __init__(
        self,
        config: ApolloConfig | None = None,
        interactive: bool = True,
    ) -> None:
        self.config = config or load_config()
        self.interactive = interactive

        # Initialize LLM
        self.llm: LLMProvider = create_llm(
            provider=self.config.api.default_provider,
            api_key=self.config.api.get_key(),
        )

        # Initialize MCP client
        self.mcp = MCPClient()

        # Build notification router from config
        self.router = self._build_router(self.config.notifications)

        # Checkpoint handler: channels if configured, else interactive/auto
        if self.router.channels:
            self.checkpoint: CheckpointHandler = ChannelCheckpointHandler(
                self.router, session_id=""
            )
        elif interactive:
            self.checkpoint = InteractiveCheckpointHandler()
        else:
            self.checkpoint = CheckpointHandler()

    # ------------------------------------------------------------------
    # Main entry point — routes to mode-specific executors
    # ------------------------------------------------------------------

    async def run(self, mission: Mission) -> Session:
        """
        Execute a session based on mission mode.

        Routes to mode-specific executors based on mission.mode.
        """
        mode = mission.mode

        if mode == ResearchMode.TRANSLATE:
            return await self.run_translate(mission)
        elif mode == ResearchMode.IMPLEMENT:
            return await self.run_implement(mission)
        elif mode == ResearchMode.COMMERCIALIZE:
            return await self.run_commercialize(mission)
        elif mode == ResearchMode.PIPELINE:
            return await self.run_pipeline(mission)
        else:
            # Discover mode (hypothesis, exploratory, meta-analysis, etc.)
            return await self.run_discover(mission)

    # ------------------------------------------------------------------
    # Discover mode (original research)
    # ------------------------------------------------------------------

    async def run_discover(self, mission: Mission) -> Session:
        """
        Execute a Discover (research) session.

        This is the original research mode from v0.1.0.
        """
        console.print(f"\n[bold green]ApolloBot — Discover Mode[/bold green]")
        console.print(f"[dim]Session: {mission.id}[/dim]")
        console.print(f"[bold]Objective:[/bold] {mission.objective}\n")

        session, provenance, heartbeat = await self._setup_session(mission)

        # Connect MCP servers for the domain
        console.print("[dim]Connecting to data sources...[/dim]")
        await self._connect_mcp_servers(mission.domain)

        available_servers = [s.name for s in self.mcp.get_servers(mission.domain)]
        console.print(f"[green]>[/green] Connected to {len(available_servers)} MCP servers")
        for name in available_servers:
            console.print(f"  [dim]- {name}[/dim]")

        # Plan
        console.print("\n[bold]Planning research...[/bold]")
        planner = ResearchPlanner(self.llm, provenance)
        plan = await planner.plan(mission, available_servers)

        console.print(f"[green]>[/green] Plan created")
        console.print(f"  [dim]- {len(plan.literature_queries)} literature queries[/dim]")
        console.print(f"  [dim]- {len(plan.data_requirements)} data requirements[/dim]")
        console.print(f"  [dim]- {len(plan.analysis_steps)} analysis steps[/dim]")
        console.print(f"  [dim]- Estimated cost: ${plan.estimated_compute_cost:.2f}[/dim]")
        console.print(f"  [dim]- Estimated time: {plan.estimated_time_hours:.1f}h[/dim]")

        # Approval for plan
        if self.interactive:
            console.print(f"\n[bold]Research approach:[/bold] {plan.summary}")
            if not await self.checkpoint.request_approval("plan", plan.summary):
                console.print("[red]Research cancelled by user.[/red]")
                session.current_phase = Phase.CANCELLED
                await heartbeat.stop()
                await self.router.disconnect_all()
                return session

        # Execute
        console.print("\n[bold]Executing research plan...[/bold]\n")
        executor = ResearchExecutor(
            llm=self.llm,
            mcp=self.mcp,
            provenance=provenance,
            checkpoint_handler=self.checkpoint,
        )

        session = await executor.execute(session, plan)

        await self._teardown_session(session, mission, heartbeat)
        return session

    # ------------------------------------------------------------------
    # Translate mode
    # ------------------------------------------------------------------

    async def run_translate(self, mission: Mission) -> Session:
        """
        Execute a Translate session.

        Converts research findings into implementation specifications.
        Requires source_session or source_paper to be set on the mission.
        """
        from apollobot.agents.translator import ResearchTranslator

        console.print(f"\n[bold green]ApolloBot — Translate Mode[/bold green]")
        console.print(f"[dim]Session: {mission.id}[/dim]")

        session, provenance, heartbeat = await self._setup_session(mission)

        # Load source session if specified
        if mission.source_session:
            source_dir = Path(self.config.output_dir) / mission.source_session
            if source_dir.exists():
                source_session = Session.load_state(source_dir)
                session.literature_corpus = source_session.literature_corpus
                session.key_findings = source_session.key_findings
                session.translation_scores = source_session.translation_scores
                provenance.link_source_session(mission.source_session, source_dir)
                console.print(f"[green]>[/green] Loaded source session: {mission.source_session}")
            else:
                console.print(f"[yellow]Warning: Source session {mission.source_session} not found[/yellow]")

        # Initialize translation report
        report = TranslationReport(
            id=f"tr-{mission.id}",
            source_session_id=mission.source_session,
            source_paper_doi=mission.source_paper,
        )
        session.translation_report = report.model_dump()

        # Connect MCP servers
        await self._connect_mcp_servers(mission.domain)

        # Run translation
        translator = ResearchTranslator(
            llm=self.llm,
            mcp=self.mcp,
            provenance=provenance,
            checkpoint_handler=self.checkpoint,
        )

        session = await translator.translate(session)

        await self._teardown_session(session, mission, heartbeat)
        return session

    # ------------------------------------------------------------------
    # Implement mode
    # ------------------------------------------------------------------

    async def run_implement(self, mission: Mission) -> Session:
        """
        Execute an Implement session.

        Builds production code from a translation spec.
        """
        from apollobot.agents.implementor import ResearchImplementor

        console.print(f"\n[bold green]ApolloBot — Implement Mode[/bold green]")
        console.print(f"[dim]Session: {mission.id}[/dim]")

        session, provenance, heartbeat = await self._setup_session(mission)

        # Load source session with translation report
        if mission.source_session:
            source_dir = Path(self.config.output_dir) / mission.source_session
            if source_dir.exists():
                source_session = Session.load_state(source_dir)
                session.translation_report = source_session.translation_report
                session.key_findings = source_session.key_findings
                provenance.link_source_session(mission.source_session, source_dir)
                console.print(f"[green]>[/green] Loaded source session: {mission.source_session}")

        # Connect MCP servers
        await self._connect_mcp_servers(mission.domain)

        # Run implementation
        implementor = ResearchImplementor(
            llm=self.llm,
            mcp=self.mcp,
            provenance=provenance,
            checkpoint_handler=self.checkpoint,
        )

        session = await implementor.implement(session)

        await self._teardown_session(session, mission, heartbeat)
        return session

    # ------------------------------------------------------------------
    # Commercialize mode
    # ------------------------------------------------------------------

    async def run_commercialize(self, mission: Mission) -> Session:
        """
        Execute a Commercialize session.

        Produces market analysis and go-to-market plan.
        """
        from apollobot.agents.commercializer import Commercializer

        console.print(f"\n[bold green]ApolloBot — Commercialize Mode[/bold green]")
        console.print(f"[dim]Session: {mission.id}[/dim]")

        session, provenance, heartbeat = await self._setup_session(mission)

        # Load source session with implementation
        if mission.source_session:
            source_dir = Path(self.config.output_dir) / mission.source_session
            if source_dir.exists():
                source_session = Session.load_state(source_dir)
                session.translation_report = source_session.translation_report
                session.key_findings = source_session.key_findings
                provenance.link_source_session(mission.source_session, source_dir)
                console.print(f"[green]>[/green] Loaded source session: {mission.source_session}")

        # Connect MCP servers
        await self._connect_mcp_servers(mission.domain)

        # Run commercialization
        commercializer = Commercializer(
            llm=self.llm,
            mcp=self.mcp,
            provenance=provenance,
            checkpoint_handler=self.checkpoint,
        )

        session = await commercializer.commercialize(session)

        await self._teardown_session(session, mission, heartbeat)
        return session

    # ------------------------------------------------------------------
    # Pipeline mode — chains all modes with checkpoints
    # ------------------------------------------------------------------

    async def run_pipeline(
        self,
        mission: Mission,
        auto_translate: bool = False,
    ) -> Session:
        """
        Execute the full pipeline: Discover → Translate → Implement → Commercialize.

        Human checkpoints at each mode boundary. With auto_translate=True,
        automatically proceeds to Translate if translation score >= 7.
        """
        console.print(f"\n[bold green]ApolloBot — Full Pipeline Mode[/bold green]")
        console.print(f"[dim]Session: {mission.id}[/dim]")
        console.print(f"[bold]Objective:[/bold] {mission.objective}\n")

        # Phase 1: Discover
        console.print("[bold]Phase 1/4: Discover[/bold]")
        discover_mission = Mission.from_objective(
            mission.objective,
            mode=mission.metadata.get("discover_mode", "hypothesis"),
            domain=mission.domain,
        )
        discover_session = await self.run_discover(discover_mission)

        if discover_session.current_phase != Phase.COMPLETE:
            console.print("[red]Discover phase failed. Pipeline halted.[/red]")
            return discover_session

        # Check translation potential
        avg_score = discover_session.translation_scores.get("average", 0)
        console.print(f"\n[bold]Translation potential: {avg_score:.1f}/10[/bold]")

        proceed_translate = False
        if auto_translate and avg_score >= 7.0:
            console.print("[green]Auto-translate triggered (score >= 7)[/green]")
            proceed_translate = True
        elif self.interactive:
            proceed_translate = await self.checkpoint.request_approval(
                "pipeline_translate",
                f"Proceed to Translate mode? (score: {avg_score:.1f}/10)"
            )

        if not proceed_translate:
            console.print("[dim]Pipeline stopped after Discover.[/dim]")
            return discover_session

        # Phase 2: Translate
        console.print("\n[bold]Phase 2/4: Translate[/bold]")
        translate_mission = Mission(
            objective=f"Translate: {mission.objective}",
            mode=ResearchMode.TRANSLATE,
            domain=mission.domain,
            source_session=discover_mission.id,
        )
        translate_session = await self.run_translate(translate_mission)

        if translate_session.current_phase != Phase.COMPLETE:
            console.print("[red]Translate phase failed. Pipeline halted.[/red]")
            return translate_session

        # Checkpoint before Implement
        if self.interactive:
            proceed_implement = await self.checkpoint.request_approval(
                "pipeline_implement",
                "Proceed to Implement mode?"
            )
            if not proceed_implement:
                console.print("[dim]Pipeline stopped after Translate.[/dim]")
                return translate_session

        # Phase 3: Implement
        console.print("\n[bold]Phase 3/4: Implement[/bold]")
        implement_mission = Mission(
            objective=f"Implement: {mission.objective}",
            mode=ResearchMode.IMPLEMENT,
            domain=mission.domain,
            source_session=translate_mission.id,
        )
        implement_session = await self.run_implement(implement_mission)

        if implement_session.current_phase != Phase.COMPLETE:
            console.print("[red]Implement phase failed. Pipeline halted.[/red]")
            return implement_session

        # Checkpoint before Commercialize
        if self.interactive:
            proceed_comm = await self.checkpoint.request_approval(
                "pipeline_commercialize",
                "Proceed to Commercialize mode?"
            )
            if not proceed_comm:
                console.print("[dim]Pipeline stopped after Implement.[/dim]")
                return implement_session

        # Phase 4: Commercialize
        console.print("\n[bold]Phase 4/4: Commercialize[/bold]")
        comm_mission = Mission(
            objective=f"Commercialize: {mission.objective}",
            mode=ResearchMode.COMMERCIALIZE,
            domain=mission.domain,
            source_session=implement_mission.id,
        )
        comm_session = await self.run_commercialize(comm_mission)

        console.print("\n[bold green]Pipeline complete![/bold green]")
        console.print(f"  Discover: {discover_mission.id}")
        console.print(f"  Translate: {translate_mission.id}")
        console.print(f"  Implement: {implement_mission.id}")
        console.print(f"  Commercialize: {comm_mission.id}")

        return comm_session

    # ------------------------------------------------------------------
    # Shared setup / teardown
    # ------------------------------------------------------------------

    async def _setup_session(
        self, mission: Mission
    ) -> tuple[Session, ProvenanceEngine, HeartbeatMonitor]:
        """Common setup for all modes."""
        if isinstance(self.checkpoint, ChannelCheckpointHandler):
            self.checkpoint.session_id = mission.id

        await self.router.connect_all()

        session = Session(mission=mission)
        session.mission.metadata["output_dir"] = self.config.output_dir
        session.init_directories()

        provenance = ProvenanceEngine(session.session_dir)
        provenance.log_event("session_started", {
            "mission_id": mission.id,
            "objective": mission.objective,
            "mode": mission.mode.value,
            "domain": mission.domain,
        })

        await self.router.dispatch(NotificationEvent(
            event_type=EventType.SESSION_STARTED,
            session_id=mission.id,
            title=f"{mission.mode.value.title()} session started",
            summary=f"Objective: {mission.objective}",
            details={"mode": mission.mode.value, "domain": mission.domain},
        ))

        heartbeat = HeartbeatMonitor(
            self.router,
            session_id=mission.id,
            interval=self.config.notifications.heartbeat_interval,
        )
        await heartbeat.start()

        return session, provenance, heartbeat

    async def _teardown_session(
        self, session: Session, mission: Mission, heartbeat: HeartbeatMonitor
    ) -> None:
        """Common teardown for all modes."""
        heartbeat.update_status(
            phase=session.current_phase.value,
            datasets=len(session.datasets),
            cost=session.cost.total_cost,
        )
        await heartbeat.stop()

        if session.current_phase.value == "complete":
            await self.router.dispatch(NotificationEvent(
                event_type=EventType.SESSION_COMPLETED,
                session_id=mission.id,
                title=f"{mission.mode.value.title()} session complete",
                summary=f"Cost: ${session.cost.total_cost:.2f} | LLM calls: {session.cost.llm_calls}",
                details={
                    "cost_usd": session.cost.total_cost,
                    "llm_calls": session.cost.llm_calls,
                    "output_dir": str(session.session_dir),
                },
            ))
        else:
            await self.router.dispatch(NotificationEvent(
                event_type=EventType.SESSION_FAILED,
                severity=EventSeverity.ERROR,
                session_id=mission.id,
                title=f"{mission.mode.value.title()} session failed",
                summary=f"Ended in phase: {session.current_phase.value}",
                details={"final_phase": session.current_phase.value},
            ))

        await self.router.disconnect_all()
        self._print_summary(session)

    # ------------------------------------------------------------------
    # Router / MCP / display helpers
    # ------------------------------------------------------------------

    def _build_router(self, notif_config: NotificationsConfig) -> NotificationRouter:
        """Build the notification router from config."""
        router = NotificationRouter()

        if not notif_config.enabled:
            return router

        for ch_cfg in notif_config.channels:
            if not ch_cfg.enabled:
                continue

            channel = None
            extras = {
                k: v
                for k, v in ch_cfg.model_dump().items()
                if k not in ("type", "enabled", "events")
            }

            if ch_cfg.type == "console":
                channel = ConsoleChannel()
            elif ch_cfg.type == "webhook":
                channel = WebhookChannel(
                    url=extras.get("url", ""),
                    secret=extras.get("secret", ""),
                    headers=extras.get("headers"),
                )
            elif ch_cfg.type == "telegram":
                from apollobot.notifications.channels.telegram import TelegramChannel

                channel = TelegramChannel(
                    token=extras.get("token", ""),
                    chat_id=extras.get("chat_id", ""),
                )
            elif ch_cfg.type == "discord":
                from apollobot.notifications.channels.discord import DiscordChannel

                channel = DiscordChannel(
                    webhook_url=extras.get("webhook_url", ""),
                    bot_token=extras.get("bot_token", ""),
                    channel_id=extras.get("channel_id", ""),
                )
            elif ch_cfg.type == "slack":
                from apollobot.notifications.channels.slack import SlackChannel

                channel = SlackChannel(webhook_url=extras.get("webhook_url", ""))
            elif ch_cfg.type == "google_chat":
                from apollobot.notifications.channels.google_chat import GoogleChatChannel

                channel = GoogleChatChannel(webhook_url=extras.get("webhook_url", ""))
            elif ch_cfg.type == "email":
                from apollobot.notifications.channels.email import EmailChannel

                channel = EmailChannel(
                    smtp_host=extras.get("smtp_host", "localhost"),
                    smtp_port=extras.get("smtp_port", 587),
                    username=extras.get("username", ""),
                    password=extras.get("password", ""),
                    from_addr=extras.get("from_addr", ""),
                    to_addrs=extras.get("to_addrs", []),
                    use_tls=extras.get("use_tls", True),
                )

            if channel:
                router.register(channel, ch_cfg.events)

        return router

    async def _connect_mcp_servers(self, domain: str) -> None:
        """Register and connect domain-specific MCP servers."""
        servers = get_domain_pack(domain)
        for srv in servers:
            self.mcp.register(MCPServerInfo(
                name=srv.name,
                url=srv.url,
                description=srv.description,
                domain=srv.domain if srv.domain != "shared" else domain,
            ))

        # Also register any custom servers from config
        for custom in self.config.custom_servers:
            self.mcp.register_from_config(custom)

    def _print_summary(self, session: Session) -> None:
        """Print a summary of the completed session."""
        console.print("\n" + "=" * 60)

        if session.current_phase.value == "complete":
            console.print("[bold green]Session complete![/bold green]")
        elif session.current_phase.value == "failed":
            console.print("[bold red]Session failed[/bold red]")
        else:
            console.print(f"[bold yellow]Session ended in phase: {session.current_phase.value}[/bold yellow]")

        console.print(f"\n[bold]Mode:[/bold] {session.mission.mode.value}")
        console.print(f"[bold]Cost:[/bold] ${session.cost.total_cost:.2f}")
        console.print(f"[bold]LLM calls:[/bold] {session.cost.llm_calls}")
        console.print(f"[bold]Output:[/bold] {session.session_dir}")

        # List key output files
        if session.session_dir.exists():
            for f in sorted(session.session_dir.rglob("*")):
                if f.is_file() and f.suffix in (".tex", ".md", ".pdf", ".json"):
                    rel = f.relative_to(session.session_dir)
                    console.print(f"  [dim]{rel}[/dim]")

        # Translation scores if available
        if session.translation_scores:
            avg = session.translation_scores.get("average", 0)
            console.print(f"\n[bold]Translation potential:[/bold] {avg:.1f}/10")
            if avg >= 7.0:
                console.print("[green]Flagged as translation candidate[/green]")

        console.print(f"\n[dim]Submit to Frontier Science Journal: apollo submit --session {session.mission.id}[/dim]")
        console.print("=" * 60)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


async def run_research(
    objective: str,
    *,
    mode: str = "hypothesis",
    domain: str = "bioinformatics",
    mission_file: str = "",
    interactive: bool = True,
) -> Session:
    """
    Convenience function to run research from a simple objective string.
    This is the main programmatic API for ApolloBot (Discover mode).
    """
    if mission_file:
        mission = Mission.from_yaml(mission_file)
    else:
        mission = Mission.from_objective(objective, mode=mode, domain=domain)

    orchestrator = Orchestrator(interactive=interactive)
    return await orchestrator.run(mission)


async def run_translate(
    *,
    session_id: str = "",
    paper_doi: str = "",
    domain: str = "bioinformatics",
    interactive: bool = True,
) -> Session:
    """Convenience function to run Translate mode."""
    mission = Mission(
        objective=f"Translate session {session_id or paper_doi}",
        mode=ResearchMode.TRANSLATE,
        domain=domain,
        source_session=session_id,
        source_paper=paper_doi,
    )
    orchestrator = Orchestrator(interactive=interactive)
    return await orchestrator.run(mission)


async def run_implement(
    *,
    session_id: str = "",
    domain: str = "bioinformatics",
    interactive: bool = True,
) -> Session:
    """Convenience function to run Implement mode."""
    mission = Mission(
        objective=f"Implement from {session_id}",
        mode=ResearchMode.IMPLEMENT,
        domain=domain,
        source_session=session_id,
    )
    orchestrator = Orchestrator(interactive=interactive)
    return await orchestrator.run(mission)


async def run_commercialize(
    *,
    session_id: str = "",
    domain: str = "bioinformatics",
    interactive: bool = True,
) -> Session:
    """Convenience function to run Commercialize mode."""
    mission = Mission(
        objective=f"Commercialize from {session_id}",
        mode=ResearchMode.COMMERCIALIZE,
        domain=domain,
        source_session=session_id,
    )
    orchestrator = Orchestrator(interactive=interactive)
    return await orchestrator.run(mission)


async def run_pipeline(
    objective: str,
    *,
    domain: str = "bioinformatics",
    auto_translate: bool = False,
    interactive: bool = True,
) -> Session:
    """Convenience function to run the full pipeline."""
    mission = Mission(
        objective=objective,
        mode=ResearchMode.PIPELINE,
        domain=domain,
    )
    orchestrator = Orchestrator(interactive=interactive)
    return await orchestrator.run_pipeline(mission, auto_translate=auto_translate)
