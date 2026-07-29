"""
Continuous runtime for ApolloBot — autonomous, self-scheduling research.

Instead of one-shot "give it a question, get back a paper", the runtime
continuously discovers questions, investigates them, and produces papers.

Inspired by OpenCat's AgentRunner architecture, adapted for scientific research.

Components:
- ResearchRunner: Main execution loop (setTimeout-chain, adaptive intervals)
- ResearchBrain: Autonomous reasoning engine (context assembly + LLM decisions)
- ResearchGuardrails: Safety guardrails (budget caps, domain restrictions)
- Watchdog: Circuit breaker for fault tolerance
- RunnerStorage: SQLite persistence for memory and history
- HealthServer: HTTP health check endpoint
- ResearchTrajectory: Cross-session learning and pattern analysis
- RuntimeProvenanceLogger: Immutable audit trail for runtime decisions
- NotifyBridge: Runtime events → notification channels
- RemoteLogTransport: Structured log shipping to remote endpoints
"""

from __future__ import annotations

from apollobot.runtime.config import RuntimeConfig
from apollobot.runtime.events import RunnerEvent, RunnerEventEmitter, RunnerEventType
from apollobot.runtime.health import HealthServer
from apollobot.runtime.metrics import ResearchMetrics, compute_metrics
from apollobot.runtime.notify_bridge import NotifyBridge
from apollobot.runtime.pidfile import PidFile
from apollobot.runtime.trajectory import ResearchTrajectory
from apollobot.runtime.provenance import RuntimeProvenanceLogger
from apollobot.runtime.remote_log import RemoteLogTransport
from apollobot.runtime.runner import ResearchRunner
from apollobot.runtime.types import BrainAction, BrainDecision, RunnerState

__all__ = [
    "BrainAction",
    "BrainDecision",
    "HealthServer",
    "NotifyBridge",
    "PidFile",
    "RemoteLogTransport",
    "ResearchMetrics",
    "ResearchTrajectory",
    "ResearchRunner",
    "RunnerEvent",
    "RunnerEventEmitter",
    "RunnerEventType",
    "RunnerState",
    "RuntimeConfig",
    "RuntimeProvenanceLogger",
    "compute_metrics",
]
