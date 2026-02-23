"""
Provenance engine — records every decision, data transformation,
and LLM interaction for full reproducibility and auditability.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class DataLineageEntry(BaseModel):
    """Tracks a single data transformation step."""

    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source: str  # e.g. "GEO:GSE184571"
    operation: str  # e.g. "filter_rows", "normalize", "merge"
    description: str
    input_hash: str = ""
    output_hash: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    script_ref: str = ""  # path to the script that performed this


class LLMCallEntry(BaseModel):
    """Records a single LLM API call."""

    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    provider: str
    model: str
    purpose: str  # e.g. "plan_research", "interpret_results", "draft_section"
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    # We store a summary/hash, not full prompt, to keep logs manageable
    prompt_hash: str = ""
    response_summary: str = ""


class ProvenanceEngine:
    """
    Central provenance tracker for a research session.

    Accumulates three log streams:
    1. execution_log — high-level events (phase transitions, decisions)
    2. data_lineage — every data transformation with hashes
    3. model_calls — every LLM interaction
    """

    def __init__(self, session_dir: Path) -> None:
        self.session_dir = session_dir
        self.provenance_dir = session_dir / "provenance"
        self.provenance_dir.mkdir(parents=True, exist_ok=True)

        self.execution_log: list[dict[str, Any]] = []
        self.data_lineage: list[DataLineageEntry] = []
        self.model_calls: list[LLMCallEntry] = []

    # ------------------------------------------------------------------
    # Execution events
    # ------------------------------------------------------------------

    def log_event(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            **(data or {}),
        }
        self.execution_log.append(entry)

    def log_decision(self, description: str, reasoning: str, alternatives: list[str] | None = None) -> None:
        self.log_event("decision", {
            "description": description,
            "reasoning": reasoning,
            "alternatives_considered": alternatives or [],
        })

    # ------------------------------------------------------------------
    # Data lineage
    # ------------------------------------------------------------------

    def log_data_transform(
        self,
        source: str,
        operation: str,
        description: str,
        input_data: bytes | str | None = None,
        output_data: bytes | str | None = None,
        parameters: dict[str, Any] | None = None,
        script_ref: str = "",
    ) -> None:
        entry = DataLineageEntry(
            source=source,
            operation=operation,
            description=description,
            input_hash=self._hash(input_data) if input_data else "",
            output_hash=self._hash(output_data) if output_data else "",
            parameters=parameters or {},
            script_ref=script_ref,
        )
        self.data_lineage.append(entry)

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    def log_llm_call(
        self,
        provider: str,
        model: str,
        purpose: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        prompt_text: str = "",
        response_summary: str = "",
    ) -> None:
        entry = LLMCallEntry(
            provider=provider,
            model=model,
            purpose=purpose,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            prompt_hash=self._hash(prompt_text) if prompt_text else "",
            response_summary=response_summary[:500],
        )
        self.model_calls.append(entry)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Flush all logs to disk."""
        (self.provenance_dir / "execution_log.json").write_text(
            json.dumps(self.execution_log, indent=2)
        )
        (self.provenance_dir / "data_lineage.json").write_text(
            json.dumps([e.model_dump() for e in self.data_lineage], indent=2)
        )
        (self.provenance_dir / "model_calls.json").write_text(
            json.dumps([e.model_dump() for e in self.model_calls], indent=2)
        )

    # ------------------------------------------------------------------
    # Replication kit
    # ------------------------------------------------------------------

    def generate_replication_kit(self, session_dir: Path) -> None:
        """
        Generate a self-contained replication kit that allows anyone
        to reproduce the research from scratch.
        """
        kit_dir = session_dir / "replication_kit"
        kit_dir.mkdir(parents=True, exist_ok=True)

        # Checksums for all data files
        checksums = {}
        data_dir = session_dir / "data"
        if data_dir.exists():
            for f in data_dir.rglob("*"):
                if f.is_file():
                    checksums[str(f.relative_to(session_dir))] = self._hash(
                        f.read_bytes()
                    )

        (kit_dir / "checksums.sha256").write_text(
            "\n".join(f"{v}  {k}" for k, v in sorted(checksums.items()))
        )

        # Replication script
        (kit_dir / "replicate.sh").write_text(
            "#!/bin/bash\n"
            "set -euo pipefail\n\n"
            "echo 'Installing ApolloBot...'\n"
            "pip install apollobot\n\n"
            "echo 'Replaying research session...'\n"
            f"apollo replay {session_dir / 'provenance' / 'execution_log.json'}\n\n"
            "echo 'Verifying checksums...'\n"
            "sha256sum -c checksums.sha256\n"
            "echo 'Replication complete.'\n"
        )
        (kit_dir / "replicate.sh").chmod(0o755)

    # ------------------------------------------------------------------
    # Cross-mode provenance
    # ------------------------------------------------------------------

    def link_source_session(self, source_session_id: str, source_dir: Path) -> None:
        """Link this session's provenance to a prior mode's output."""
        self.log_event("cross_mode_link", {
            "source_session_id": source_session_id,
            "source_dir": str(source_dir),
        })

        # Copy source provenance summary as reference
        source_prov = source_dir / "provenance" / "execution_log.json"
        if source_prov.exists():
            dest = self.provenance_dir / "source_provenance.json"
            dest.write_text(source_prov.read_text())

    def validate_cross_references(self) -> dict[str, Any]:
        """Validate that cross-mode references are consistent."""
        issues: list[str] = []

        source_prov = self.provenance_dir / "source_provenance.json"
        if source_prov.exists():
            try:
                source_log = json.loads(source_prov.read_text())
                # Check that source session completed successfully
                completed = any(
                    e.get("event") == "phase_completed"
                    for e in source_log
                )
                if not completed:
                    issues.append("Source session has no completed phases")
            except (json.JSONDecodeError, KeyError):
                issues.append("Source provenance file is corrupted")
        else:
            issues.append("No source provenance linked")

        result = {
            "valid": len(issues) == 0,
            "issues": issues,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }
        self.log_event("cross_reference_validation", result)
        return result

    def get_provenance_chain(self) -> list[dict[str, Any]]:
        """Get the full provenance chain across modes."""
        chain = []

        # Check for source provenance
        source_prov = self.provenance_dir / "source_provenance.json"
        if source_prov.exists():
            try:
                chain.append({
                    "type": "source",
                    "log": json.loads(source_prov.read_text()),
                })
            except json.JSONDecodeError:
                pass

        # Add current session's provenance
        chain.append({
            "type": "current",
            "execution_log": self.execution_log,
            "data_lineage": [e.model_dump() for e in self.data_lineage],
            "model_calls": [e.model_dump() for e in self.model_calls],
        })

        return chain

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(data: bytes | str) -> str:
        if isinstance(data, str):
            data = data.encode()
        return hashlib.sha256(data).hexdigest()[:16]
