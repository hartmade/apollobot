"""Fast, bounded question framing for the free Frontier Science gateway."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import unicodedata

from apollobot.agents import create_llm
from apollobot.core import ApolloConfig, load_config
from apollobot.service.models import QuestionCheck, RunEstimate

logger = logging.getLogger(__name__)

QUESTION_DOMAINS = {
    "biology",
    "medicine",
    "physics",
    "climate",
    "chemistry",
    "computer-science",
    "economics",
    "interdisciplinary",
}
APOLLO_DOMAINS = {"bioinformatics", "physics", "cs_ml", "comp_chem", "economics"}
ANSWERABILITY_STATES = {"investigable", "needs-refinement", "unsafe"}
NOVELTY_STATES = {"open", "adjacent-work-likely", "well-studied"}
RESEARCH_MODES = {"hypothesis", "exploratory", "meta-analysis", "replication", "simulation"}

BLOCKED_QUESTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "biological-harm",
        re.compile(
            r"\b(?:engineer|modify|optimi[sz]e|design|select)\b.{0,80}"
            r"\b(?:pathogen|virus|bacteri(?:a|um)|fungus)\b.{0,80}"
            r"\b(?:virulence|pathogenicity|transmissibility|host range|immune evasion|weapon)",
            re.IGNORECASE,
        ),
    ),
    (
        "biological-harm",
        re.compile(
            r"\b(?:make|build|create|assemble|synthesi[sz]e|aerosoli[sz]e)\b.{0,70}"
            r"\b(?:bioweapon|weaponized pathogen|harmful pathogen|lethal virus|"
            r"biological weapon)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "weapons-or-toxic-agents",
        re.compile(
            r"\b(?:make|build|create|synthesi[sz]e|manufacture|design|optimi[sz]e)\b.{0,70}"
            r"\b(?:bomb|explosive|nerve agent|chemical weapon|radiological weapon|"
            r"poison gas|lethal toxin)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "weapons-or-toxic-agents",
        re.compile(
            r"\b(?:weaponize|poison|kill|injure)\b.{0,60}"
            r"\b(?:chemical|compound|toxin|explosive|population|people|person)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "cyber-abuse",
        re.compile(
            r"\b(?:write|build|create|deploy|modify)\b.{0,60}"
            r"\b(?:ransomware|credential stealer|keylogger|botnet|destructive malware|"
            r"computer virus)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "cyber-abuse",
        re.compile(
            r"\b(?:hack|breach|compromise|exfiltrate|take over)\b.{0,60}"
            r"\b(?:account|server|network|system|database|wallet|device)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "cyber-abuse",
        re.compile(
            r"\b(?:bypass|evade)\b.{0,45}"
            r"\b(?:authentication|access control|antivirus|malware detection|"
            r"security monitoring)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "self-harm",
        re.compile(
            r"\b(?:kill myself|end my life|commit suicide|suicide method|"
            r"least painful way to die)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "sexual-exploitation",
        re.compile(
            r"\b(?:generate|create|make|obtain|share)\b.{0,50}"
            r"\b(?:sexual|explicit|nude)\b.{0,30}\b(?:child|minor|underage)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "illegal-drug-production",
        re.compile(
            r"\b(?:make|cook|synthesi[sz]e|manufacture)\b.{0,50}"
            r"\b(?:methamphetamine|fentanyl|heroin|illegal narcotic)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "individual-medical-direction",
        re.compile(
            r"\b(?:what|which|how much)\b.{0,45}"
            r"\b(?:dose|dosage|medication|drug|treatment)\b.{0,55}"
            r"\b(?:i|me|my|my child|my baby)\b",
            re.IGNORECASE,
        ),
    ),
)


def classify_question_safety(question: str) -> str | None:
    """Return a bounded high-risk category without sending the prompt to a model."""
    normalized = " ".join(unicodedata.normalize("NFKC", question).split())
    for category, pattern in BLOCKED_QUESTION_PATTERNS:
        if pattern.search(normalized):
            return category
    return None


class QuestionFramer:
    def __init__(self, config: ApolloConfig | None = None) -> None:
        self.config = config or load_config()
        self.last_provider_error: str | None = None

    async def frame(self, question: str) -> QuestionCheck:
        question = question.strip()
        if len(question) < 12 or len(question) > 2_000:
            raise ValueError("Question must be between 12 and 2,000 characters")

        if classify_question_safety(question):
            return self._blocked_check(question)

        if self.config.api.get_key():
            try:
                timeout_seconds = float(os.getenv("APOLLOBOT_FRAMER_TIMEOUT", "12"))
                async with asyncio.timeout(timeout_seconds):
                    check = await self._frame_with_model(question)
                    self.last_provider_error = None
                    return (
                        self._blocked_check(question) if check.answerability == "unsafe" else check
                    )
            except Exception as error:
                self.last_provider_error = type(error).__name__
                # The gateway must remain available if a provider is degraded.
                # The response explicitly identifies the deterministic fallback.
                logger.warning(
                    "Question framing provider failed; using local fallback (%s)",
                    type(error).__name__,
                )
        if not self.config.api.get_key():
            self.last_provider_error = "unconfigured"
        return self._frame_locally(question)

    async def _frame_with_model(self, question: str) -> QuestionCheck:
        llm = create_llm(
            provider=self.config.api.default_provider,
            api_key=self.config.api.get_key(),
        )
        raw = await llm.complete_json(
            messages=[{"role": "user", "content": f"Scientific question:\n{question}"}],
            system=(
                "You are the question gateway for a computational science platform. "
                "Assess the question without pretending to search literature or run an experiment. "
                "Return JSON only with title and a domain chosen from biology, medicine, "
                "physics, climate, chemistry, computer-science, economics, or interdisciplinary. "
                "Include apollo_domain, chosen from bioinformatics, physics, cs_ml, comp_chem, "
                "or economics. Include mode, chosen from hypothesis, exploratory, "
                "meta-analysis, replication, or simulation. Include answerability, novelty, "
                "rationale, up to three "
                "falsifiable hypotheses, and five proposedSteps objects with type, label, and "
                "detail. Include an estimate with durationMinutes, computeUsd, and "
                "literatureTargets. Frame the first runnable pilot, not the entire research "
                "program: it must fit within 60 minutes, $5 of compute, and 24 literature "
                "targets. Larger follow-on work can be proposed after the pilot. Do not claim "
                "novelty is "
                "established; use adjacent-work-likely unless another triage state is justified."
                "Mark answerability unsafe and return no hypotheses or operational proposed steps "
                "for requests that meaningfully facilitate biological, chemical, weapons, cyber, "
                "sexual-exploitation, self-harm, illegal-drug, or individualized medical harm. "
                "Benign prevention, detection, epidemiology, and non-operational risk analysis "
                "should not be marked unsafe merely because they discuss a hazard."
            ),
        )
        return self._normalize_model_check(question, raw)

    def _normalize_model_check(self, question: str, raw: object) -> QuestionCheck:
        """Bound an untrusted model response to the public question-check contract."""
        fallback = self._frame_locally(question)
        normalized = fallback.model_dump(by_alias=True)
        if not isinstance(raw, dict):
            raise ValueError("Question framing model returned a non-object response")

        for field in ("title", "rationale"):
            value = raw.get(field)
            if isinstance(value, str) and value.strip():
                normalized[field] = value.strip()

        for field, allowed in (
            ("domain", QUESTION_DOMAINS),
            ("apollo_domain", APOLLO_DOMAINS),
            ("mode", RESEARCH_MODES),
            ("answerability", ANSWERABILITY_STATES),
            ("novelty", NOVELTY_STATES),
        ):
            value = raw.get(field)
            if isinstance(value, str) and value in allowed:
                normalized[field] = value

        hypotheses = raw.get("hypotheses")
        if isinstance(hypotheses, list):
            bounded_hypotheses = [
                value.strip()[:500]
                for value in hypotheses
                if isinstance(value, str) and value.strip()
            ][:3]
            if bounded_hypotheses:
                normalized["hypotheses"] = bounded_hypotheses

        proposed_steps = raw.get("proposedSteps", raw.get("proposed_steps"))
        if isinstance(proposed_steps, list):
            bounded_steps: list[dict[str, str]] = []
            for item in proposed_steps[:5]:
                if not isinstance(item, dict):
                    continue
                step = {
                    field: value.strip()[:500]
                    for field in ("type", "label", "detail")
                    if isinstance((value := item.get(field)), str) and value.strip()
                }
                if len(step) == 3:
                    bounded_steps.append(step)
            if bounded_steps:
                normalized["proposedSteps"] = bounded_steps

        estimate = raw.get("estimate")
        if isinstance(estimate, dict):
            bounded_estimate = dict(normalized["estimate"])
            for field, lower, upper in (
                ("durationMinutes", 1, 60),
                ("computeUsd", 0, 5),
                ("literatureTargets", 0, 24),
            ):
                value = estimate.get(field)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    bounded_estimate[field] = min(upper, max(lower, value))
            normalized["estimate"] = bounded_estimate

        normalized["question"] = question
        normalized["source"] = "apollobot"
        normalized.pop("id", None)
        return QuestionCheck.model_validate(normalized)

    def _frame_locally(self, question: str) -> QuestionCheck:
        lower = question.lower()
        domain, apollo_domain = detect_domain(lower)
        mode = detect_mode(lower)
        vague = len(question.split()) < 7
        answerability = "needs-refinement" if vague else "investigable"
        title = re.sub(r"[?!.]+$", "", question).strip()
        if len(title) > 110:
            title = title[:107].rstrip() + "…"

        return QuestionCheck(
            question=question,
            title=title,
            domain=domain,
            apollo_domain=apollo_domain,
            mode=mode,
            answerability=answerability,
            novelty="adjacent-work-likely",
            rationale=(
                "The question can be translated into falsifiable claims and investigated "
                "with literature, public data, and computational tests."
            ),
            hypotheses=[
                "The proposed relationship is observable in an appropriate public dataset.",
                "The result remains after the strongest obvious confounders are controlled.",
                "A robustness test can identify the boundary where the finding no longer holds.",
            ],
            proposedSteps=[
                {
                    "type": "literature",
                    "label": "Map prior work",
                    "detail": "Find the closest claims, datasets, and unresolved disagreements.",
                },
                {
                    "type": "hypothesis",
                    "label": "Define falsifiable claims",
                    "detail": "Specify outcomes, controls, exclusions, and failure conditions.",
                },
                {
                    "type": "experiment",
                    "label": "Design the experiment",
                    "detail": "Select data and analysis methods before execution.",
                },
                {
                    "type": "compute",
                    "label": "Run and stress-test",
                    "detail": "Execute in a captured environment with robustness checks.",
                },
                {
                    "type": "review",
                    "label": "Review and publish",
                    "detail": (
                        "Link every claim to evidence, limitations, and reproducible artifacts."
                    ),
                },
            ],
            estimate=RunEstimate(
                durationMinutes=45 if mode == "simulation" else 25,
                computeUsd=4.5 if mode == "simulation" else 2.4,
                literatureTargets=24,
            ),
            source="local-framer",
        )

    def _blocked_check(self, question: str) -> QuestionCheck:
        domain, apollo_domain = detect_domain(question.lower())
        title = re.sub(r"[?!.]+$", "", question).strip()
        if len(title) > 110:
            title = title[:107].rstrip() + "…"
        return QuestionCheck(
            question=question,
            title=title,
            domain=domain,
            apollo_domain=apollo_domain,
            mode=detect_mode(question.lower()),
            answerability="unsafe",
            novelty="adjacent-work-likely",
            rationale=(
                "This question needs a human safety review before Frontier Science can plan "
                "tools or experiments. Reframe it toward risk analysis, prevention, detection, "
                "or other non-operational research."
            ),
            hypotheses=[],
            proposedSteps=[
                {
                    "type": "safety",
                    "label": "Human safety review required",
                    "detail": (
                        "No tools, retrieval, or compute will run. Reframe toward prevention, "
                        "detection, or non-operational risk analysis."
                    ),
                }
            ],
            estimate=RunEstimate(
                durationMinutes=1,
                computeUsd=0,
                literatureTargets=0,
            ),
            source="local-framer",
        )


def detect_domain(question: str) -> tuple[str, str]:
    checks: list[tuple[str, str, str]] = [
        (r"gene|protein|cell|genom|microbi|plant|organism", "biology", "bioinformatics"),
        (r"patient|clinical|disease|therapy|drug|health", "medicine", "bioinformatics"),
        (r"climate|temperature|\bheat\b|carbon|weather|emission|urban tree", "climate", "physics"),
        (r"molecule|chemical|electrolyte|catalyst|compound", "chemistry", "comp_chem"),
        (r"quantum|particle|material|energy|force", "physics", "physics"),
        (r"model|algorithm|software|computer|neural|\bllm\b", "computer-science", "cs_ml"),
        (r"market|wage|inflation|economic|price|employment", "economics", "economics"),
    ]
    for pattern, field, apollo_domain in checks:
        if re.search(pattern, question):
            return field, apollo_domain
    return "interdisciplinary", "bioinformatics"


def detect_mode(question: str) -> str:
    if re.search(r"replicat|reproduc", question):
        return "replication"
    if re.search(r"simulat|model what|forecast", question):
        return "simulation"
    if re.search(r"systematic review|meta.?analysis", question):
        return "meta-analysis"
    if re.search(r"relationship|correlat|effect|whether|\bdoes\b|\bdo\b", question):
        return "hypothesis"
    return "exploratory"
