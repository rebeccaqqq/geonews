"""Second-opinion review of each proposal by Claude.

The math picks the trades; Claude reads the proposal like a cautious human
advisor would - sanity-checking concentration, beta, turnover and anything
odd - and returns a verdict plus a short plain-English explanation that goes
into your Slack/SMS message.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from .config import Config

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the risk reviewer for a personal, long-term investment agent.
The owner's mandate: maximize long-term (10+ year) return while keeping portfolio beta near the market (target ~1.1, hard ceiling 1.5). The agent proposes rebalancing trades from a quantitative optimizer; you review them before the owner approves.

Check for: concentration, beta outside the mandate, excessive turnover, trades that contradict the stated targets, selling positions the owner said to keep, and anything that looks like a data error (e.g. absurd prices or weights).

Respond with ONLY a JSON object, no prose around it:
{"verdict": "approve" | "caution" | "reject", "summary": "<=3 sentences in plain English for a phone notification", "concerns": ["..."]}
Use "reject" only for a clear error or mandate violation. Normal rebalancing deserves "approve"."""


@dataclass
class Review:
    verdict: str
    summary: str
    concerns: list[str]

    def render(self) -> str:
        icon = {"approve": "OK", "caution": "CAUTION", "reject": "REJECT"}.get(self.verdict, self.verdict)
        lines = [f"AI review: {icon} - {self.summary}"]
        lines += [f"  - {c}" for c in self.concerns]
        return "\n".join(lines)


def review_proposal(cfg: Config, payload: dict) -> Review | None:
    if not cfg.ai_review.enabled:
        return None
    try:
        import anthropic
    except ImportError:
        log.warning("anthropic package not installed; skipping AI review")
        return None

    client = anthropic.Anthropic()
    try:
        response = client.beta.messages.create(
            model=cfg.ai_review.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            output_config={"effort": "medium"},
            # If the primary model declines, let the API retry on its recommended fallback.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": json.dumps(payload, indent=2, default=str)}],
        )
    except anthropic.AuthenticationError:
        log.error("ANTHROPIC_API_KEY invalid; skipping AI review")
        return None
    except anthropic.RateLimitError:
        log.warning("Claude rate limited; skipping AI review")
        return None
    except anthropic.APIStatusError as e:
        log.error("Claude API error %s: %s", e.status_code, e.message)
        return None
    except anthropic.APIConnectionError:
        log.error("Could not reach Claude API; skipping AI review")
        return None

    if response.stop_reason == "refusal":
        return Review("caution", "AI reviewer declined to review this proposal.", [])

    text = "".join(b.text for b in response.content if b.type == "text").strip()
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        data = json.loads(text[start:end])
        return Review(
            verdict=str(data.get("verdict", "caution")).lower(),
            summary=str(data.get("summary", "")),
            concerns=[str(c) for c in data.get("concerns", [])],
        )
    except (ValueError, json.JSONDecodeError):
        log.warning("Unparseable AI review: %s", text[:500])
        return Review("caution", text[:300], [])
