"""Configuration loading: YAML file for strategy settings, .env for secrets."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


@dataclass
class AIReviewConfig:
    enabled: bool = True
    model: str = "claude-opus-5-5"
    can_block_auto: bool = True


@dataclass
class NotifyConfig:
    slack: bool = True
    sms: bool = True


@dataclass
class Config:
    mode: str = "approve"
    broker: str = "robinhood"
    benchmark: str = "SPY"

    target_beta: float = 1.10
    min_beta: float = 0.90
    max_beta: float = 1.50
    beta_penalty: float = 4.0
    risk_aversion: float = 2.0

    universe: list[str] = field(default_factory=lambda: ["SPY", "QQQ", "VEA", "AVUV"])
    min_weight: float = 0.0
    max_weight: float = 0.30
    min_positions: int = 4

    risk_free_rate: float = 0.04
    equity_risk_premium: float = 0.05
    momentum_tilt: float = 0.02
    lookback_years: int = 3

    drift_threshold: float = 0.03
    min_trade_usd: float = 25.0
    cash_buffer_pct: float = 0.01
    max_turnover_pct: float = 0.25
    max_order_usd: float = 5000.0
    max_daily_notional_usd: float = 15000.0
    proposal_ttl_minutes: int = 240
    transaction_cost_bps: float = 5.0

    sell_outside_universe: bool = False
    do_not_sell: list[str] = field(default_factory=list)
    blocklist: list[str] = field(default_factory=list)

    ai_review: AIReviewConfig = field(default_factory=AIReviewConfig)
    notify: NotifyConfig = field(default_factory=NotifyConfig)
    state_dir: str = "./state"

    def validate(self) -> None:
        if self.mode not in {"dry_run", "approve", "auto"}:
            raise ValueError(f"mode must be dry_run|approve|auto, got {self.mode!r}")
        if not (0 < self.min_beta <= self.target_beta <= self.max_beta):
            raise ValueError("need 0 < min_beta <= target_beta <= max_beta")
        if self.max_beta > 1.5:
            raise ValueError("max_beta above 1.5 is outside the agent's mandate")
        if self.max_weight * len(self.universe) < 1:
            raise ValueError("max_weight too small for the universe to sum to 100%")

    @property
    def state_path(self) -> Path:
        p = Path(self.state_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


def load_config(path: str | os.PathLike | None = None) -> Config:
    if load_dotenv:
        load_dotenv()
    path = Path(path or os.environ.get("INVEST_AGENT_CONFIG", "config.yaml"))
    raw: dict = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text()) or {}

    known = {f.name for f in fields(Config)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")

    raw["ai_review"] = AIReviewConfig(**(raw.get("ai_review") or {}))
    raw["notify"] = NotifyConfig(**(raw.get("notify") or {}))
    cfg = Config(**raw)
    cfg.universe = [t.upper() for t in cfg.universe]
    cfg.do_not_sell = [t.upper() for t in cfg.do_not_sell]
    cfg.blocklist = [t.upper() for t in cfg.blocklist]
    cfg.validate()
    return cfg
