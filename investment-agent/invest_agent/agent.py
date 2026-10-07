"""The agent loop: analyze -> propose -> (approve) -> execute -> report."""

from __future__ import annotations

import logging
import time

import pandas as pd

from .ai_review import review_proposal
from .approvals import ApprovalStore, Proposal
from .brokers import Broker, get_broker
from .config import Config
from .data import load_prices
from .metrics import TRADING_DAYS, betas
from .notify import notify
from .optimizer import optimize
from .rebalance import RebalancePlan, plan_rebalance

log = logging.getLogger(__name__)

MAX_PRICE_MOVE = 0.03  # skip an order if price moved >3% since the proposal


def portfolio_beta(weights: pd.Series, asset_betas: pd.Series) -> float:
    w = weights.reindex(asset_betas.index).fillna(0)
    return float((w * asset_betas).sum())


def build_plan(cfg: Config, broker: Broker, prices: pd.DataFrame | None = None):
    positions = broker.positions()
    cash = broker.cash()
    held = list(positions)
    tickers = sorted(set(cfg.universe) | set(held) | {cfg.benchmark})
    if prices is None:
        prices = load_prices(tickers, period=f"{cfg.lookback_years + 1}y", cache_dir=cfg.state_path / "cache")

    opt = optimize(prices, cfg)
    quotes = broker.quotes(sorted(set(opt.weights.index) | set(held)))
    plan = plan_rebalance(positions, cash, opt.weights, cfg, quotes)

    window = prices.iloc[-cfg.lookback_years * TRADING_DAYS :]
    rets = window.pct_change()
    all_betas = betas(rets.drop(columns=[cfg.benchmark], errors="ignore"), rets[cfg.benchmark])
    all_betas[cfg.benchmark] = 1.0
    all_betas = all_betas.fillna(1.0)  # unknown history -> assume market beta
    beta_now = portfolio_beta(plan.current_weights, all_betas)
    beta_after = portfolio_beta(plan.target_weights, all_betas)
    return opt, plan, beta_now, beta_after


def format_proposal(cfg: Config, opt, plan: RebalancePlan, beta_now: float, beta_after: float) -> str:
    lines = [
        f"*Portfolio:* ${plan.total_value:,.0f} (cash ${plan.cash:,.0f})",
        f"*Beta:* {beta_now:.2f} now -> {beta_after:.2f} after (target {cfg.target_beta:.2f}, max {cfg.max_beta:.2f})",
        f"*Model:* exp. return {opt.expected_return:.1%}/yr, vol {opt.volatility:.1%}",
        "",
        f"*Proposed orders ({len(plan.trades)}, ${plan.gross_notional:,.0f} gross):*",
    ]
    lines += [f"`{t}`" for t in plan.trades]
    if plan.notes:
        lines += ["", "_Notes:_"] + [f"- {n}" for n in plan.notes]
    return "\n".join(lines)


def run(cfg: Config, broker: Broker | None = None, prices: pd.DataFrame | None = None) -> Proposal | None:
    broker = broker or get_broker(cfg.broker, cfg.state_dir)
    store = ApprovalStore(cfg.state_dir)
    opt, plan, beta_now, beta_after = build_plan(cfg, broker, prices)
    log.info("Target:\n%s", opt.describe())

    if not plan.trades:
        msg = f"Investment agent: portfolio within drift bands, no trades. Beta {beta_now:.2f}, value ${plan.total_value:,.0f}."
        notify(cfg, msg)
        return None

    already = store.executed_notional_today()
    if already + plan.gross_notional > cfg.max_daily_notional_usd:
        notify(cfg, f"Investment agent: proposal of ${plan.gross_notional:,.0f} would exceed the "
                    f"${cfg.max_daily_notional_usd:,.0f} daily cap (${already:,.0f} already traded). Skipping.")
        return None

    text = format_proposal(cfg, opt, plan, beta_now, beta_after)
    review = review_proposal(
        cfg,
        {
            "mandate": {"target_beta": cfg.target_beta, "max_beta": cfg.max_beta, "max_weight": cfg.max_weight},
            "portfolio_value": plan.total_value,
            "cash": plan.cash,
            "beta_now": beta_now,
            "beta_after": beta_after,
            "current_weights": plan.current_weights.round(4).to_dict(),
            "target_weights": plan.target_weights.round(4).to_dict(),
            "optimizer_inputs": opt.asset_table.round(4).to_dict(orient="index"),
            "trades": [t.to_dict() for t in plan.trades],
            "notes": plan.notes,
            "do_not_sell": cfg.do_not_sell,
        },
    )
    if review:
        text += "\n\n" + review.render()

    if cfg.mode == "dry_run":
        notify(cfg, "*[DRY RUN]* Investment agent proposal\n" + text)
        return None

    proposal = store.create(plan.trades, text, cfg.proposal_ttl_minutes)

    if cfg.mode == "auto":
        blocked = review and review.verdict == "reject" and cfg.ai_review.can_block_auto
        if not blocked and broker.market_open():
            notify(cfg, "*Auto-executing* investment agent proposal\n" + text)
            return execute(cfg, proposal.code, broker, approver="auto")
        reason = "AI reviewer rejected it" if blocked else "market is closed"
        text = f"_Auto-execution paused ({reason}); approval required._\n" + text

    sms = (
        f"Invest agent: {len(plan.trades)} orders, ${plan.gross_notional:,.0f}. "
        f"Beta {beta_now:.2f}->{beta_after:.2f}.\n"
        + "\n".join(f"{t.side.upper()} {t.symbol} ${t.notional_usd:,.0f}" for t in plan.trades)
        + (f"\n{review.summary}" if review else "")
        + f"\nReply YES {proposal.code} to execute or NO {proposal.code}. Expires in {cfg.proposal_ttl_minutes}m."
    )
    notify(cfg, f"*Investment agent proposal* `{proposal.code}`\n{text}", sms_text=sms, proposal_code=proposal.code)
    return proposal


def reject(cfg: Config, code: str, approver: str = "cli") -> bool:
    store = ApprovalStore(cfg.state_dir)
    p = store.claim(code, "rejected")
    if p:
        store.finish(code, "rejected", f"rejected by {approver}")
        notify(cfg, f"Proposal {code} rejected by {approver}. No orders placed.")
    return p is not None


def execute(cfg: Config, code: str, broker: Broker | None = None, approver: str = "cli") -> Proposal | None:
    broker = broker or get_broker(cfg.broker, cfg.state_dir)
    store = ApprovalStore(cfg.state_dir)
    existing = store.get(code)
    if not existing:
        notify(cfg, f"Proposal {code} not found.")
        return None
    if existing.status != "pending" or existing.expired:
        notify(cfg, f"Proposal {code} is {existing.status if not existing.expired else 'expired'}; nothing done.")
        return None
    if not broker.market_open():
        notify(cfg, f"Market is closed; proposal {code} stays pending until it expires. Approve again during market hours.")
        return None

    p = store.claim(code, "executing")
    if not p:
        return None  # someone else (Slack vs SMS) already acted on it

    results = []
    quotes = broker.quotes([t.symbol for t in p.trades])
    ordered = [t for t in p.trades if t.side == "sell"] + [t for t in p.trades if t.side == "buy"]
    for t in ordered:
        now_px = quotes.get(t.symbol)
        if now_px and t.price and abs(now_px / t.price - 1) > MAX_PRICE_MOVE:
            results.append(f"SKIP {t.symbol}: price moved {now_px / t.price - 1:+.1%} since proposal")
            continue
        if t.side == "buy":
            # The cash buffer was already reserved when the plan was sized.
            amount = min(t.notional_usd, max(broker.cash(), 0))
            if amount < cfg.min_trade_usd:
                results.append(f"SKIP BUY {t.symbol}: insufficient settled cash")
                continue
            r = broker.buy_dollars(t.symbol, amount)
        else:
            r = broker.sell_dollars(t.symbol, t.notional_usd)
        results.append(f"{'OK ' if r.ok else 'ERR'} {r.side.upper()} {r.symbol} ${r.notional_usd:,.2f} {r.message}")
        if t.side == "sell":
            time.sleep(1)  # give the sell a moment to free up buying power

    ok = all(not line.startswith("ERR") for line in results)
    summary = "\n".join(results)
    store.finish(code, "executed" if ok else "failed", summary)
    notify(cfg, f"Proposal {code} executed (approved by {approver}):\n```{summary}```",
           sms_text=f"Invest agent {code} done:\n{summary}")
    return store.get(code)
