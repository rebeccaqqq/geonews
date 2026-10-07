"""Command line entry point: `python -m invest_agent <command>`."""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd


def _cfg(args):
    from .config import load_config

    return load_config(args.config)


def cmd_run(args):
    from .agent import run

    cfg = _cfg(args)
    if args.mode:
        cfg.mode = args.mode
    p = run(cfg)
    print(p.summary if p else "No proposal created (see notifications / logs).")


def cmd_approve(args):
    from .agent import execute

    p = execute(_cfg(args), args.code.upper(), approver="cli")
    print(p.result if p else "Nothing executed.")


def cmd_reject(args):
    from .agent import reject

    print("Rejected." if reject(_cfg(args), args.code.upper()) else "Not pending.")


def cmd_serve(args):
    from .server import serve

    serve(_cfg(args), host=args.host, port=args.port)


def cmd_target(args):
    """Show the optimizer's target portfolio without touching any account."""
    from .data import load_prices
    from .optimizer import optimize

    cfg = _cfg(args)
    prices = load_prices(cfg.universe + [cfg.benchmark], period=f"{cfg.lookback_years + 1}y",
                         cache_dir=cfg.state_path / "cache")
    opt = optimize(prices, cfg)
    print(opt.describe())
    print()
    print(opt.asset_table.round(3).to_string())


def cmd_export_holdings(args):
    """Write your current brokerage holdings to CSV (input for `backtest`)."""
    from .brokers import get_broker

    cfg = _cfg(args)
    broker = get_broker(args.broker or cfg.broker, cfg.state_dir)
    pos = broker.positions()
    df = pd.DataFrame([{"symbol": p.symbol, "shares": p.quantity, "value": round(p.value, 2)} for p in pos.values()])
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} holdings to {args.out} (cash ${broker.cash():,.2f} not included)")


def cmd_backtest(args):
    from .backtest import backtest_portfolio, holdings_to_weights, load_holdings
    from .data import load_prices
    from .report import render_html

    cfg = _cfg(args) if args.with_strategy else None
    holdings = load_holdings(args.holdings)
    syms = holdings["symbol"].tolist()
    tickers = sorted(set(syms) | {args.benchmark} | (set(cfg.universe) if cfg else set()))

    if args.max:
        prices = load_prices(tickers, period="max", cache_dir="./state/cache")
        # "max" = as far back as your holdings go (dynamic join for late IPOs)
        first = min(prices[s].first_valid_index() for s in syms if s in prices and prices[s].notna().any())
        bench_first = prices[args.benchmark].first_valid_index()
        start = max(first, bench_first)
    else:
        warmup = (cfg.lookback_years if cfg else 0) + 1
        prices = load_prices(tickers, period=f"{args.years + warmup}y", cache_dir="./state/cache")
        start = prices.index[-1] - pd.DateOffset(years=args.years)

    last = prices.ffill().iloc[-1]
    weights = holdings_to_weights(holdings, last)
    result = backtest_portfolio(
        weights, prices, benchmark=args.benchmark, rebalance=args.rebalance, align=args.align,
        cost_bps=args.cost_bps, strategy_cfg=cfg, start=start,
    )
    out = render_html(result, args.out)

    pd.set_option("display.width", 160)
    print(result.title)
    cols = ["cagr", "volatility", "sharpe", "max_drawdown", "beta", "alpha_annual", "end_value"]
    print(result.stats[[c for c in cols if c in result.stats]].round(3).to_string())
    for n in result.notes:
        print("note:", n)
    print(f"\nReport: {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="invest_agent", description="Beta-banded long-term investment agent")
    ap.add_argument("--config", default=None, help="config.yaml path")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="analyze account, propose trades, notify (and auto-execute in auto mode)")
    p.add_argument("--mode", choices=["dry_run", "approve", "auto"])
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("approve", help="approve and execute a pending proposal")
    p.add_argument("code")
    p.set_defaults(fn=cmd_approve)

    p = sub.add_parser("reject", help="reject a pending proposal")
    p.add_argument("code")
    p.set_defaults(fn=cmd_reject)

    p = sub.add_parser("serve", help="listen for Slack button clicks and SMS replies")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("target", help="print the optimizer's target portfolio")
    p.set_defaults(fn=cmd_target)

    p = sub.add_parser("export-holdings", help="dump current brokerage holdings to CSV")
    p.add_argument("--out", default="holdings.csv")
    p.add_argument("--broker", choices=["robinhood", "paper"])
    p.set_defaults(fn=cmd_export_holdings)

    p = sub.add_parser("backtest", help="backtest a holdings CSV over the last N years or max history")
    p.add_argument("--holdings", default="holdings.csv")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--years", type=int, default=10)
    g.add_argument("--max", action="store_true", help="use the longest history available")
    p.add_argument("--benchmark", default="SPY")
    p.add_argument("--rebalance", choices=["none", "monthly", "quarterly", "annual"], default="none",
                   help="'none' = buy and hold today's mix; otherwise reset to today's weights periodically")
    p.add_argument("--align", choices=["dynamic", "common"], default="dynamic",
                   help="dynamic: late-IPO holdings join when they list; common: start when all have data")
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--with-strategy", action="store_true", help="also backtest the agent's strategy for comparison")
    p.add_argument("--out", default="reports/backtest.html")
    p.set_defaults(fn=cmd_backtest)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
