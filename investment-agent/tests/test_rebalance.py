import pandas as pd

from invest_agent.rebalance import Position, plan_rebalance


def _px(*syms):
    return {s: 100.0 for s in syms}


def test_buys_from_cash(cfg):
    cfg.max_turnover_pct = 1.0
    cfg.max_daily_notional_usd = 1e9
    cfg.max_order_usd = 1e9
    target = pd.Series({"VTI": 0.5, "QQQ": 0.5})
    plan = plan_rebalance({}, 10_000, target, cfg, _px("VTI", "QQQ"))
    assert {t.side for t in plan.trades} == {"buy"}
    assert abs(plan.gross_notional - 9_900) < 1  # 1% cash buffer


def test_limits_and_do_not_sell(cfg):
    cfg.do_not_sell = ["AAPL"]
    cfg.sell_outside_universe = True
    cfg.universe.append("AAPL")
    positions = {"AAPL": Position("AAPL", 100, 100.0), "VTI": Position("VTI", 100, 100.0)}
    target = pd.Series({"QQQ": 1.0})
    plan = plan_rebalance(positions, 0, target, cfg, _px("AAPL", "VTI", "QQQ"))
    assert not any(t.symbol == "AAPL" and t.side == "sell" for t in plan.trades)
    for t in plan.trades:
        assert t.notional_usd <= cfg.max_order_usd
    assert sum(t.notional_usd for t in plan.trades if t.side == "buy") <= sum(
        t.notional_usd for t in plan.trades if t.side == "sell"
    ) + 1e-6
    assert any("do_not_sell" in n for n in plan.notes)


def test_outside_universe_left_alone(cfg):
    positions = {"TSLA": Position("TSLA", 10, 100.0)}
    plan = plan_rebalance(positions, 5_000, pd.Series({"VTI": 1.0}), cfg, _px("TSLA", "VTI"))
    assert "TSLA" in plan.locked
    assert all(t.symbol != "TSLA" for t in plan.trades)


def test_small_drift_no_trades(cfg):
    positions = {"VTI": Position("VTI", 50, 100.0), "QQQ": Position("QQQ", 49, 100.0)}
    plan = plan_rebalance(positions, 100, pd.Series({"VTI": 0.5, "QQQ": 0.5}), cfg, _px("VTI", "QQQ"))
    assert plan.trades == []
