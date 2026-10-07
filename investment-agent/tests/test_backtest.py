import pandas as pd

from invest_agent.backtest import backtest_portfolio, holdings_to_weights, simulate
from invest_agent.report import render_html
from tests.conftest import make_prices


def test_single_asset_buy_and_hold_matches_price(prices):
    eq, _ = simulate(prices[["VTI"]], lambda d, a: pd.Series({"VTI": 1.0}))
    ratio = eq / prices["VTI"]
    assert (ratio - ratio.iloc[0]).abs().max() < 1e-6


def test_holdings_shares_to_weights():
    h = pd.DataFrame({"symbol": ["A", "B"], "shares": [10, 30]})
    w = holdings_to_weights(h, pd.Series({"A": 30.0, "B": 10.0}))
    assert abs(w["A"] - 0.5) < 1e-12


def test_late_listing_dynamic_and_common():
    px = make_prices(late={"HIGH": "2017-06-01"})
    w = pd.Series({"VTI": 0.5, "HIGH": 0.5})
    dyn = backtest_portfolio(w, px, rebalance="none")
    com = backtest_portfolio(w, px, rebalance="none", align="common")
    assert dyn.curves.index[0] < com.curves.index[0]
    assert com.curves.index[0] >= pd.Timestamp("2017-06-01")
    assert any("Joined late" in n for n in dyn.notes)


def test_full_report_with_strategy(prices, cfg, tmp_path):
    w = pd.Series({"VTI": 0.6, "QQQ": 0.4})
    res = backtest_portfolio(w, prices, rebalance="quarterly", strategy_cfg=cfg, start=prices.index[300])
    assert "Agent strategy" in res.curves
    sb = res.stats.loc["Agent strategy", "beta"]
    assert cfg.min_beta - 0.25 < sb < cfg.max_beta + 0.1
    out = render_html(res, tmp_path / "r" / "backtest.html")
    assert out.exists() and "Summary statistics" in out.read_text()
