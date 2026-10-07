import numpy as np
import pandas as pd

from invest_agent import metrics


def test_beta_of_levered_series():
    rng = np.random.default_rng(1)
    m = pd.Series(rng.normal(0, 0.01, 1000))
    a = 1.5 * m + rng.normal(0, 0.002, 1000)
    assert abs(metrics.beta(a, m) - 1.5) < 0.05


def test_cagr_and_drawdown():
    idx = pd.date_range("2020-01-01", periods=3, freq="365D")
    eq = pd.Series([100.0, 50.0, 121.0], index=idx)
    assert abs(metrics.max_drawdown(eq) + 0.5) < 1e-12
    assert abs(metrics.cagr(eq) - 0.1) < 0.01


def test_summary_against_itself_has_beta_one(prices):
    s = metrics.summary(prices["SPY"], prices["SPY"])
    assert abs(s["beta"] - 1) < 1e-9
    assert abs(s["alpha_annual"]) < 1e-9
