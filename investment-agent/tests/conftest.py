import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from invest_agent.config import Config  # noqa: E402

BETAS = {"LOWB": 0.6, "MIDB": 0.9, "VTI": 1.0, "QQQ": 1.2, "SMH": 1.6, "HIGH": 1.9}


def make_prices(years: int = 6, seed: int = 0, late: dict | None = None) -> pd.DataFrame:
    """Synthetic one-factor market: r_i = beta_i * r_m + noise."""
    rng = np.random.default_rng(seed)
    n = years * 252
    idx = pd.bdate_range("2015-01-02", periods=n)
    mkt = rng.normal(0.0004, 0.011, n)
    data = {"SPY": 100 * np.cumprod(1 + mkt)}
    for i, (t, b) in enumerate(BETAS.items()):
        eps = rng.normal(0.0001 * (i % 3), 0.008, n)
        data[t] = 50 * np.cumprod(1 + b * mkt + eps)
    df = pd.DataFrame(data, index=idx)
    for t, start in (late or {}).items():
        df.loc[: pd.Timestamp(start) - pd.Timedelta(days=1), t] = np.nan
    return df


@pytest.fixture
def prices():
    return make_prices()


@pytest.fixture
def cfg(tmp_path):
    c = Config(
        universe=list(BETAS),
        benchmark="SPY",
        max_weight=0.35,
        state_dir=str(tmp_path / "state"),
        mode="approve",
        broker="paper",
    )
    c.ai_review.enabled = False
    c.notify.slack = False
    c.notify.sms = False
    c.validate()
    return c
