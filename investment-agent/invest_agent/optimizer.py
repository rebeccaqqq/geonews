"""Long-horizon, beta-banded portfolio optimizer.

Expected returns use a CAPM prior plus a small momentum tilt:

    mu_i = rf + beta_i * ERP + momentum_tilt * z(12-1 month momentum)_i

The CAPM prior keeps forecasts sane (higher beta -> proportionally higher
expected return, nothing more), and momentum is the most persistent
cross-sectional anomaly in the literature. Historical mean returns are
deliberately NOT used: they are far too noisy and cause wild overfitting.

We then solve:

    maximize   w'mu - (risk_aversion/2) w'Sigma w - beta_penalty (w'beta - target_beta)^2
    subject to sum(w) = 1,  min_beta <= w'beta <= max_beta,  min_w <= w_i <= max_w

so the portfolio sits near target_beta (~1.1) and only drifts toward
max_beta (1.5) when the extra expected return outweighs risk + penalty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import Config
from .metrics import TRADING_DAYS, betas


@dataclass
class OptimizationResult:
    weights: pd.Series
    expected_return: float
    volatility: float
    beta: float
    asset_table: pd.DataFrame  # per-asset beta, momentum z, mu, weight

    def describe(self) -> str:
        lines = [
            f"Expected return {self.expected_return:.1%}/yr | vol {self.volatility:.1%} | beta {self.beta:.2f}",
        ]
        for t, w in self.weights.sort_values(ascending=False).items():
            if w > 0:
                lines.append(f"  {t:<6} {w:6.1%}")
        return "\n".join(lines)


def shrink_covariance(returns: pd.DataFrame, delta: float = 0.3) -> pd.DataFrame:
    """Shrink sample covariance toward a constant-correlation target."""
    sample = returns.cov().values
    std = np.sqrt(np.diag(sample))
    corr = sample / np.outer(std, std)
    n = len(std)
    avg_corr = (corr.sum() - n) / (n * (n - 1)) if n > 1 else 0.0
    target = avg_corr * np.outer(std, std)
    np.fill_diagonal(target, std**2)
    shrunk = delta * target + (1 - delta) * sample
    return pd.DataFrame(shrunk * TRADING_DAYS, index=returns.columns, columns=returns.columns)


def momentum_z(prices: pd.DataFrame) -> pd.Series:
    """12-1 month momentum, cross-sectionally z-scored and clipped to +/-2."""
    if len(prices) < 252:
        return pd.Series(0.0, index=prices.columns)
    mom = prices.iloc[-21] / prices.iloc[-252] - 1
    sd = mom.std()
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=prices.columns)
    return ((mom - mom.mean()) / sd).clip(-2, 2)


def optimize(prices: pd.DataFrame, cfg: Config, as_of: pd.Timestamp | None = None) -> OptimizationResult:
    """Compute target weights using only data on or before `as_of` (no look-ahead)."""
    if as_of is not None:
        prices = prices.loc[:as_of]
    window = prices.iloc[-int(cfg.lookback_years * TRADING_DAYS) :]

    bench = cfg.benchmark
    if bench not in window.columns:
        raise ValueError(f"benchmark {bench} missing from price data")

    candidates = [t for t in cfg.universe if t in window.columns and t not in cfg.blocklist]
    # Need at least ~1 year of history to estimate beta/covariance.
    candidates = [t for t in candidates if window[t].notna().sum() >= TRADING_DAYS]
    if len(candidates) < 2:
        raise ValueError("Not enough assets with >=1y of history to optimize")

    px = window[candidates].dropna()
    rets = px.pct_change().dropna()
    mkt = window[bench].pct_change().reindex(rets.index)

    b = betas(rets, mkt).reindex(candidates).fillna(1.0)
    z = momentum_z(prices[candidates].dropna())
    mu = cfg.risk_free_rate + b * cfg.equity_risk_premium + cfg.momentum_tilt * z
    sigma = shrink_covariance(rets).loc[candidates, candidates]

    n = len(candidates)
    max_w = max(cfg.max_weight, 1.0 / n + 1e-9)
    mu_v, b_v, s_v = mu.values, b.values, sigma.values

    def objective(w: np.ndarray) -> float:
        port_beta = w @ b_v
        return -(w @ mu_v) + 0.5 * cfg.risk_aversion * (w @ s_v @ w) + cfg.beta_penalty * (port_beta - cfg.target_beta) ** 2

    def grad(w: np.ndarray) -> np.ndarray:
        port_beta = w @ b_v
        return -mu_v + cfg.risk_aversion * (s_v @ w) + 2 * cfg.beta_penalty * (port_beta - cfg.target_beta) * b_v

    constraints = [
        {"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones_like(w)},
        {"type": "ineq", "fun": lambda w: w @ b_v - cfg.min_beta, "jac": lambda w: b_v},
        {"type": "ineq", "fun": lambda w: cfg.max_beta - w @ b_v, "jac": lambda w: -b_v},
    ]
    bounds = [(cfg.min_weight, max_w)] * n
    x0 = np.full(n, 1.0 / n)
    res = minimize(objective, x0, jac=grad, bounds=bounds, constraints=constraints, method="SLSQP",
                   options={"maxiter": 500, "ftol": 1e-10})
    if not res.success:
        raise RuntimeError(f"Optimizer failed: {res.message}")

    w = pd.Series(res.x, index=candidates).clip(lower=0)
    # Drop dust positions (<0.5%) and renormalize, respecting the cap.
    w[w < 0.005] = 0.0
    w = w / w.sum()
    w = w.clip(upper=max_w)
    w = w / w.sum()

    table = pd.DataFrame({"beta": b, "momentum_z": z, "expected_return": mu, "weight": w}).sort_values(
        "weight", ascending=False
    )
    return OptimizationResult(
        weights=w,
        expected_return=float(w @ mu),
        volatility=float(np.sqrt(w @ sigma @ w)),
        beta=float(w @ b),
        asset_table=table,
    )
