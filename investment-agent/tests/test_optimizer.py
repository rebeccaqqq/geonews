import numpy as np

from invest_agent.optimizer import optimize


def test_weights_valid_and_beta_in_band(prices, cfg):
    res = optimize(prices, cfg)
    w = res.weights
    assert abs(w.sum() - 1) < 1e-6
    assert (w >= 0).all() and (w <= cfg.max_weight + 1e-6).all()
    assert cfg.min_beta - 1e-3 <= res.beta <= cfg.max_beta + 1e-3


def test_beta_stays_near_target_with_strong_penalty(prices, cfg):
    cfg.beta_penalty = 200
    res = optimize(prices, cfg)
    assert abs(res.beta - cfg.target_beta) < 0.05


def test_beta_never_exceeds_ceiling_even_without_penalty(prices, cfg):
    cfg.beta_penalty = 0
    cfg.risk_aversion = 0.01
    cfg.equity_risk_premium = 0.20  # make high beta very attractive
    res = optimize(prices, cfg)
    assert res.beta <= cfg.max_beta + 1e-3
    assert res.beta > cfg.target_beta  # it does use the extra room when it pays


def test_blocklist_respected(prices, cfg):
    cfg.blocklist = ["SMH"]
    assert "SMH" not in optimize(prices, cfg).weights.index


def test_no_lookahead(prices, cfg):
    as_of = prices.index[800]
    a = optimize(prices, cfg, as_of=as_of).weights
    shocked = prices.copy()
    shocked.iloc[801:] *= np.linspace(1, 5, len(shocked) - 801)[:, None]
    b = optimize(shocked, cfg, as_of=as_of).weights
    assert (a - b).abs().max() < 1e-9
