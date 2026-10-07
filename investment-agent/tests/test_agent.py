from invest_agent import agent
from invest_agent.approvals import ApprovalStore
from invest_agent.brokers.paper import PaperBroker


def _broker(cfg, prices):
    last = prices.iloc[-1]
    return PaperBroker(cfg.state_dir, starting_cash=50_000, price_source=lambda s: last[s])


def test_propose_then_approve_executes(prices, cfg):
    broker = _broker(cfg, prices)
    p = agent.run(cfg, broker=broker, prices=prices)
    assert p is not None and p.status == "pending"
    assert broker.positions() == {}  # nothing executed before approval

    done = agent.execute(cfg, p.code, broker=broker)
    assert done.status == "executed", done.result
    assert len(broker.positions()) >= 3
    # Double approval (e.g. Slack + SMS race) must not execute twice.
    assert agent.execute(cfg, p.code, broker=broker) is None


def test_reject(prices, cfg):
    broker = _broker(cfg, prices)
    p = agent.run(cfg, broker=broker, prices=prices)
    assert agent.reject(cfg, p.code)
    assert ApprovalStore(cfg.state_dir).get(p.code).status == "rejected"
    assert agent.execute(cfg, p.code, broker=broker) is None


def test_dry_run_creates_nothing(prices, cfg):
    cfg.mode = "dry_run"
    broker = _broker(cfg, prices)
    assert agent.run(cfg, broker=broker, prices=prices) is None
    assert broker.positions() == {}


def test_auto_mode_executes_and_lands_near_target_beta(prices, cfg):
    cfg.mode = "auto"
    cfg.max_turnover_pct = 1.0
    cfg.max_daily_notional_usd = 1e9
    cfg.max_order_usd = 1e9
    broker = _broker(cfg, prices)
    p = agent.run(cfg, broker=broker, prices=prices)
    assert p.status == "executed"
    _, plan, _, beta_after = agent.build_plan(cfg, broker, prices)
    assert cfg.min_beta - 0.05 <= beta_after <= cfg.max_beta
