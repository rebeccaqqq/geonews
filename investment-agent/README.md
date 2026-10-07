# Investment Agent

A self-hosted "AI brokerage" for one person:

1. **Agent:** aims for maximum long-term return with portfolio **beta near the market (target 1.1), never above 1.5**. It decides on trades, gets a second opinion from Claude, messages you on **Slack and/or SMS**, and places the orders on **Robinhood** once you approve.
2. **Backtester:** runs **your current portfolio** over the **last 10 years (or the full available history)**, compares it with SPY and with the agent's own strategy, and writes an HTML report.

> **Read this first**
> - **Robinhood has no official trading API.** This uses [`robin_stocks`](https://github.com/jmfernandes/robin_stocks), which calls Robinhood's private app endpoints. It can break without warning, and automated trading may violate Robinhood's Terms of Service, which could get the account restricted. If you'd rather have a supported API, Alpaca, Schwab and Interactive Brokers all have official ones. You can add one by writing a new class in `invest_agent/brokers/`.
> - This is software, not financial advice. Start with `mode: dry_run`, then move to `approve`. Only use `auto` after you've watched it for a while.
> - Backtests show the past. They don't predict the future.

---

## Quick start (local)

```bash
cd investment-agent
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp config.example.yaml config.yaml
cp .env.example .env        # add your credentials
```

### Backtest your current portfolio

```bash
# Option A: pull holdings straight from Robinhood
python -m invest_agent export-holdings --out holdings.csv

# Option B: write holdings.csv yourself (see holdings.example.csv)
#   symbol,shares        or   symbol,value      or   symbol,weight

python -m invest_agent backtest --holdings holdings.csv                 # last 10 years
python -m invest_agent backtest --holdings holdings.csv --max           # full history
python -m invest_agent backtest --holdings holdings.csv --rebalance quarterly --with-strategy
```

Open `reports/backtest.html`. It contains:

- Growth of $10k, drawdowns, **rolling 1-year beta**, and calendar-year returns, for your portfolio vs SPY (and vs the agent's strategy with `--with-strategy`)
- CAGR, volatility, Sharpe, Sortino, max drawdown, Calmar, beta, alpha, tracking error, information ratio, and up/down capture
- A per-holding table (CAGR, vol, beta, drawdown, beta contribution) and a correlation matrix
- CSV copies of everything

How the backtest works:

- **Dividend-adjusted** prices, so returns are total returns.
- `--rebalance none` (the default) buys today's mix once and holds it. `monthly`/`quarterly`/`annual` resets to today's weights on that schedule.
- **Holdings that didn't exist 10 years ago** (recent IPOs and newer ETFs) are handled two ways. `--align dynamic` (the default) spreads their weight over your other holdings until they list, then adds them. `--align common` starts the backtest on the date every holding has data. The report tells you which holdings started late.
- Backtesting **today's** holdings has hindsight bias: you own them partly because they did well. Read the comparison with that in mind.

### Look at the agent's target portfolio (no account needed)

```bash
python -m invest_agent target
```

### Run the agent

```bash
python -m invest_agent run --mode dry_run    # computes and notifies, never trades
python -m invest_agent run                   # uses the mode in config.yaml
python -m invest_agent approve ABC123        # approve from the terminal
python -m invest_agent reject ABC123
python -m invest_agent serve                 # listens for Slack clicks and SMS replies
```

---

## How the agent decides

**Expected returns:** each asset's return is estimated as the risk-free rate plus its beta times an equity risk premium, plus a small **momentum tilt** (12-month return excluding the latest month, z-scored, ±2% per z). The model doesn't extrapolate raw historical averages. Those are noisy enough to cause severe overfitting.

**Optimizer:** solves the following with SLSQP:

```
maximize   expected return  −  (λ/2)·variance  −  penalty·(β − 1.10)²
s.t.       Σw = 1,   0.90 ≤ β ≤ 1.50,   0 ≤ wᵢ ≤ 30%
```

The penalty pulls beta toward 1.1. Beta goes higher, up to the hard 1.5 ceiling, only when the extra expected return is worth it. Betas are Blume-adjusted, and the covariance matrix is shrunk toward constant correlation.

**Universe:** broad ETFs: SPY, VTI, QQQ, XLK, SMH, VUG, MTUM, QUAL, AVUV, VEA, VWO, XLV, and BRK-B. You can edit the list in `config.yaml`. Holdings outside the universe are **left alone** by default, so nothing gets sold that would trigger surprise capital gains. Set `sell_outside_universe: true` to let the agent manage them. `do_not_sell` protects specific tickers.

**Guardrails** (every trade passes all of these):

| Guardrail | Default |
|---|---|
| Rebalance only positions that drifted by more than | 3 percentage points |
| Max turnover per run | 25% of portfolio |
| Max single order | $5,000 |
| Max traded per day | $15,000 |
| Cash buffer | 1% |
| Buys funded only by cash + sells (never margin) | always |
| Skip an order if the price moved >3% since the proposal | always |
| Proposals expire after | 4 hours |
| Only one live proposal; Slack/SMS double-approval can't execute twice | always |
| Orders only during regular market hours | always |

**Claude review:** before you're notified, Claude (`claude-opus-5-5`) gets the full proposal and checks it for concentration, beta-mandate violations, turnover, and anything that looks like a data error. It returns *approve / caution / reject* and a short summary, which is included in your message. In `auto` mode, a *reject* blocks execution and asks you instead. Set `ai_review.enabled: false` to turn this off.

**Modes:**

- `dry_run`: notify only.
- `approve`: you tap **Approve** in Slack or reply `YES ABC123` by text.
- `auto`: executes when the market is open and the AI review doesn't reject, then notifies you.

---

## Setting up notifications

### Slack (recommended: approve buttons, no public URL needed)
1. Create an app at https://api.slack.com/apps. Choose **From scratch**.
2. **Socket Mode:** turn it on and create an app-level token with the `connections:write` scope. That token is `SLACK_APP_TOKEN` (`xapp-…`).
3. **Interactivity & Shortcuts:** turn it on. Socket Mode means no URL is needed.
4. **OAuth & Permissions:** add the bot scope `chat:write`, install the app to your workspace, and copy the bot token into `SLACK_BOT_TOKEN` (`xoxb-…`).
5. Invite the bot to a private channel (`/invite @yourbot`). Put the channel ID in `SLACK_CHANNEL_ID` and **your** member ID in `SLACK_APPROVER_USER_ID`. Only that user can approve trades.

### SMS via Twilio (optional)
1. Buy a Twilio number and fill in `TWILIO_*` and `ALERT_TO_NUMBER` (your phone).
2. Proposals are texted to you whenever Twilio is configured. **Replying** `YES <code>` needs Twilio to reach your server. Set up HTTPS with `deploy/oracle/Caddyfile`, set the number's *Messaging webhook* to `https://your-domain/sms`, and put the same URL in `TWILIO_WEBHOOK_URL`. Replies are checked against Twilio's signature and must come from `ALERT_TO_NUMBER`.
3. US numbers must be A2P 10DLC-registered (or use a toll-free number) before texts are delivered.

### Robinhood
Set `RH_USERNAME` and `RH_PASSWORD`. For unattended logins, turn on 2FA with an **authenticator app** in Robinhood. When it shows the QR code, choose "can't scan" and copy the text secret into `RH_MFA_SECRET`. The agent generates codes from it. The session token is cached in the service user's home directory.

---

## Deploy on Oracle Cloud

1. Create an **Always Free** Ampere A1 VM (Ubuntu 22.04/24.04). Slack-only setups need no inbound ports.
2. `git clone` this repo onto the VM, then run:
   ```bash
   bash investment-agent/deploy/oracle/setup.sh
   ```
   This installs the app into `/opt/invest-agent` as a locked-down `invest` system user and enables two systemd units:
   - `invest-agent-serve.service`: always running; listens for approvals.
   - `invest-agent-run.timer`: runs the agent **every Monday at 10:15 New York time**. Edit `OnCalendar` to change it.
3. Fill in `/opt/invest-agent/.env` and `config.yaml` (start with `mode: dry_run`), then `sudo systemctl restart invest-agent-serve`.
4. Logs: `journalctl -u invest-agent-serve -f` and `journalctl -u invest-agent-run`.
5. For SMS replies: point a domain at the VM, open ports 80/443 in the VCN security list **and** in the VM's iptables, and install Caddy with the provided `Caddyfile`.

---

## Tests

```bash
pip install -r requirements.txt
python -m pytest -q
```

The tests use synthetic market data. They don't touch the network or a broker. They cover:

- beta and metric math
- the optimizer's constraints and that it never looks ahead
- rebalancing guardrails
- backtests of holdings that start trading partway through the window
- the HTML report
- the full propose → approve → execute flow on the paper broker, including double-approval protection

## Layout

```
invest_agent/
  config.py       settings (config.yaml) + secrets (.env)
  data.py         Yahoo Finance prices, cached
  metrics.py      beta, CAGR, Sharpe, drawdown, capture ratios…
  optimizer.py    beta-banded mean-variance optimizer
  rebalance.py    target weights → guarded orders
  ai_review.py    Claude second-opinion on each proposal
  approvals.py    SQLite proposal store (atomic approve)
  notify.py       Slack (buttons) + Twilio SMS
  server.py       Slack Socket Mode + SMS webhook listener
  agent.py        run / execute / reject
  backtest.py     portfolio + walk-forward strategy backtests
  report.py       HTML report
  brokers/        robinhood.py, paper.py
deploy/oracle/    setup.sh, systemd units, Caddyfile
```
