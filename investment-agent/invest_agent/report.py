"""Render a BacktestResult to a self-contained HTML report (+ CSVs)."""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mtick  # noqa: E402

from .backtest import BacktestResult  # noqa: E402
from .metrics import drawdown_series  # noqa: E402

# Validated categorical slots 1-3 (blue, orange, aqua): distinguishable for
# colour-blind readers when all three appear together.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

PCT_ROWS = {"total_return", "cagr", "volatility", "max_drawdown", "best_day", "worst_day", "pct_positive_days",
            "alpha_annual", "tracking_error", "benchmark_cagr"}
RATIO_ROWS = {"sharpe", "sortino", "calmar", "beta", "correlation", "information_ratio", "upside_capture",
              "downside_capture"}
LABELS = {
    "end_value": "Ending value of $10k", "total_return": "Total return", "cagr": "CAGR (annual)",
    "volatility": "Volatility (annual)", "sharpe": "Sharpe", "sortino": "Sortino", "max_drawdown": "Max drawdown",
    "calmar": "Calmar", "beta": "Beta", "correlation": "Correlation to benchmark", "alpha_annual": "Alpha (annual)",
    "tracking_error": "Tracking error", "information_ratio": "Information ratio",
    "upside_capture": "Upside capture", "downside_capture": "Downside capture",
    "best_day": "Best day", "worst_day": "Worst day",
}


def _style(ax, title: str):
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=10)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=144, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _color(name: str) -> str:
    """Colour follows the series, so it stays the same on every chart."""
    if name.startswith("Benchmark"):
        return SERIES_COLORS[1]
    if name.startswith("Agent"):
        return SERIES_COLORS[2]
    return SERIES_COLORS[0]


def _legend(ax):
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower left", bbox_to_anchor=(0, 1.0),
              ncol=3, borderaxespad=0.2)


def _line_chart(df, title, yfmt=None, log=False, refs=(), end_labels=False):
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=SURFACE)
    _style(ax, title)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=26)
    for col in df.columns:
        s = df[col].dropna()
        ax.plot(s.index, s.values, color=_color(col), linewidth=2, label=col)
        if end_labels:
            ax.annotate(f" ${s.values[-1]:,.0f}", (s.index[-1], s.values[-1]), color=INK_2, fontsize=8, va="center")
    for y, label in refs:
        ax.axhline(y, color=INK_2, linewidth=1, linestyle="--")
        ax.annotate(label, (1.0, y), xycoords=("axes fraction", "data"), color=INK_2, fontsize=8,
                    va="center", ha="left")
    if log:
        ax.set_yscale("log")
        money = mtick.FuncFormatter(lambda v, _: f"${v:,.0f}")
        ax.yaxis.set_major_formatter(money)
        ax.yaxis.set_minor_formatter(money)
    elif yfmt:
        ax.yaxis.set_major_formatter(yfmt)
    _legend(ax)
    return _png(fig)


def _yearly_chart(df):
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=SURFACE)
    _style(ax, "Calendar-year returns")
    ax.set_title("Calendar-year returns", loc="left", color=INK, fontsize=12, fontweight="bold", pad=26)
    n = len(df.columns)
    width = 0.8 / n
    x = range(len(df.index))
    for i, col in enumerate(df.columns):
        ax.bar([xi + (i - (n - 1) / 2) * width for xi in x], df[col].values, width=width * 0.92,
               color=_color(col), label=col)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_xticks(list(x), [str(y) for y in df.index])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    _legend(ax)
    return _png(fig)


def _fmt(row: str, v) -> str:
    if v is None or v != v:  # NaN
        return "-"
    if row in PCT_ROWS:
        return f"{v:.1%}"
    if row in RATIO_ROWS:
        return f"{v:.2f}"
    if isinstance(v, float):
        return f"${v:,.0f}"
    return html.escape(str(v))


def _table(df, fmt_by_col=None, fmt_by_row=None) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = []
    for idx, row in df.iterrows():
        cells = []
        for c, v in row.items():
            if fmt_by_row:
                cells.append(_fmt(fmt_by_row(idx), v))
            elif fmt_by_col and c in fmt_by_col:
                cells.append(fmt_by_col[c](v))
            else:
                cells.append(html.escape(str(v)))
        label = LABELS.get(idx, idx) if fmt_by_row else idx
        body.append(f"<tr><th>{html.escape(str(label))}</th>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return f"<table><thead><tr><th></th>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def render_html(result: BacktestResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    curves = result.curves
    dd = curves.apply(lambda s: drawdown_series(s.dropna()))

    charts = [
        _line_chart(curves, "Growth of $10,000 (log scale)", log=True, end_labels=True),
        _line_chart(dd, "Drawdown from peak", yfmt=mtick.PercentFormatter(1.0)),
    ]
    if not result.rolling_beta.empty:
        charts.append(_line_chart(result.rolling_beta, "Rolling 1-year beta vs benchmark",
                                  refs=[(1.0, " market 1.0"), (1.5, " ceiling 1.5")]))
    charts.append(_yearly_chart(result.yearly))

    rows = [r for r in LABELS if r in result.stats.columns]
    stats_t = result.stats[rows].T
    pct = lambda v: "-" if v != v else f"{v:.1%}"  # noqa: E731
    two = lambda v: "-" if v != v else f"{v:.2f}"  # noqa: E731
    assets_t = result.asset_stats.copy()
    yearly_t = result.yearly.copy()
    yearly_t.index = yearly_t.index.astype(str)

    p_stats = result.stats.iloc[0]
    headline = (
        f"Your portfolio returned <b>{p_stats['cagr']:.1%}/yr</b> vs "
        f"<b>{p_stats['benchmark_cagr']:.1%}/yr</b> for the benchmark, with beta <b>{p_stats['beta']:.2f}</b> "
        f"and a worst drawdown of <b>{p_stats['max_drawdown']:.1%}</b>."
    )
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in result.notes)
    imgs = "".join(f'<img alt="chart" src="data:image/png;base64,{c}">' for c in charts)

    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Portfolio backtest</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:{SURFACE};color:{INK};max-width:1040px;margin:32px auto;padding:0 16px}}
h1{{margin-bottom:4px}} .sub{{color:{INK_2};margin-top:0}}
img{{width:100%;margin:12px 0}}
table{{border-collapse:collapse;margin:12px 0 28px;font-size:14px;font-variant-numeric:tabular-nums}}
th,td{{padding:6px 12px;border-bottom:1px solid {GRID};text-align:right}} th:first-child{{text-align:left}}
thead th{{color:{INK_2};font-weight:600}}
.note{{color:{INK_2};font-size:13px}}
</style></head><body>
<h1>Portfolio backtest</h1><p class="sub">{html.escape(result.title)}</p>
<p>{headline}</p>
{imgs}
<h2>Summary statistics</h2>{_table(stats_t, fmt_by_row=lambda r: r)}
<h2>Holdings</h2>{_table(assets_t, fmt_by_col={"weight": pct, "cagr": pct, "volatility": pct, "beta": two,
                                              "max_drawdown": pct, "beta_contribution": two})}
<h2>Calendar-year returns</h2>{_table(yearly_t, fmt_by_col={c: pct for c in yearly_t.columns})}
<h2>Correlation between holdings</h2>{_table(result.correlation, fmt_by_col={c: two for c in result.correlation.columns})}
<h2>Notes &amp; caveats</h2><ul class="note">{notes}
<li>Prices are dividend-adjusted (total return). Taxes are not modeled. Past performance does not predict future results.</li></ul>
</body></html>"""
    path.write_text(doc)

    out_dir = path.parent
    curves.to_csv(out_dir / "equity_curves.csv")
    result.stats.to_csv(out_dir / "stats.csv")
    result.yearly.to_csv(out_dir / "yearly_returns.csv")
    result.asset_stats.to_csv(out_dir / "holdings_stats.csv")
    return path
