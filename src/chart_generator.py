import os
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PREDICTIONS_PATH = "predictions.csv"
OUTPUT_PATH = "oraculo-sp500/index.html"
MIN_CANDLES = 8       # minimum weeks of candlesticks to show
MAX_CANDLES = 24      # cap to keep the chart readable

# ── Injected CSS ────────────────────────────────────────────────────────────

_CSS = """
:root {
  --bg:         #0f172a;
  --surface:    #1e293b;
  --surface-2:  #263548;
  --border:     #334155;
  --accent:     #3b82f6;
  --text:       #f1f5f9;
  --text-muted: #94a3b8;
  --tag-bg:     #172554;
  --tag-text:   #93c5fd;
  --radius:     12px;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: var(--bg);
  color: var(--text);
  margin: 0;
  line-height: 1.6;
}

/* Navigation bar */
.o-nav {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 28px;
  background: #111827;
  border-bottom: 1px solid var(--border);
  position: sticky;
  top: 0;
  z-index: 100;
}
.o-nav__back {
  color: var(--accent);
  text-decoration: none;
  font-size: 0.875rem;
  font-weight: 600;
  transition: opacity 0.15s;
}
.o-nav__back:hover { opacity: 0.75; }
.o-nav__title {
  font-size: 0.875rem;
  color: var(--text-muted);
  font-weight: 500;
}

/* Chart container */
.o-chart-wrapper {
  max-width: 1200px;
  margin: 0 auto;
  padding: 32px 24px 8px;
}

/* Explanatory section */
.o-info {
  background: var(--bg);
  padding: 48px 24px 72px;
  border-top: 1px solid var(--border);
  margin-top: 16px;
}
.o-info__container { max-width: 1100px; margin: 0 auto; }

.o-section-label {
  font-size: 0.75rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--text-muted);
  margin-bottom: 28px;
}

.o-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}

/* Cards */
.o-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 28px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.o-card--wide { grid-column: 1 / -1; }
.o-card--accent { border-top: 3px solid var(--accent); }

.o-card h2 {
  font-size: 1rem;
  font-weight: 700;
  color: var(--text);
  margin: 0;
}
.o-card p {
  font-size: 0.9rem;
  color: var(--text-muted);
  margin: 0;
}
.o-card ul {
  padding-left: 18px;
  display: flex;
  flex-direction: column;
  gap: 5px;
  margin: 0;
}
.o-card li {
  font-size: 0.9rem;
  color: var(--text-muted);
}
.o-card li strong { color: var(--text); }

/* Feature table */
.o-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.o-table th {
  text-align: left;
  padding: 9px 12px;
  color: var(--text-muted);
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  border-bottom: 1px solid var(--border);
}
.o-table td {
  padding: 9px 12px;
  color: var(--text-muted);
  border-bottom: 1px solid rgba(51,65,85,0.45);
  vertical-align: top;
  line-height: 1.55;
}
.o-table td:first-child {
  color: var(--tag-text);
  font-weight: 600;
  white-space: nowrap;
}
.o-table tr:last-child td { border-bottom: none; }

/* Model comparison chips */
.o-model-row {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}
.o-model-chip {
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 16px;
  flex: 1;
  min-width: 160px;
}
.o-model-chip__name {
  font-size: 0.875rem;
  font-weight: 700;
  color: var(--text);
  margin-bottom: 4px;
}
.o-model-chip__desc {
  font-size: 0.8rem;
  color: var(--text-muted);
  line-height: 1.5;
}

/* Disclaimer footer */
.o-footer {
  text-align: center;
  padding: 20px 24px;
  color: var(--text-muted);
  font-size: 0.8125rem;
  border-top: 1px solid var(--border);
}
.o-footer a { color: var(--accent); text-decoration: none; }
.o-footer a:hover { text-decoration: underline; }

/* Responsive */
@media (max-width: 640px) {
  .o-grid { grid-template-columns: 1fr; }
  .o-card--wide { grid-column: auto; }
  .o-nav__title { display: none; }
  .o-table { font-size: 0.78rem; }
  .o-chart-wrapper { padding: 16px 12px 0; }
}
"""

# ── Static HTML blocks ───────────────────────────────────────────────────────

_HEAD_EXTRA = (
    '<meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width, initial-scale=1">'
    '<title>The Oracle &#8212; S&amp;P 500 Weekly Forecast</title>'
    "<style>" + _CSS + "</style>"
)

_NAV_HTML = """\
<nav class="o-nav">
  <a href="../index.html" class="o-nav__back">&#8592; Portfolio</a>
  <span class="o-nav__title">The Oracle &middot; S&amp;P 500 Weekly Forecast</span>
</nav>
<div class="o-chart-wrapper">"""

_FEATURES_TABLE = """\
<table class="o-table">
  <thead>
    <tr>
      <th>Feature</th>
      <th>Source</th>
      <th>Why it matters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>RSI (14)</td>
      <td>S&amp;P 500 weekly</td>
      <td>Momentum oscillator — signals overbought (&gt;70) or oversold (&lt;30) conditions.</td>
    </tr>
    <tr>
      <td>EMA (12)</td>
      <td>S&amp;P 500 weekly</td>
      <td>Exponential MA weighted toward recent prices — reacts faster than SMA to trend shifts.</td>
    </tr>
    <tr>
      <td>Price / SMA20</td>
      <td>Derived</td>
      <td>How far price has deviated from its 20-week average — mean-reversion signal.</td>
    </tr>
    <tr>
      <td>Trend Spread</td>
      <td>Derived</td>
      <td>(SMA20 &minus; SMA50) / SMA50 &times; 100. Positive = short trend above long trend (bullish); negative = compression or downtrend. Price-level independent.</td>
    </tr>
    <tr>
      <td>VIX Level</td>
      <td>CBOE daily (Fri close)</td>
      <td>Market fear gauge — elevated VIX precedes volatile weeks and risk-off selling.</td>
    </tr>
    <tr>
      <td>VIX Change</td>
      <td>CBOE daily (derived)</td>
      <td>Week-over-week % change in VIX. A spike matters more than the absolute level — captures fear acceleration.</td>
    </tr>
    <tr>
      <td>Lag Return &minus;1W</td>
      <td>S&amp;P 500 weekly</td>
      <td>Previous week's return — short-term momentum signal.</td>
    </tr>
    <tr>
      <td>Lag Return &minus;2W</td>
      <td>S&amp;P 500 weekly</td>
      <td>Two-week-ago return — momentum continuation signal.</td>
    </tr>
    <tr>
      <td>Lag Return &minus;3W</td>
      <td>S&amp;P 500 weekly</td>
      <td>Three-week-ago return — baseline for momentum decay pattern.</td>
    </tr>
    <tr>
      <td>Realized Vol</td>
      <td>S&amp;P 500 daily</td>
      <td>Std dev of 5 daily returns within the week — current volatility regime proxy.</td>
    </tr>
  </tbody>
</table>"""


def _build_info_section(
    predicted_return_pct: float,
    predicted_price: float,
    price_range_low: float,
    price_range_high: float,
    now_str: str,
) -> str:
    direction = "upward" if predicted_return_pct >= 0 else "downward"
    sign = "+" if predicted_return_pct >= 0 else ""

    return f"""\
</div><!-- close o-chart-wrapper -->

<section class="o-info">
  <div class="o-info__container">

    <p class="o-section-label">Methodology &amp; Features</p>

    <div class="o-grid">

      <!-- How it works -->
      <div class="o-card o-card--wide o-card--accent">
        <h2>How It Works</h2>
        <p>
          Every Monday, the pipeline downloads 5 years of S&amp;P 500 (OHLCV) and VIX data,
          engineers 10 technical features for the most recent week, and retrains both models
          on a rolling 2-year window (104 weeks). The model with the lower walk-forward RMSE
          wins and generates the weekly forecast.
          This week&#8217;s forecast: <strong style="color:#3b82f6">${predicted_price:,.0f}</strong>
          &mdash; range <strong style="color:#94a3b8">${price_range_low:,.0f}&ndash;${price_range_high:,.0f}</strong>
          ({sign}{predicted_return_pct:.2f}%) &mdash; the model expects a <strong>{direction}</strong> week for the S&amp;P 500.
        </p>
      </div>

      <!-- Features table -->
      <div class="o-card o-card--wide">
        <h2>Input Features (10)</h2>
        <p style="margin-bottom:16px">
          All features are computed using data available at or before Friday&#8217;s close of week W.
          The target (next week&#8217;s return) uses the Friday W+1 close.
          No future data ever leaks into the features.
        </p>
        {_FEATURES_TABLE}
      </div>

      <!-- Why these models -->
      <div class="o-card">
        <h2>Why XGBoost &amp; Random Forest?</h2>
        <p>
          Financial return data is noisy and non-linear — linear regression consistently
          underperforms. Both tree-based ensembles handle this well without manual
          feature interaction engineering.
        </p>
        <div class="o-model-row">
          <div class="o-model-chip">
            <div class="o-model-chip__name">XGBoost</div>
            <div class="o-model-chip__desc">
              Gradient boosting: each tree corrects the residuals of the previous one.
              Aggressive learner — excels when the signal is consistent.
            </div>
          </div>
          <div class="o-model-chip">
            <div class="o-model-chip__name">Random Forest</div>
            <div class="o-model-chip__desc">
              Bagging ensemble: many uncorrelated trees vote together.
              Conservative learner — more robust when data is noisy or regime-shifting.
            </div>
          </div>
        </div>
        <p>
          By running both every week and picking the winner by walk-forward RMSE, the
          pipeline adapts to whichever model fits the current market regime better.
        </p>
      </div>

      <!-- Validation & no-leakage -->
      <div class="o-card">
        <h2>Walk-Forward Validation &amp; No Data Leakage</h2>
        <ul>
          <li>
            <strong>TimeSeriesSplit (5 folds):</strong> in every fold, training data
            always precedes validation data in time. Standard k-fold would mix future
            data into training — a critical flaw in time-series forecasting.
          </li>
          <li>
            <strong>RobustScaler over StandardScaler:</strong> financial returns have
            extreme outliers (COVID crash, flash crashes). RobustScaler uses median +
            IQR instead of mean + std, making it resistant to those spikes.
          </li>
          <li>
            <strong>Rolling 2-year window:</strong> the market of 2020 behaves
            structurally differently from 2024. Training on all available history would
            dilute current-regime patterns; 104 weeks keeps the model anchored to
            recent market dynamics.
          </li>
          <li>
            <strong>Scaler fit on train only:</strong> the RobustScaler is fitted
            exclusively on training rows and then applied to the prediction row —
            never the reverse.
          </li>
        </ul>
      </div>

    </div><!-- close o-grid -->
  </div><!-- close o-info__container -->
</section>

<footer class="o-footer">
  <p>
    Educational project &mdash; Not financial advice. &nbsp;&middot;&nbsp;
    Last updated: {now_str} &nbsp;&middot;&nbsp;
    <a href="../index.html">&#8592; Back to Portfolio</a>
  </p>
</footer>"""


# ── Main function ────────────────────────────────────────────────────────────

def generate_chart(gspc_weekly: pd.DataFrame, predicted_return_pct: float, winning_model: str):
    """
    Generates oraculo-sp500/index.html: Plotly candlestick chart + predictions,
    wrapped with navigation, styling, and an explanatory methodology section.
    """
    predictions_df = pd.read_csv(PREDICTIONS_PATH, parse_dates=["week_start", "week_end"])

    # Show all predictions; cap candles for readability
    n_candles = min(max(MIN_CANDLES, len(predictions_df) + 2), MAX_CANDLES)
    candles_df = gspc_weekly.tail(n_candles).copy()

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],
        shared_xaxes=False,
        vertical_spacing=0.08,
        subplot_titles=("S&P 500 Price + Predictions", "Model Metrics"),
        specs=[[{"type": "candlestick"}], [{"type": "table"}]],
    )

    # Candlestick traces
    fig.add_trace(
        go.Candlestick(
            x=candles_df.index,
            open=candles_df["Open"],
            high=candles_df["High"],
            low=candles_df["Low"],
            close=candles_df["Close"],
            name="S&P 500",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1, col=1,
    )

    # Build forecast and actual scatter data from all prediction rows
    fore_x, fore_y, fore_text, fore_colors = [], [], [], []
    act_x,  act_y,  act_text               = [], [], []

    for _, row in predictions_df.iterrows():
        base       = float(row["sp500_close_prev_friday"])
        pred_ret   = float(row["predicted_return_pct"])
        pred_price = (
            float(row["predicted_price"])
            if "predicted_price" in predictions_df.columns and pd.notna(row.get("predicted_price"))
            else base * (1 + pred_ret / 100)
        )

        has_actual = (
            pd.notna(row["actual_return_pct"])
            and str(row["actual_return_pct"]).strip() != ""
        )

        fore_x.append(row["week_end"])
        fore_y.append(pred_price)
        fore_colors.append("#3b82f6" if has_actual else "#f59e0b")
        fore_text.append(
            f"<b>Forecast</b>: ${pred_price:,.0f} ({pred_ret:+.2f}%)"
            f"<br>Model: {row['winning_model']}"
            f"<br>Week: {row['week_end'].strftime('%b %d, %Y')}"
        )

        if has_actual:
            actual_ret = float(row["actual_return_pct"])
            act_price  = (
                float(row["actual_price"])
                if "actual_price" in predictions_df.columns and pd.notna(row.get("actual_price"))
                   and str(row.get("actual_price", "")).strip() != ""
                else base * (1 + actual_ret / 100)
            )
            error = actual_ret - pred_ret
            act_x.append(row["week_end"])
            act_y.append(act_price)
            act_text.append(
                f"<b>Actual</b>: ${act_price:,.0f} ({actual_ret:+.2f}%)"
                f"<br>Forecast: ${pred_price:,.0f} ({pred_ret:+.2f}%)"
                f"<br>Error: {error:+.2f} pp"
                f"<br>Week: {row['week_end'].strftime('%b %d, %Y')}"
            )

    # Forecast trace
    fig.add_trace(
        go.Scatter(
            x=fore_x, y=fore_y,
            mode="markers+lines",
            name="Forecast",
            marker=dict(symbol="diamond", size=11, color=fore_colors,
                        line=dict(width=1, color="rgba(255,255,255,0.3)")),
            line=dict(color="#3b82f6", dash="dot", width=1.5),
            text=fore_text,
            hoverinfo="text",
        ),
        row=1, col=1,
    )

    # Actual close trace (only for weeks that already closed)
    if act_x:
        fig.add_trace(
            go.Scatter(
                x=act_x, y=act_y,
                mode="markers+lines",
                name="Actual Close",
                marker=dict(symbol="circle", size=10, color="#10b981",
                            line=dict(width=1, color="rgba(255,255,255,0.3)")),
                line=dict(color="#10b981", dash="solid", width=1.5),
                text=act_text,
                hoverinfo="text",
            ),
            row=1, col=1,
        )

    # Price columns (present in newly-generated CSVs)
    has_price_cols = "predicted_price" in predictions_df.columns

    # Annotation for the most recent (open) forecast only
    if fore_x:
        latest_fore_x = fore_x[-1]
        latest_fore_y = fore_y[-1]
        curr_row = predictions_df.iloc[-1]

        if has_price_cols:
            curr_price      = float(curr_row["predicted_price"])
            curr_range_low  = float(curr_row["price_range_low"])
            curr_range_high = float(curr_row["price_range_high"])
            ann_text = (
                f"CURRENT FORECAST<br>"
                f"<b>${curr_price:,.0f}</b><br>"
                f"${curr_range_low:,.0f} – ${curr_range_high:,.0f}"
            )
            # Shaded band spanning the forecast week
            x0_band = (curr_row["week_start"] - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
            x1_band = curr_row["week_end"].strftime("%Y-%m-%d")
            fig.add_shape(
                type="rect",
                xref="x", yref="y",
                x0=x0_band, x1=x1_band,
                y0=curr_range_low, y1=curr_range_high,
                fillcolor="rgba(59,130,246,0.10)",
                line=dict(color="rgba(59,130,246,0.35)", width=1, dash="dot"),
                layer="below",
            )
        else:
            sign = "+" if predicted_return_pct >= 0 else ""
            ann_text = f"CURRENT FORECAST<br>{sign}{predicted_return_pct:.2f}%"

        fig.add_annotation(
            x=latest_fore_x, y=latest_fore_y,
            text=ann_text,
            showarrow=True, arrowhead=2,
            arrowcolor="#f59e0b",
            font=dict(color="#f59e0b", size=11),
            bgcolor="rgba(0,0,0,0.6)",
            bordercolor="#f59e0b",
            row=1, col=1,
        )

    # Metrics table (most recent run) — read from metrics.json for full data
    import json as _json
    try:
        with open("metrics.json") as _f:
            mx = _json.load(_f)
    except Exception:
        mx = {}

    if mx:
        winner = mx.get("winning_model", "")
        row_colors = [
            ["#0f4c75" if winner == m else "#1e293b" for m in ["XGBoost", "RandomForest"]]
        ] * 6
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Model", "RMSE", "MAE", "Dir. Acc.", "Precision", "Recall / F1"],
                    fill_color="#1e293b",
                    font=dict(color="white", size=11),
                    align="center",
                ),
                cells=dict(
                    values=[
                        ["XGBoost", "RandomForest"],
                        [f"{mx.get('rmse_xgb', 0):.3f}", f"{mx.get('rmse_rf', 0):.3f}"],
                        [f"{mx.get('mae_xgb', 0):.3f}",  f"{mx.get('mae_rf', 0):.3f}"],
                        [f"{mx.get('dir_acc_xgb', 0):.1f}%", f"{mx.get('dir_acc_rf', 0):.1f}%"],
                        [f"{mx.get('precision_xgb', 0):.1f}%", f"{mx.get('precision_rf', 0):.1f}%"],
                        [f"{mx.get('recall_xgb', 0):.1f}% / {mx.get('f1_xgb', 0):.1f}",
                         f"{mx.get('recall_rf', 0):.1f}% / {mx.get('f1_rf', 0):.1f}"],
                    ],
                    fill_color=row_colors,
                    font=dict(color="white", size=11),
                    align="center",
                ),
            ),
            row=2, col=1,
        )

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    fig.update_layout(
        title=dict(
            text=f"The Oracle — S&P 500 Weekly Forecast | Updated: {now_str}",
            font=dict(size=18, color="white"),
        ),
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=750,
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.write_html(OUTPUT_PATH, include_plotlyjs="cdn", config={"displayModeBar": True})

    # Post-process: inject nav, CSS, and explanatory section
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        html = f.read()

    html = html.replace("<html>", '<html lang="en">', 1)
    html = html.replace("<head>", f"<head>{_HEAD_EXTRA}", 1)
    html = html.replace("<body>", f"<body>\n{_NAV_HTML}", 1)
    # Gather price data for info section (read from last CSV row if available)
    _curr = predictions_df.iloc[-1]
    _info_price = (
        float(_curr["predicted_price"])
        if has_price_cols and pd.notna(_curr.get("predicted_price"))
        else float(_curr["sp500_close_prev_friday"]) * (1 + predicted_return_pct / 100)
    )
    _info_low = (
        float(_curr["price_range_low"])
        if has_price_cols and pd.notna(_curr.get("price_range_low"))
        else _info_price
    )
    _info_high = (
        float(_curr["price_range_high"])
        if has_price_cols and pd.notna(_curr.get("price_range_high"))
        else _info_price
    )

    html = html.replace(
        "</body>",
        f"\n{_build_info_section(predicted_return_pct, _info_price, _info_low, _info_high, now_str)}\n</body>",
        1,
    )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[chart_generator] Dashboard generated at {OUTPUT_PATH}.")
