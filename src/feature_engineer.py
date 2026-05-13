import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

TRAIN_WINDOW_WEEKS = 104  # 2 years
MIN_ROWS_AFTER_DROPNA = 80

# sma_20 and sma_50 are computed as intermediates but replaced by
# price_to_sma20 (mean-reversion) and trend_spread (trend direction).
# Feeding raw price-level SMAs adds collinearity without extra signal.
FEATURE_COLS = [
    "rsi_14",
    "ema_12",
    "price_to_sma20",   # Close / SMA20 - 1  (mean-reversion distance)
    "trend_spread",     # (SMA20 - SMA50) / SMA50  (trend expansion/compression)
    "vix_weekly",
    "vix_change",       # week-over-week VIX % change (fear acceleration)
    "lag_ret_1w",
    "lag_ret_2w",
    "lag_ret_3w",
    "realized_vol",
]


def _rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=length - 1, min_periods=length).mean()
    avg_loss = loss.ewm(com=length - 1, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(gspc_weekly: pd.DataFrame, gspc_daily: pd.DataFrame, vix_weekly: pd.Series) -> pd.DataFrame:
    """
    Builds weekly feature DataFrame with no lookahead.
    All features for week W use only data with timestamp <= Friday close of W.
    """
    df = gspc_weekly.copy()

    # Momentum oscillator
    df["rsi_14"] = _rsi(df["Close"], length=14)

    # Moving averages (kept as intermediates; not fed raw to the model)
    sma_20 = df["Close"].rolling(20).mean()
    sma_50 = df["Close"].rolling(50).mean()
    df["ema_12"]        = df["Close"].ewm(span=12, adjust=False).mean()
    df["price_to_sma20"] = df["Close"] / sma_20 - 1
    df["trend_spread"]   = (sma_20 - sma_50) / sma_50 * 100  # % gap between short and long trend

    # VIX: level + week-over-week acceleration
    df = df.join(vix_weekly, how="left")
    df["vix_change"] = df["vix_weekly"].pct_change() * 100

    # Lag returns (momentum / autocorrelation features)
    weekly_ret = df["Close"].pct_change() * 100
    df["lag_ret_1w"] = weekly_ret.shift(1)
    df["lag_ret_2w"] = weekly_ret.shift(2)
    df["lag_ret_3w"] = weekly_ret.shift(3)

    # Intraweek realized volatility (std of 5 daily returns)
    daily_ret = gspc_daily["Close"].pct_change()
    realized_vol = daily_ret.resample("W-FRI").std() * 100
    realized_vol.name = "realized_vol"
    df = df.join(realized_vol, how="left")

    # Target: next week's percentage return
    df["target"] = weekly_ret.shift(-1)

    df = df.dropna(subset=FEATURE_COLS + ["target"])

    if len(df) < MIN_ROWS_AFTER_DROPNA:
        raise ValueError(
            f"Only {len(df)} rows after dropna (minimum {MIN_ROWS_AFTER_DROPNA}). "
            "Check yfinance connection or download period."
        )

    print(f"[feature_engineer] Clean dataset: {len(df)} weeks.")
    return df


def prepare_datasets(df: pd.DataFrame) -> tuple:
    """
    Splits into training window (rolling 2-year) and prediction row (current week).
    RobustScaler is fit exclusively on training data — never on the prediction row.
    Returns (X_train_scaled, y_train, X_pred_scaled, scaler, last_close, last_date).
    """
    train_df = df.iloc[-(TRAIN_WINDOW_WEEKS + 1):-1] if len(df) > TRAIN_WINDOW_WEEKS + 1 else df.iloc[:-1]
    pred_row = df.iloc[[-1]]

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["target"].values
    X_pred  = pred_row[FEATURE_COLS].values

    # RobustScaler uses median + IQR: resistant to extreme return outliers
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_pred_scaled  = scaler.transform(X_pred)

    last_close = pred_row["Close"].iloc[0]
    last_date  = pred_row.index[-1]

    print(f"[feature_engineer] Train: {len(X_train)} weeks. Predicting week after: {last_date.date()}")
    return X_train_scaled, y_train, X_pred_scaled, scaler, last_close, last_date
