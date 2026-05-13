import csv
import os
import shutil
from datetime import date, timedelta

import numpy as np
import pandas as pd
import joblib

MODEL_PATH = "model.joblib"
PREDICTIONS_PATH = "predictions.csv"

CSV_COLUMNS = [
    "week_start",
    "week_end",
    "predicted_price",
    "price_range_low",
    "price_range_high",
    "predicted_return_pct",
    "actual_return_pct",
    "actual_price",
    "winning_model",
    "rmse_xgb",
    "rmse_rf",
    "mae_xgb",
    "mae_rf",
    "dir_acc_xgb",
    "dir_acc_rf",
    "sp500_close_prev_friday",
]


def predict(X_pred: np.ndarray, last_close: float, last_date, metrics: dict, winning_model: str) -> float:
    model = joblib.load(MODEL_PATH)
    predicted_return_pct = float(model.predict(X_pred)[0])

    rmse_winner      = metrics["rmse_xgb"] if winning_model == "XGBoost" else metrics["rmse_rf"]
    predicted_price  = round(last_close * (1 + predicted_return_pct / 100), 2)
    price_range_low  = round(last_close * (1 + (predicted_return_pct - rmse_winner) / 100), 2)
    price_range_high = round(last_close * (1 + (predicted_return_pct + rmse_winner) / 100), 2)

    print(
        f"[predictor] Forecast: {predicted_return_pct:+.4f}% -> "
        f"${predicted_price:,.2f}  (Range: ${price_range_low:,.2f} – ${price_range_high:,.2f})"
    )

    week_start = last_date + timedelta(days=3)
    week_end   = last_date + timedelta(days=7)

    _append_prediction(
        week_start=week_start.date() if hasattr(week_start, "date") else week_start,
        week_end=week_end.date() if hasattr(week_end, "date") else week_end,
        predicted_return_pct=predicted_return_pct,
        predicted_price=predicted_price,
        price_range_low=price_range_low,
        price_range_high=price_range_high,
        winning_model=winning_model,
        metrics=metrics,
        sp500_close_prev_friday=last_close,
    )

    return predicted_return_pct


def _append_prediction(
    week_start, week_end,
    predicted_return_pct, predicted_price, price_range_low, price_range_high,
    winning_model, metrics, sp500_close_prev_friday,
):
    file_exists = os.path.isfile(PREDICTIONS_PATH)

    if file_exists:
        with open(PREDICTIONS_PATH, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("week_start") == str(week_start):
                    print(f"[predictor] Already have prediction for {week_start} — skipping duplicate.")
                    return

    with open(PREDICTIONS_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "week_start":             week_start,
            "week_end":               week_end,
            "predicted_price":        predicted_price,
            "price_range_low":        price_range_low,
            "price_range_high":       price_range_high,
            "predicted_return_pct":   round(predicted_return_pct, 4),
            "actual_return_pct":      "",
            "actual_price":           "",
            "winning_model":          winning_model,
            "rmse_xgb":               round(metrics["rmse_xgb"], 4),
            "rmse_rf":                round(metrics["rmse_rf"],  4),
            "mae_xgb":                round(metrics["mae_xgb"],  4),
            "mae_rf":                 round(metrics["mae_rf"],   4),
            "dir_acc_xgb":            round(metrics["dir_acc_xgb"], 2),
            "dir_acc_rf":             round(metrics["dir_acc_rf"],  2),
            "sp500_close_prev_friday": round(sp500_close_prev_friday, 2),
        })
    print(f"[predictor] Row appended to {PREDICTIONS_PATH}.")


def update_actuals(gspc_weekly: pd.DataFrame):
    """
    Fills actual_return_pct and actual_price in predictions.csv for rows whose
    week_end has already passed and whose actual_return_pct is still empty.
    """
    if not os.path.isfile(PREDICTIONS_PATH):
        print("[predictor] predictions.csv does not exist yet — nothing to update.")
        return

    rows = []
    updated = 0
    with open(PREDICTIONS_PATH, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        for row in reader:
            if row["actual_return_pct"] == "" and row["week_end"]:
                week_end = pd.Timestamp(row["week_end"])
                if week_end <= pd.Timestamp(date.today()):
                    try:
                        close_prev = float(row["sp500_close_prev_friday"])
                        subset = gspc_weekly[gspc_weekly.index <= week_end]
                        if not subset.empty:
                            close_curr = float(subset["Close"].iloc[-1])
                            row["actual_return_pct"] = round((close_curr / close_prev - 1) * 100, 4)
                            if "actual_price" in fieldnames:
                                row["actual_price"] = round(close_curr, 2)
                            updated += 1
                    except Exception as e:
                        print(f"[predictor] Warning: could not compute actual for {row['week_end']}: {e}")
            rows.append(row)

    if updated:
        tmp = PREDICTIONS_PATH + ".tmp"
        with open(tmp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        shutil.move(tmp, PREDICTIONS_PATH)
        print(f"[predictor] {updated} row(s) updated with actual return/price.")
    else:
        print("[predictor] No pending rows to update.")
