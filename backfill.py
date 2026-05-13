"""
Backfills predictions.csv with walk-forward historical predictions.
Run once:  python backfill.py
Run with custom depth:  python backfill.py 12
"""
import csv
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from datetime import timedelta

from data_loader import load_data
from feature_engineer import build_features, prepare_datasets, TRAIN_WINDOW_WEEKS
from trainer import train_and_select
from predictor import PREDICTIONS_PATH, CSV_COLUMNS

N_BACKFILL = int(sys.argv[1]) if len(sys.argv) > 1 else 10


def backfill():
    print("=" * 60)
    print(f"Backfilling {N_BACKFILL} weeks of historical predictions")
    print("=" * 60)

    gspc_daily, gspc_weekly, vix_weekly = load_data()
    df = build_features(gspc_weekly, gspc_daily, vix_weekly)

    print(f"\nDataset ready: {len(df)} rows available.\n")

    new_rows = []

    # b=N_BACKFILL is the oldest week; b=1 is last week
    for b in range(N_BACKFILL, 0, -1):
        df_hist = df.iloc[:-b]

        if len(df_hist) < TRAIN_WINDOW_WEEKS + 10:
            print(f"[backfill] Offset -{b}: not enough data, skipping.")
            continue

        print(f"[backfill] Offset -{b}: predicting week after {df_hist.index[-1].date()}...")

        X_train, y_train, X_pred, scaler, last_close, last_date = prepare_datasets(df_hist)
        model, winning_model, metrics = train_and_select(X_train, y_train)

        predicted_return_pct = float(model.predict(X_pred)[0])

        # Actual return for the predicted week is the target of the pred_row in the full df
        # df["target"].iloc[-(b+1)] = (Close[-(b)] / Close[-(b+1)] - 1) * 100
        actual_return_pct = round(float(df["target"].iloc[-(b + 1)]), 4)

        rmse_winner      = metrics["rmse_xgb"] if winning_model == "XGBoost" else metrics["rmse_rf"]
        predicted_price  = round(last_close * (1 + predicted_return_pct / 100), 2)
        price_range_low  = round(last_close * (1 + (predicted_return_pct - rmse_winner) / 100), 2)
        price_range_high = round(last_close * (1 + (predicted_return_pct + rmse_winner) / 100), 2)
        actual_price     = round(last_close * (1 + actual_return_pct / 100), 2)

        week_start = last_date + timedelta(days=3)
        week_end = last_date + timedelta(days=7)

        row = {
            "week_start": week_start.date() if hasattr(week_start, "date") else week_start,
            "week_end":   week_end.date()   if hasattr(week_end,   "date") else week_end,
            "predicted_price":      predicted_price,
            "price_range_low":      price_range_low,
            "price_range_high":     price_range_high,
            "predicted_return_pct": round(predicted_return_pct, 4),
            "actual_return_pct":    actual_return_pct,
            "actual_price":         actual_price,
            "winning_model":        winning_model,
            "rmse_xgb":   round(metrics["rmse_xgb"], 4),
            "rmse_rf":    round(metrics["rmse_rf"],  4),
            "mae_xgb":    round(metrics["mae_xgb"],  4),
            "mae_rf":     round(metrics["mae_rf"],   4),
            "dir_acc_xgb": round(metrics["dir_acc_xgb"], 2),
            "dir_acc_rf":  round(metrics["dir_acc_rf"],  2),
            "sp500_close_prev_friday": round(last_close, 2),
        }
        new_rows.append(row)
        print(f"         Predicted: {predicted_return_pct:+.4f}%  Actual: {actual_return_pct:+.4f}%  Model: {winning_model}")

    if not new_rows:
        print("No rows generated.")
        return

    # Read existing rows, avoid duplicates by week_start
    existing_starts = set()
    existing_rows = []
    if os.path.exists(PREDICTIONS_PATH):
        with open(PREDICTIONS_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing_starts.add(str(r["week_start"]))
                existing_rows.append(r)

    deduplicated = [r for r in new_rows if str(r["week_start"]) not in existing_starts]

    # Historical rows go first (chronological order)
    all_rows = deduplicated + existing_rows

    with open(PREDICTIONS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nAdded {len(deduplicated)} new rows. Total in CSV: {len(all_rows)}")
    print(f"Done. Run 'python pipeline.py' to regenerate the dashboard.")


if __name__ == "__main__":
    backfill()
