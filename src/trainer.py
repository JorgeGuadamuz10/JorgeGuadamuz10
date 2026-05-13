import json
from datetime import date

import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

MODEL_PATH = "model.joblib"
METRICS_PATH = "metrics.json"
N_SPLITS = 5

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0,
)

RF_PARAMS = dict(
    n_estimators=300,
    max_depth=5,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)


def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_pred) == np.sign(y_true)) * 100)


def _confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Directional confusion matrix.
    Positive class = market goes UP (return > 0).
    """
    up_pred = np.sign(y_pred) > 0
    up_true = np.sign(y_true) > 0

    tp = int(np.sum(up_pred & up_true))
    fp = int(np.sum(up_pred & ~up_true))
    tn = int(np.sum(~up_pred & ~up_true))
    fn = int(np.sum(~up_pred & up_true))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision * 100, 1),
        "recall":    round(recall    * 100, 1),
        "f1":        round(f1        * 100, 1),
    }


def _walk_forward_cv(model_cls, params: dict, X: np.ndarray, y: np.ndarray) -> dict:
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    rmses, maes, dir_accs = [], [], []
    all_true, all_pred = [], []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        m = model_cls(**params)
        m.fit(X[train_idx], y[train_idx])
        preds = m.predict(X[val_idx])
        rmses.append(np.sqrt(mean_squared_error(y[val_idx], preds)))
        maes.append(mean_absolute_error(y[val_idx], preds))
        dir_accs.append(_directional_accuracy(y[val_idx], preds))
        all_true.extend(y[val_idx])
        all_pred.extend(preds)

    conf = _confusion_metrics(np.array(all_true), np.array(all_pred))

    return {
        "rmse":    float(np.mean(rmses)),
        "mae":     float(np.mean(maes)),
        "dir_acc": float(np.mean(dir_accs)),
        **conf,
    }


def train_and_select(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    """
    Runs walk-forward CV for XGBoost and RandomForest.
    Selects winner by lowest RMSE, retrains on full X_train,
    serializes to model.joblib and saves metrics.json.
    Returns (model, winning_model_name, metrics_dict).
    """
    print("[trainer] Walk-forward CV — XGBoost...")
    xgb_m = _walk_forward_cv(XGBRegressor, XGB_PARAMS, X_train, y_train)
    print(
        f"[trainer] XGBoost -> RMSE={xgb_m['rmse']:.4f}  MAE={xgb_m['mae']:.4f}"
        f"  DirAcc={xgb_m['dir_acc']:.1f}%"
        f"  Prec={xgb_m['precision']:.1f}%  Rec={xgb_m['recall']:.1f}%  F1={xgb_m['f1']:.1f}%"
    )

    print("[trainer] Walk-forward CV — RandomForest...")
    rf_m = _walk_forward_cv(RandomForestRegressor, RF_PARAMS, X_train, y_train)
    print(
        f"[trainer] RandomForest -> RMSE={rf_m['rmse']:.4f}  MAE={rf_m['mae']:.4f}"
        f"  DirAcc={rf_m['dir_acc']:.1f}%"
        f"  Prec={rf_m['precision']:.1f}%  Rec={rf_m['recall']:.1f}%  F1={rf_m['f1']:.1f}%"
    )

    if xgb_m["rmse"] <= rf_m["rmse"]:
        winning_name = "XGBoost"
        winner = XGBRegressor(**XGB_PARAMS)
    else:
        winning_name = "RandomForest"
        winner = RandomForestRegressor(**RF_PARAMS)

    print(f"[trainer] Winner: {winning_name}. Retraining on full dataset...")
    winner.fit(X_train, y_train)
    joblib.dump(winner, MODEL_PATH)
    print(f"[trainer] Model saved to {MODEL_PATH}.")

    metrics = {
        "run_date":     str(date.today()),
        "winning_model": winning_name,
        # Regression metrics
        "rmse_xgb":    xgb_m["rmse"],   "rmse_rf":    rf_m["rmse"],
        "mae_xgb":     xgb_m["mae"],    "mae_rf":     rf_m["mae"],
        # Directional accuracy
        "dir_acc_xgb": xgb_m["dir_acc"], "dir_acc_rf": rf_m["dir_acc"],
        # Confusion matrix — aggregated across all CV folds
        "precision_xgb": xgb_m["precision"], "precision_rf": rf_m["precision"],
        "recall_xgb":    xgb_m["recall"],    "recall_rf":    rf_m["recall"],
        "f1_xgb":        xgb_m["f1"],        "f1_rf":        rf_m["f1"],
        "tp_xgb": xgb_m["tp"], "fp_xgb": xgb_m["fp"],
        "tn_xgb": xgb_m["tn"], "fn_xgb": xgb_m["fn"],
        "tp_rf":  rf_m["tp"],  "fp_rf":  rf_m["fp"],
        "tn_rf":  rf_m["tn"],  "fn_rf":  rf_m["fn"],
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[trainer] Metrics saved to {METRICS_PATH}.")

    return winner, winning_name, metrics
