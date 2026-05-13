"""
El Oraculo del S&P 500 - Pipeline principal.
Uso: python pipeline.py
"""
import io
import json
import sys
import os

# Fuerza UTF-8 en stdout/stderr para compatibilidad cross-platform
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Asegurar que src/ esté en el path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import load_data
from feature_engineer import build_features, prepare_datasets
from trainer import train_and_select
from predictor import predict, update_actuals
from chart_generator import generate_chart


def main():
    print("=" * 60)
    print("El Oráculo del S&P 500 — Iniciando pipeline")
    print("=" * 60)

    # Paso 1: Descarga de datos
    print("\n[pipeline] PASO 1: Descarga de datos")
    gspc_daily, gspc_weekly, vix_weekly = load_data()

    # Paso 2: Actualizar retorno real de la semana anterior (si ya cerró)
    print("\n[pipeline] PASO 2: Actualizando retorno real de semana anterior")
    update_actuals(gspc_weekly)

    # Paso 3: Feature engineering
    print("\n[pipeline] PASO 3: Feature engineering")
    df = build_features(gspc_weekly, gspc_daily, vix_weekly)

    # Paso 4: Preparar datasets
    print("\n[pipeline] PASO 4: Preparar datasets de entrenamiento")
    X_train, y_train, X_pred, scaler, last_close, last_date = prepare_datasets(df)

    # Paso 5: Entrenamiento y selección de modelo
    print("\n[pipeline] PASO 5: Entrenamiento y selección de modelo")
    model, winning_model, metrics = train_and_select(X_train, y_train)

    # Paso 6: Predicción
    print("\n[pipeline] PASO 6: Generando predicción")
    predicted_return_pct = predict(X_pred, last_close, last_date, metrics, winning_model)
    metrics["predicted_return_pct"] = round(predicted_return_pct, 4)

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Paso 7: Generar dashboard
    print("\n[pipeline] PASO 7: Generando dashboard")
    generate_chart(gspc_weekly, predicted_return_pct, winning_model)

    print("\n" + "=" * 60)
    print(f"Pipeline completado.")
    print(f"  Modelo ganador : {winning_model}")
    print(f"  Predicción     : {predicted_return_pct:+.4f}% para la semana siguiente")
    print(f"  Dashboard      : index.html")
    print("=" * 60)


if __name__ == "__main__":
    main()
