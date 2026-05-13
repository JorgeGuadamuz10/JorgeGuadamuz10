# Project Plan: El Oráculo del S&P 500

## 1. Project Summary

El Oráculo es un sistema de predicción financiera que estima el **retorno porcentual semanal del índice S&P 500** usando Machine Learning. Cada lunes, un pipeline automatizado descarga datos actualizados de mercado vía `yfinance`, construye un conjunto de features de indicadores técnicos (RSI, medias móviles, VIX, lags), reentrena dos modelos (XGBoost y Random Forest Regressor) con una ventana deslizante de los últimos 2 años, compara su desempeño, y publica el resultado en una página web estática vía GitHub Pages. El sistema está diseñado como pieza de portafolio profesional: muestra habilidades en feature engineering, evaluación de modelos de ML, automatización con GitHub Actions y visualización interactiva con Plotly. La v1 predice únicamente el retorno del S&P 500 (`^GSPC`) y muestra las últimas 4 semanas en un gráfico de velas japonesas con la predicción superpuesta.

---

## 2. Tech Stack

| Capa | Tecnología | Versión | Justificación |
|------|-----------|---------|---------------|
| Lenguaje | Python | 3.11 | Ecosistema ML más completo; soporte LTS activo |
| ML — Modelo A | XGBoost | 2.x | Maneja relaciones no lineales, robusto en datos tabulares pequeños |
| ML — Modelo B | scikit-learn (RandomForestRegressor) | 1.4.x | Mismo API que XGBoost vía sklearn, fácil comparación |
| Feature engineering | pandas-ta | 0.3.x | Calcula RSI, SMA, EMA con una línea; evita reimplementar fórmulas |
| Datos en vivo | yfinance | 0.2.x | Acceso gratuito a precios históricos y actuales del S&P 500 y VIX |
| Visualización | Plotly | 5.x | Gráfico de velas interactivo exportable como HTML estático |
| Serialización de modelos | joblib | incluido en scikit-learn | Guarda el modelo ganador entre ejecuciones |
| Automatización | GitHub Actions | — | Cron gratuito; no requiere servidor propio |
| Hosting del dashboard | GitHub Pages | — | Sirve el HTML estático generado por Plotly; URL pública gratuita |
| Persistencia de predicciones | CSV plano (`predictions.csv`) | — | Acumula predicciones semana a semana; legible sin base de datos |

### Librerías clave

- **`yfinance`** — descarga OHLCV semanal del `^GSPC` y cierre diario del `^VIX`
- **`pandas-ta`** — calcula `RSI(14)`, `SMA(20)`, `SMA(50)`, `EMA(12)` directamente sobre DataFrames de pandas
- **`xgboost`** — modelo principal tipo gradient boosting para regresión
- **`scikit-learn`** — `RandomForestRegressor`, `TimeSeriesSplit`, `RobustScaler`, métricas (`mean_squared_error`, `mean_absolute_error`)
- **`plotly`** — `go.Candlestick` + `go.Scatter` para velas + líneas de predicción; exportado con `fig.write_html()`
- **`joblib`** — `dump()` / `load()` para serializar el modelo ganador en `model.joblib`
- **`numpy`** — cálculo de volatilidad realizada y operaciones vectoriales

---

## 3. Arquitectura

### Resumen del sistema

El pipeline corre completamente dentro de un workflow de GitHub Actions activado por un cron cada lunes a las 10:00 UTC (cuando el mercado americano ya cerró el viernes anterior). El script Python descarga datos, construye features, reentrena ambos modelos con los últimos 2 años, selecciona al ganador según RMSE en validación walk-forward, genera la predicción de la semana en curso, actualiza `predictions.csv` y regenera el archivo `index.html` con el gráfico de Plotly. GitHub Actions hace commit de esos dos archivos al repositorio, y GitHub Pages sirve el `index.html` automáticamente en la URL pública.

### Mapa de componentes

| Componente | Tecnología | Responsabilidad | Se comunica con |
|-----------|-----------|----------------|----------------|
| `pipeline.py` | Python 3.11 | Orquesta todo el flujo de extremo a extremo | Todos los demás módulos |
| `data_loader.py` | yfinance + pandas | Descarga OHLCV semanal `^GSPC` y VIX diario; resamplea a frecuencia semanal | `feature_engineer.py` |
| `feature_engineer.py` | pandas-ta + pandas | Construye las 10 features; elimina NaNs; aplica `RobustScaler` | `trainer.py` |
| `trainer.py` | scikit-learn + XGBoost | Walk-forward CV, entrena ambos modelos, selecciona ganador, serializa | `predictor.py`, `model.joblib` |
| `predictor.py` | joblib + pandas | Carga `model.joblib`, construye features de la semana actual, genera predicción | `predictions.csv` |
| `chart_generator.py` | Plotly | Lee `predictions.csv`, genera `index.html` con gráfico interactivo | `index.html` |
| GitHub Actions workflow | `.github/workflows/weekly.yml` | Ejecuta el pipeline cada lunes; hace commit de outputs | Repositorio GitHub |
| GitHub Pages | — | Sirve `index.html` como página pública | Visitantes del portafolio |

### Diagrama de arquitectura

```
┌─────────────────────────────────────────────────────────────────────┐
│                     GitHub Actions (cron: lunes 10:00 UTC)          │
│                                                                     │
│  ┌──────────────┐    ┌────────────────────┐    ┌─────────────────┐ │
│  │ data_loader  │───▶│ feature_engineer   │───▶│    trainer      │ │
│  │              │    │                    │    │                 │ │
│  │ yfinance:    │    │ RSI(14)            │    │ TimeSeriesSplit │ │
│  │  ^GSPC OHLCV │    │ SMA(20), SMA(50)   │    │ XGBoost        │ │
│  │  ^VIX close  │    │ EMA(12)            │    │ RandomForest   │ │
│  └──────────────┘    │ VIX semanal        │    │ → model.joblib │ │
│                      │ lag_ret_1/2/3w     │    └────────┬────────┘ │
│                      │ realized_vol       │             │          │
│                      └────────────────────┘             ▼          │
│                                                  ┌─────────────┐   │
│                                                  │  predictor  │   │
│                                                  │             │   │
│                                                  │ predicción  │   │
│                                                  │ semana W+1  │   │
│                                                  └──────┬──────┘   │
│                                                         │           │
│                                                         ▼           │
│                                              predictions.csv        │
│                                                (acumulativo)        │
│                                                         │           │
│                                                         ▼           │
│                                             ┌──────────────────┐   │
│                                             │ chart_generator  │   │
│                                             │ Plotly candlestk │   │
│                                             │ + scatter preds  │   │
│                                             └────────┬─────────┘   │
│                                                      │              │
│                                                 index.html          │
└──────────────────────────────────────────────────────┼─────────────┘
                                                        │
                                              git commit & push
                                                        │
                                                        ▼
                                              ┌──────────────────┐
                                              │  GitHub Pages    │
                                              │  (URL pública)   │
                                              │  tu-usuario      │
                                              │  .github.io/     │
                                              │  oraculo-sp500   │
                                              └──────────────────┘
```

### Entornos

| Entorno | Propósito | Ejecución |
|---------|----------|-----------|
| Local (dev) | Desarrollo, pruebas manuales | `python pipeline.py` en Windows |
| Producción | Pipeline semanal automático | GitHub Actions + GitHub Pages |

---

## 4. ML System Design

### Definición del problema

- **Tipo de tarea**: Regresión sobre series temporales
- **Variable objetivo (`y`)**: Retorno porcentual semanal del S&P 500, definido como:
  `y = (close_W / close_W-1 - 1) * 100`
  donde `close_W` es el precio de cierre del viernes de la semana `W`.
  Ejemplo: si el S&P cerró en 5200 el viernes anterior y en 5330 este viernes, `y = 2.50`.
- **Horizonte de predicción**: 1 semana hacia adelante (se predice `y` de la semana `W+1` usando features conocidos al cierre de la semana `W`)

### Features de entrada

| Feature | Tipo | Fuente | Cómo se calcula |
|---------|------|--------|----------------|
| `rsi_14` | float | `^GSPC` diario → resampleado | RSI de 14 períodos semanales vía `pandas_ta.rsi(length=14)` |
| `sma_20` | float | `^GSPC` semanal | Media móvil simple de 20 semanas |
| `sma_50` | float | `^GSPC` semanal | Media móvil simple de 50 semanas |
| `ema_12` | float | `^GSPC` semanal | Media móvil exponencial de 12 semanas |
| `price_to_sma20` | float | derivado | `close / sma_20 - 1`: distancia relativa al soporte |
| `vix_weekly` | float | `^VIX` diario → último valor de la semana | Cierre del VIX del viernes de la semana `W` |
| `lag_ret_1w` | float | `^GSPC` semanal | Retorno % de la semana `W-1` |
| `lag_ret_2w` | float | `^GSPC` semanal | Retorno % de la semana `W-2` |
| `lag_ret_3w` | float | `^GSPC` semanal | Retorno % de la semana `W-3` |
| `realized_vol` | float | `^GSPC` diario | Desviación estándar de los retornos diarios dentro de la semana `W` (5 valores) |

**Total: 10 features**. Todos se escalan con `RobustScaler` antes del entrenamiento.

**Regla crítica de no-lookahead**: todos los features de la semana `W` se calculan usando exclusivamente precios con timestamp ≤ cierre del viernes de `W`. El target `y` de la semana `W` usa el cierre del viernes de `W` y el cierre del viernes de `W+1`. Por tanto, para predecir `W+1` en producción se usan features de `W`, que ya son datos históricos conocidos.

### Pipeline de datos

| Paso | Descripción | Herramienta | Output |
|------|-------------|-------------|--------|
| Ingesta | Descarga `^GSPC` diario y `^VIX` diario desde yfinance; período: últimos 5 años (training) + semana actual | `yfinance.download()` | DataFrame con columnas OHLCV + VIX |
| Resampleo | Convierte datos diarios a frecuencia semanal (`W-FRI`): Open=first, High=max, Low=min, Close=last, Volume=sum | `pandas.resample('W-FRI')` | DataFrame semanal |
| Feature engineering | Calcula los 10 features descritos arriba | `pandas_ta` + `pandas` | DataFrame con columnas de features |
| Construcción del target | `y = (close.shift(-1) / close - 1) * 100`; elimina última fila (no hay target futuro conocido) | pandas | Columna `target` |
| Limpieza | Elimina filas con NaN (primeras ~50 semanas por warm-up de SMA50) | `df.dropna()` | Dataset limpio |
| Split de entrenamiento | Ventana deslizante: últimas 104 semanas (2 años) al momento de cada ejecución semanal | índices temporales | `X_train`, `y_train` |
| Escalado | `RobustScaler` fit exclusivamente sobre `X_train`; transform sobre `X_train` y `X_pred` | `sklearn.preprocessing.RobustScaler` | Arrays escalados |
| Almacenamiento intermedio | Dataset procesado en memoria (no se persiste en disco) | — | — |

### Selección de modelos

| Modelo | Por qué considerarlo | Limitación | ¿Seleccionado? |
|--------|---------------------|-----------|---------------|
| XGBoost Regressor | Maneja relaciones no lineales entre indicadores técnicos; resistente a outliers; rápido en datos tabulares pequeños (~104 filas) | Puede sobreajustarse si los hiperparámetros son agresivos | ✅ Sí |
| RandomForest Regressor | Ensemble de árboles con alta varianza reducida por bagging; menos propenso a overfitting que XGBoost sin tuning | Más lento de entrenar que XGBoost; predicciones menos extremas | ✅ Sí |
| Baseline (media histórica) | Referencia mínima: predice siempre el retorno promedio histórico de las últimas 104 semanas | No aprende nada; sirve solo como piso de comparación | ✅ Sí (como benchmark) |

### Configuración de los modelos

**XGBoost Regressor** (parámetros fijos en v1, sin tuning automático):
```python
XGBRegressor(
    n_estimators=300,
    max_depth=3,          # poco profundo para evitar overfitting en ~100 filas
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)
```

**RandomForest Regressor**:
```python
RandomForestRegressor(
    n_estimators=300,
    max_depth=5,
    min_samples_leaf=5,   # evita hojas con muy pocas observaciones
    random_state=42,
    n_jobs=-1
)
```

### Validación walk-forward

Se usa `TimeSeriesSplit(n_splits=5)` de scikit-learn. En cada fold, el conjunto de entrenamiento es siempre anterior al de validación — nunca hay mezcla temporal.

```
Fold 1: Train [sem 1..70]  → Val [sem 71..83]
Fold 2: Train [sem 1..83]  → Val [sem 84..91]
Fold 3: Train [sem 1..91]  → Val [sem 92..97]
Fold 4: Train [sem 1..97]  → Val [sem 98..101]
Fold 5: Train [sem 1..101] → Val [sem 102..104]
```

El modelo que obtenga menor RMSE promedio en los 5 folds es declarado **ganador** y se reentrena sobre las 104 semanas completas antes de hacer la predicción final.

### Métricas de evaluación

| Métrica | Fórmula | Interpretación |
|---------|---------|----------------|
| RMSE | `sqrt(mean((y_pred - y_real)^2))` | Penaliza errores grandes; en % puntos de retorno |
| MAE | `mean(abs(y_pred - y_real))` | Error absoluto promedio; más intuitivo |
| Directional Accuracy | `mean(sign(y_pred) == sign(y_real)) * 100` | % de semanas que el modelo acertó si el mercado subía o bajaba |

Las métricas se calculan sobre los folds de validación walk-forward. Se imprimen en el log de GitHub Actions y se guardan en `metrics.json` para referencia.

### Serving / predicción semanal

- **Modo**: batch (una predicción por ejecución)
- **Input**: features de la semana actual `W` (datos conocidos al momento de correr el lunes)
- **Output**: `predicted_return_pct` — un float; ejemplo: `1.47` (significa +1.47% esperado para la semana)
- **Reentrenamiento**: cada lunes, siempre, con la ventana deslizante de 2 años
- **Versionado del modelo**: `model.joblib` se sobreescribe cada semana (v1 no requiere historial de modelos)

---

## 5. Pipeline de automatización

### Trigger

Cron de GitHub Actions: **todos los lunes a las 10:00 UTC** (cuando ya cerró el mercado americano el viernes previo).

```yaml
# .github/workflows/weekly.yml
on:
  schedule:
    - cron: '0 10 * * 1'
  workflow_dispatch:     # permite correr manualmente desde la UI de GitHub
```

### Pasos de ejecución

| Paso | Nombre | Descripción | Input | Output | En caso de fallo |
|------|--------|-------------|-------|--------|-----------------|
| 1 | Setup Python | Instala Python 3.11 y dependencias desde `requirements.txt` | `requirements.txt` | Entorno listo | Falla el job; notificación por email de GitHub |
| 2 | Descarga de datos | `data_loader.py` llama a `yfinance.download('^GSPC', period='5y')` y `yfinance.download('^VIX', period='2y')` | Internet / yfinance API | `gspc_raw.pkl`, `vix_raw.pkl` en memoria | Retry 3 veces con 30s de espera; si falla, aborta con exit code 1 |
| 3 | Feature engineering | `feature_engineer.py` construye los 10 features; dropna; aplica RobustScaler | DataFrames crudos | `X_train`, `y_train`, `X_pred` (semana actual) | Aborta si quedan menos de 80 filas tras dropna |
| 4 | Entrenamiento y selección | `trainer.py` corre walk-forward CV para XGBoost y RF; calcula métricas; declara ganador; reentrena ganador en dataset completo; serializa en `model.joblib` | `X_train`, `y_train` | `model.joblib`, `metrics.json` | Aborta; no actualiza `predictions.csv` |
| 5 | Predicción | `predictor.py` carga `model.joblib`; transforma `X_pred`; genera `predicted_return_pct` | `model.joblib`, `X_pred` | Fila nueva para `predictions.csv` |  Aborta |
| 6 | Actualización del CSV | Appends una fila con `week_start`, `week_end`, `predicted_return_pct`, `actual_return_pct` (NaN si aún no cerró), `winning_model`, `rmse_xgb`, `rmse_rf`, `dir_acc_xgb`, `dir_acc_rf` | `predictions.csv` existente | `predictions.csv` actualizado | Aborta |
| 7 | Actualización del actual | Rellena `actual_return_pct` de la semana anterior (que ya cerró) en `predictions.csv` | `predictions.csv`, yfinance | `predictions.csv` con actual de semana previa | Loguea warning; no aborta |
| 8 | Generación del dashboard | `chart_generator.py` lee las últimas 4 semanas de `predictions.csv`; genera `index.html` | `predictions.csv` | `index.html` | Aborta |
| 9 | Commit y push | `git add predictions.csv index.html metrics.json model.joblib` + commit + push al branch `main` | Archivos generados | Repo actualizado → GitHub Pages se republica | Retry 1 vez; si falla, loguea error |

### Manejo de errores

- **Política de retry**: pasos de descarga de datos → 3 reintentos con backoff de 30 segundos.
- **Fallo total**: si cualquier paso de 2 al 8 falla, GitHub Actions reporta el job como fallido; GitHub envía email automático al dueño del repositorio.
- **Sin dead-letter**: v1 no tiene cola de mensajes fallidos; el CSV simplemente no se actualiza esa semana.
- **Alerta visual**: el README incluye un badge de GitHub Actions (`[![Weekly Pipeline](URL_BADGE)](URL_WORKFLOW)`) que se pone rojo si el último run falló.

### Observabilidad

- **Logs**: todos los pasos hacen `print()` con timestamps al stdout de GitHub Actions; visible en la pestaña Actions del repositorio.
- **`metrics.json`**: persiste en el repo las métricas de cada run semanal:
  ```json
  {
    "run_date": "2025-05-12",
    "winning_model": "XGBoost",
    "rmse_xgb": 1.23,
    "rmse_rf": 1.41,
    "mae_xgb": 0.98,
    "dir_acc_xgb": 57.3,
    "dir_acc_rf": 54.1,
    "predicted_return_pct": 1.47
  }
  ```
- **GitHub Actions badge**: en README, muestra ✅ o ❌ del último run.

---

## 6. Dashboard — Especificación del gráfico

### Archivo: `index.html`

Generado por Plotly con `fig.write_html('index.html', include_plotlyjs='cdn')`. Servido por GitHub Pages.

### Contenido del gráfico

El gráfico tiene **dos capas superpuestas** en el mismo eje de tiempo:

**Capa 1 — Velas japonesas** (`go.Candlestick`):
- Fuente: precios OHLC semanales del `^GSPC` de las últimas 4 semanas
- Velas verdes = semana alcista (close > open); rojas = bajista
- Hover muestra: Open, High, Low, Close, semana

**Capa 2 — Predicciones** (`go.Scatter`, modo `markers+lines`):
- Punto por semana en el precio de cierre predicho implícito (calculado como `close_semana_anterior * (1 + predicted_return_pct/100)`)
- Color: azul con marcador de diamante
- Línea punteada conecta los puntos de predicción
- Hover muestra: "Predicción: +1.47%" y "Modelo: XGBoost"
- Semana futura (aún sin cerrar): marcador en naranja con etiqueta "PREDICCIÓN ACTUAL"

**Panel inferior** (subplot):
- Tabla de métricas del último run: RMSE, MAE, Directional Accuracy para cada modelo
- Ganador destacado en negrita

**Título del gráfico**: "🔮 El Oráculo del S&P 500 — Predicción semanal | Actualizado: [fecha]"

**Layout**:
- Fondo oscuro (`template='plotly_dark'`)
- Selector de rango deshabilitado (siempre muestra las 4 semanas fijas)
- `config={'displayModeBar': True}` para que el usuario pueda hacer zoom y pan

---

## 7. Modelo de datos — `predictions.csv`

Archivo CSV plano acumulativo. Una fila por semana ejecutada.

| Columna | Tipo | Descripción | Ejemplo |
|---------|------|-------------|---------|
| `week_start` | date (YYYY-MM-DD) | Lunes de la semana predicha | `2025-05-12` |
| `week_end` | date (YYYY-MM-DD) | Viernes de la semana predicha | `2025-05-16` |
| `predicted_return_pct` | float | Retorno % predicho por el modelo ganador | `1.47` |
| `actual_return_pct` | float / NaN | Retorno % real (se rellena la semana siguiente) | `0.83` |
| `winning_model` | string | `"XGBoost"` o `"RandomForest"` | `"XGBoost"` |
| `rmse_xgb` | float | RMSE walk-forward de XGBoost en ese run | `1.23` |
| `rmse_rf` | float | RMSE walk-forward de Random Forest en ese run | `1.41` |
| `mae_xgb` | float | MAE walk-forward de XGBoost | `0.98` |
| `mae_rf` | float | MAE walk-forward de Random Forest | `1.12` |
| `dir_acc_xgb` | float | Directional Accuracy % de XGBoost | `57.3` |
| `dir_acc_rf` | float | Directional Accuracy % de Random Forest | `54.1` |
| `sp500_close_prev_friday` | float | Precio de cierre del S&P 500 el viernes anterior | `5210.45` |

---

## 8. Estructura del repositorio

```
oraculo-sp500/
├── .github/
│   └── workflows/
│       └── weekly.yml          # GitHub Actions: cron + steps
├── src/
│   ├── data_loader.py          # Descarga yfinance, resamplea a semanal
│   ├── feature_engineer.py     # Construye los 10 features + RobustScaler
│   ├── trainer.py              # Walk-forward CV, selección de modelo, serialización
│   ├── predictor.py            # Carga model.joblib, genera predicción
│   └── chart_generator.py      # Lee predictions.csv, genera index.html
├── pipeline.py                 # Orquestador: llama a los módulos en orden
├── predictions.csv             # Histórico acumulativo de predicciones
├── metrics.json                # Métricas del último run
├── model.joblib                # Modelo ganador serializado
├── index.html                  # Dashboard Plotly (GitHub Pages lo sirve)
├── requirements.txt            # Dependencias con versiones fijas
└── README.md                   # Badge de Actions + descripción del proyecto
```

---

## 9. Fases del proyecto

### Fase 1 — Pipeline local funcional
**Meta**: que `python pipeline.py` corra de extremo a extremo en local sin errores y genere un `index.html` real.

- [ ] `data_loader.py`: descarga `^GSPC` y `^VIX`, resamplea a semanal, retorna DataFrame limpio
- [ ] `feature_engineer.py`: calcula los 10 features, aplica `RobustScaler`, construye `X_train`, `y_train`, `X_pred`
- [ ] `trainer.py`: walk-forward CV con `TimeSeriesSplit(n_splits=5)`, entrena XGBoost y RF, selecciona ganador por RMSE, serializa en `model.joblib`, guarda `metrics.json`
- [ ] `predictor.py`: carga `model.joblib`, transforma `X_pred`, retorna `predicted_return_pct`
- [ ] Creación de `predictions.csv` con la primera fila de predicción
- [ ] `chart_generator.py`: genera `index.html` con velas + predicción usando Plotly dark theme
- [ ] `pipeline.py`: orquesta todos los módulos en orden correcto
- [ ] Prueba manual completa en Windows local

**Excluye**: automatización, publicación online

**Esfuerzo estimado**: 3–4 horas de implementación con Claude Code

---

### Fase 2 — Automatización y publicación
**Meta**: el pipeline corre automáticamente cada lunes y publica el resultado en una URL pública.

- [ ] `requirements.txt` con versiones fijas (`pip freeze > requirements.txt` tras prueba local)
- [ ] `.github/workflows/weekly.yml`: cron `0 10 * * 1`, setup Python 3.11, install deps, run `pipeline.py`, git commit + push
- [ ] Configurar GitHub Pages en el repositorio: branch `main`, root `/` (sirve `index.html`)
- [ ] Primer run manual del workflow vía `workflow_dispatch` para validar que funciona en el entorno de Actions
- [ ] Badge de GitHub Actions en `README.md`
- [ ] Paso de "relleno del actual": al inicio de cada run, actualizar `actual_return_pct` de la semana anterior en `predictions.csv`

**Esfuerzo estimado**: 1–2 horas

---

### Fase 3 — Pulido del portafolio
**Meta**: que el proyecto sea presentable y auditable para un reclutador.

- [ ] `README.md` completo: descripción del proyecto, metodología, screenshot del dashboard, badge de pipeline
- [ ] Comentarios docstring en cada módulo Python explicando el "por qué" de cada decisión técnica
- [ ] Panel de métricas en el dashboard (RMSE, MAE, Directional Accuracy comparando ambos modelos)
- [ ] Manejo de error en el workflow: mensaje claro en el log si yfinance falla
- [ ] Validar que el `index.html` se ve bien en móvil (Plotly es responsivo por defecto)

**Esfuerzo estimado**: 1 hora

---

## 10. Buenas prácticas aplicadas

### Series temporales y ML financiero
- **Sin data leakage**: `TimeSeriesSplit` garantiza que el modelo nunca ve datos futuros durante la validación. No usar `cross_val_score` estándar.
- **No-lookahead en features**: todos los indicadores técnicos de la semana `W` se calculan con `df.shift(1)` donde corresponde para asegurar que solo usan información disponible antes del cierre de `W`.
- **`RobustScaler` obligatorio**: los retornos financieros tienen outliers extremos (semanas de COVID, crisis). `StandardScaler` sería distorsionado por esos valores; `RobustScaler` usa mediana y IQR.
- **Ventana deslizante de 2 años**: el mercado de 2020 es estructuralmente diferente al de 2023. Entrenar sobre toda la historia diluiría patrones recientes relevantes.

### Reproducibilidad
- **`random_state=42`** en `XGBRegressor`, `RandomForestRegressor`, y `TimeSeriesSplit`. Cada run con los mismos datos produce exactamente el mismo modelo.
- **`requirements.txt` con versiones fijas**: `xgboost==2.0.3`, `scikit-learn==1.4.2`, etc. Evita que una actualización silenciosa de librerías rompa el pipeline.

### Calidad del código
- Cada módulo en `src/` tiene una sola responsabilidad (SRP). `pipeline.py` solo orquesta.
- Todos los parámetros configurables (períodos de RSI, ventana de entrenamiento, número de semanas en el gráfico) se definen como constantes en la parte superior de cada módulo, no hardcodeados dentro de las funciones.
- Los errores de descarga de yfinance se capturan con `try/except` y loguean con contexto antes de re-lanzar.

### Portafolio
- El `README.md` explica la metodología en lenguaje accesible: qué predice, cómo, y por qué las decisiones técnicas tomadas. Un reclutador no técnico entiende el valor; uno técnico puede auditar el código.
- `metrics.json` persiste en el repo: cualquier persona puede ver el desempeño histórico del modelo sin ejecutar nada.
- El badge de GitHub Actions en el README muestra en tiempo real que el proyecto está vivo y automatizado — una señal fuerte de madurez de ingeniería.
