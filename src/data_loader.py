import time
import yfinance as yf
import pandas as pd

GSPC_TICKER = "^GSPC"
VIX_TICKER = "^VIX"
DOWNLOAD_PERIOD = "5y"
MAX_RETRIES = 3
RETRY_DELAY = 30


def _download_with_retry(ticker: str, period: str) -> pd.DataFrame:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[data_loader] Descargando {ticker} (intento {attempt}/{MAX_RETRIES})...")
            df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            if df.empty:
                raise ValueError(f"yfinance retornó DataFrame vacío para {ticker}")
            print(f"[data_loader] {ticker}: {len(df)} filas descargadas.")
            return df
        except Exception as e:
            print(f"[data_loader] ERROR en intento {attempt}: {e}")
            if attempt < MAX_RETRIES:
                print(f"[data_loader] Reintentando en {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                raise


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Descarga datos diarios de ^GSPC y ^VIX y los resamplea a frecuencia semanal (W-FRI).
    Retorna (gspc_daily, gspc_weekly, vix_weekly).
    """
    gspc_daily = _download_with_retry(GSPC_TICKER, DOWNLOAD_PERIOD)
    vix_daily = _download_with_retry(VIX_TICKER, DOWNLOAD_PERIOD)

    # Aplanar columnas MultiIndex si yfinance las devuelve así
    if isinstance(gspc_daily.columns, pd.MultiIndex):
        gspc_daily.columns = gspc_daily.columns.get_level_values(0)
    if isinstance(vix_daily.columns, pd.MultiIndex):
        vix_daily.columns = vix_daily.columns.get_level_values(0)

    gspc_weekly = gspc_daily.resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna()

    vix_weekly = vix_daily["Close"].resample("W-FRI").last().dropna()
    vix_weekly.name = "vix_weekly"

    print(f"[data_loader] GSPC diario: {len(gspc_daily)} días.")
    print(f"[data_loader] GSPC semanal: {len(gspc_weekly)} semanas.")
    print(f"[data_loader] VIX semanal: {len(vix_weekly)} semanas.")

    return gspc_daily, gspc_weekly, vix_weekly
