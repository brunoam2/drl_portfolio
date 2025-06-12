from pathlib import Path
from datetime import timedelta
import sys
import pandas as pd

# Añadimos el project root al path para que `import src` funcione
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import download_price_data, get_combined_data

# --- Configuración de usuario ---
TICKERS      = ["SPY", "TLT", "GLD"]
DATA_START   = "2006-01-01"
DATA_END     = "2024-12-31"
FRED_API_KEY = "14f7d5dc732ec05d73eab8ab21852b5a"

# Directorio de salida
OUT_DIR = Path(__file__).parents[1] / "datasets"
OUT_DIR.mkdir(exist_ok=True)

# Calculamos fecha de descarga anticipada para indicadores lookback
download_start = pd.to_datetime(DATA_START) - timedelta(days=300)

# Descarga de precios y macro + cálculo de indicadores
raw_df = download_price_data(TICKERS, download_start, DATA_END)
combined_df = get_combined_data(TICKERS, download_start, DATA_END, FRED_API_KEY)

# Filtramos al rango definido por el usuario
raw_df = raw_df.loc[DATA_START:]
combined_df = combined_df.loc[DATA_START:]

# Guardamos los ficheros
raw_df.to_csv(OUT_DIR / "raw_data.csv")
combined_df.to_csv(OUT_DIR / "combined_data.csv")

print(
    f"Datos guardados en {OUT_DIR} — "
    f"filas: {len(combined_df)}, columnas: {combined_df.shape[1]}"
)