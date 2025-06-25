from pathlib import Path
from datetime import timedelta
import sys
import pandas as pd

# Se añade el project root al path para que `import src` funcione
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import get_combined_dataset

# --- Configuración de usuario ---
TICKERS      = ["SPY", "TLT", "GLD"]
DATA_START   = "2006-01-01"
DATA_END     = "2024-12-31"
FRED_API_KEY = "tu_clave_fred_api"  # Reemplazar con tu clave de API de FRED

# Directorio de salida
OUT_DIR = Path(__file__).parents[1] / "datasets"

def main():
    """Descarga y almacena precios y dataset combinado."""
    # Crear directorio de salida
    OUT_DIR.mkdir(exist_ok=True)

    # Fecha de descarga anticipada para indicadores lookback
    download_start = pd.to_datetime(DATA_START) - timedelta(days=300)

    # Descarga de datos
    combined_df = get_combined_dataset(TICKERS, download_start, DATA_END, FRED_API_KEY)

    # Filtrado al rango definido
    combined_df = combined_df.loc[DATA_START:]

    # Guardado de ficheros
    combined_df.to_csv(OUT_DIR / "combined_data.csv")

    # Informe de resultados
    print(f"Datos guardados en {OUT_DIR} — filas: {len(combined_df)}, columnas: {combined_df.shape[1]}")

if __name__ == "__main__":
    main()