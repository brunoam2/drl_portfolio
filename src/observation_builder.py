# src/observation_builder.py

import numpy as np
import pandas as pd

def _zscore_norm(data: np.ndarray) -> np.ndarray:
    """
    Normaliza z-score por columna, manejando NaNs.
    """
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0) + 1e-8
    return (data - mean) / std


def _build_asset_block(combined_data: pd.DataFrame, dates: pd.DatetimeIndex, asset: str) -> np.ndarray:
    """
    Construye y normaliza el bloque de datos para un activo en las fechas dadas:
    - Para CASH devuelve un array de ceros de forma (len(dates), 1).
    - Para otros activos, filtra columnas que empiecen por "{asset}_", convierte a float32 y normaliza por z-score.
    """
    if asset == "CASH":
        return np.zeros((len(dates), 1), dtype=np.float32)
    prefix = f"{asset}_"
    cols = [c for c in combined_data.columns if c.startswith(prefix)]
    if not cols:
        raise KeyError(f"No se encontraron columnas para el activo {asset}")
    window = combined_data.loc[dates, cols].to_numpy(dtype=np.float32)
    return _zscore_norm(window)


def build_observation_mlp(combined_data: pd.DataFrame, step: int, lookback: int, asset_names: list, macro_cols: list) -> np.ndarray:
    """
    Observación para MLP (vector 1D):
    - Últimos `lookback` días de datos.
    - Columnas macro normalizadas.
    - Datos de cada activo normalizados (incluido CASH).
    - macro_cols: lista de nombres de columnas macroeconómicas a utilizar.
    Devuelve un vector 1D de longitud lookback * (n_macro + n_activos).
    """
    dates = combined_data.index[step - lookback + 1 : step + 1]
    # Validar indicadores macro
    missing_macro = [col for col in macro_cols if col not in combined_data.columns]
    if missing_macro:
        raise KeyError(f"Faltan columnas macro: {missing_macro}")
    # Normalizar datos macro
    macro_window = combined_data.loc[dates, macro_cols].to_numpy(dtype=np.float32)
    macro_norm = _zscore_norm(macro_window)
    # Construir bloques: macro + activos
    blocks = [macro_norm] + [
        _build_asset_block(combined_data, dates, asset) for asset in asset_names
    ]
    obs = np.concatenate(blocks, axis=1).flatten().astype(np.float32)
    return obs


def build_observation_cnn(combined_data: pd.DataFrame, step: int, lookback: int, asset_names: list, macro_cols: list) -> np.ndarray:
    """
    Observación para CNN (tensor 3D):
    - Para cada activo, concatena sus datos normalizados con los indicadores macro.
    - Normaliza z-score cada bloque.
    - macro_cols: lista de nombres de columnas macroeconómicas a utilizar.
    Devuelve un array de forma (n_activos, lookback, n_feats), donde n_feats = n_asset_features + n_macro_features.
    """
    dates = combined_data.index[step - lookback + 1 : step + 1]
    # Validar indicadores macro
    missing_macro = [col for col in macro_cols if col not in combined_data.columns]
    if missing_macro:
        raise KeyError(f"Faltan columnas macro: {missing_macro}")
    # Normalizar datos macro
    macro_window = combined_data.loc[dates, macro_cols].to_numpy(dtype=np.float32)
    macro_norm = _zscore_norm(macro_window)
    # Construir tensor para cada activo: bloque activo + macro
    tensors = [
        np.concatenate([_build_asset_block(combined_data, dates, asset), macro_norm], axis=1).astype(np.float32)
        for asset in asset_names
    ]
    return np.stack(tensors, axis=0)