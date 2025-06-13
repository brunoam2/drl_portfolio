

import numpy as np

def build_observation_mlp(combined_data, step, lookback, asset_names=None):
    """
    Construye la observación para una red MLP usando:
    - SPY_LogReturn
    - TLT_LogReturn
    - GLD_LogReturn
    - SPY_RelativeSMA200
    - DX_Return
    La salida es un array NumPy de forma (5, lookback).
    """
    # Índice de inicio de la ventana
    start = step - (lookback - 1)
    # Selección de la ventana de datos
    window = combined_data.iloc[start:step+1]

    # Lista de columnas a usar
    cols = [
        "SPY_LogReturn",
        "TLT_LogReturn",
        "GLD_LogReturn",
        "SPY_RelativeSMA200",
        "DX_Return"
    ]

    # Normalización z-score por columna (ventana)
    normed = {}
    for col in cols:
        series = window[col]
        mu = series.mean()
        sigma = series.std()
        # Evitar división por cero
        normed[col] = (series - mu) / (sigma if sigma != 0 else 1.0)

    # Concatenar todas las columnas normalizadas en un vector 1D
    obs = np.concatenate([normed[col].values for col in cols])

    return obs

def build_observation_cnn(combined_data, step, lookback, asset_names=None):
    return 