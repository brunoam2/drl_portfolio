import numpy as np

def zscore_normalize(series):
    """Normaliza una serie usando z-score."""
    mu = series.mean()
    sigma = series.std()
    return (series - mu) / (sigma if sigma != 0 else 1.0)

def build_observation_mlp(combined_data, step, lookback, current_weights=None):
    """Construye la observación para una red MLP con características normalizadas y pesos opcionales."""
    start = step - (lookback - 1)
    window = combined_data.iloc[start:step+1]

    cols = [
        "SPY", "SPY_RelativeSMA50", "SPY_RelativeSMA21", "SPY_RSI14", "SPY_Volatility21",
        "GLD", "GLD_RelativeSMA50", "GLD_RelativeSMA21", "GLD_RSI14", "GLD_Volatility21",
        "TLT", "TLT_RelativeSMA50", "TLT_RelativeSMA21", "TLT_RSI14", "TLT_Volatility21",
        "DX_Filled", "CPIAUCSL_Filled", "FederalFundsRate_Filled"
    ]

    normed_matrix = np.stack([zscore_normalize(window[col]).values for col in cols], axis=1)
    obs = normed_matrix.flatten()
    if current_weights is not None:
        obs = np.concatenate([obs, current_weights])
    return obs