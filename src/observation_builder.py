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
        "SPY_LogReturn", "SPY_RelativeSMA200", "SPY_RSI14", "SPY_Volatility21",
        "GLD_LogReturn", "GLD_RelativeSMA200", "GLD_RSI14", "GLD_Volatility21",
        "TLT_LogReturn", "TLT_RelativeSMA200", "TLT_RSI14", "TLT_Volatility21",
        "DX_LogReturn", "CPIAUCSL_LogReturn", "FederalFundsRate_LogReturn"
    ]

    normed = {}
    for col in cols:
        normed[col] = zscore_normalize(window[col])

    obs = np.concatenate([normed[col].values for col in cols])

    if current_weights is not None:
        obs = np.concatenate([obs, current_weights])

    return obs

def build_observation_cnn(combined_data, step, lookback):
    return 

def build_observation_rnn(combined_data, step, lookback, current_weights=None):
    """Construye la observación para una red RNN con características normalizadas y pesos opcionales."""
    start = step - (lookback - 1)
    window = combined_data.iloc[start:step+1]

    cols = [
        "SPY_LogReturn", "SPY_RelativeSMA200", "SPY_RSI14", "SPY_Volatility21",
        "GLD_LogReturn", "GLD_RelativeSMA200", "GLD_RSI14", "GLD_Volatility21",
        "TLT_LogReturn", "TLT_RelativeSMA200", "TLT_RSI14", "TLT_Volatility21",
        "DX_LogReturn", "CPIAUCSL_LogReturn", "FederalFundsRate_LogReturn"
    ]

    normed = []
    for col in cols:
        normed.append(zscore_normalize(window[col]).values)

    obs = np.stack(normed, axis=1)  # (lookback, num_features)

    if current_weights is not None:
        weights_channel = np.tile(current_weights, (lookback, 1))
        obs = np.concatenate([obs, weights_channel], axis=1)

    return obs