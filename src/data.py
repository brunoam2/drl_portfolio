import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import ta


def download_price_data(tickers, start_date, end_date):
    """Descargar precios de cierre ajustados desde Yahoo Finance."""
    df = yf.download(tickers, start=start_date, end=end_date)['Close']
    return df if isinstance(df, pd.DataFrame) else df.to_frame()


def compute_technical_indicators(price_df):
    """Calcular indicadores técnicos diarios por ticker."""
    indicator_dfs = []
    for ticker in price_df.columns:
        price_series = price_df[ticker]
        indicator_df = pd.DataFrame(index=price_series.index)

        # Retornos aritméticos
        returns = price_series.pct_change().fillna(0)
        indicator_df[f"{ticker}_Return"] = returns

        # Retornos logarítmicos
        log_return = np.log(price_series / price_series.shift(1))
        indicator_df[f"{ticker}_LogReturn"] = log_return

        # Diferencia porcentual SMA50
        sma_50 = price_series.rolling(window=50, min_periods=50).mean()
        indicator_df[f"{ticker}_RelativeSMA50"] = (price_series - sma_50) / sma_50

        # Diferencia porcentual SMA21
        sma_21 = price_series.rolling(window=21, min_periods=21).mean()
        indicator_df[f"{ticker}_RelativeSMA21"] = (price_series - sma_21) / sma_21

        # RSI14
        rsi_14 = ta.momentum.RSIIndicator(close=price_series, window=14).rsi()
        indicator_df[f"{ticker}_RSI14"] = rsi_14

        # VOL21 volatilidad histórica
        volatility_21 = log_return.rolling(window=21, min_periods=21).std()
        indicator_df[f"{ticker}_Volatility21"] = volatility_21

        indicator_dfs.append(indicator_df)

    return pd.concat(indicator_dfs, axis=1)


def download_macroeconomic_data(fred_api_key, start_date, end_date, reference_index):
    """Descargar datos macroeconómicos diarios: valores originales y versiones forward-filled de CPI, FEDFUNDS y DX."""
    fred_client = Fred(api_key=fred_api_key)
    macro_df = pd.DataFrame(index=reference_index)

    # Definición de series macroeconómicas: (símbolo, nombre de columna)
    macro_series = [
        ('CPIAUCSL', 'CPIAUCSL'),
        ('FEDFUNDS', 'FederalFundsRate'),
        ('DTWEXAFEGS', 'DX'),
    ]

    for symbol, column_name in macro_series:
        original_series = fred_client.get_series(symbol, observation_start=start_date, observation_end=end_date)
                # Guardar solo la versión forward-filled
        raw = original_series.reindex(reference_index)
        macro_df[f"{column_name}_Filled"] = raw.ffill()
    return macro_df


def get_combined_dataset(tickers, start_date, end_date, fred_api_key):
    """Combinar datos técnicos y macroeconómicos en un DataFrame limpio."""
    price_df = download_price_data(tickers, start_date, end_date)
    tech_df = compute_technical_indicators(price_df)
    macro_df = download_macroeconomic_data(fred_api_key, start_date, end_date, price_df.index)
    combined_df = pd.concat([price_df, tech_df, macro_df], axis=1).dropna()
    return combined_df


def load_price_data(filepath, start_date, end_date):
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df[(df.index >= start_date) & (df.index <= end_date)]