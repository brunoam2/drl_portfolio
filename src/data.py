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

        # Retornos logarítmicos
        log_return = np.log(price_series / price_series.shift(1))
        indicator_df[f"{ticker}_LogReturn"] = log_return

        # Retornos lineales (aritméticos)
        arith_return = price_series.pct_change()
        indicator_df[f"{ticker}_Return"] = arith_return

        # Diferencia porcentual SMA200
        sma_200 = price_series.rolling(window=200, min_periods=200).mean()
        indicator_df[f"{ticker}_RelativeSMA200"] = (price_series - sma_200) / sma_200

        # RSI14 normalizado
        rsi_14 = ta.momentum.RSIIndicator(close=price_series, window=14).rsi() / 100.0
        indicator_df[f"{ticker}_RSI14"] = rsi_14

        # VOL21 volatilidad histórica
        volatility_21 = log_return.rolling(window=21, min_periods=21).std()
        indicator_df[f"{ticker}_Volatility21"] = volatility_21

        indicator_dfs.append(indicator_df)

    return pd.concat(indicator_dfs, axis=1)


def download_macroeconomic_data(fred_api_key, start_date, end_date, reference_index):
    """Descargar datos macroeconómicos diarios (raw, aritméticos y logarítmicos): CPI, GDPC1, FEDFUNDS, VIX, DX."""
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
        # Guardar valor raw
        raw = original_series.reindex(reference_index).ffill()
        macro_df[f"{column_name}_Raw"] = raw

        # Retorno aritmético diario, con 0 explícito para días sin cambio o sin datos previos
        arith = raw.pct_change().fillna(0)
        macro_df[f"{column_name}_Return"] = arith

        # Retorno logarítmico diario, también rellenando NaN con 0
        log = np.log1p(arith).fillna(0)
        macro_df[f"{column_name}_LogReturn"] = log
        continue

    return macro_df


def get_combined_dataset(tickers, start_date, end_date, fred_api_key):
    """Combinar datos técnicos y macroeconómicos en un DataFrame limpio."""
    price_df = download_price_data(tickers, start_date, end_date)
    tech_df = compute_technical_indicators(price_df)
    macro_df = download_macroeconomic_data(fred_api_key, start_date, end_date, price_df.index)
    combined_df = pd.concat([price_df, tech_df, macro_df], axis=1).dropna()
    return combined_df