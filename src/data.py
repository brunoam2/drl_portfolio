import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import ta


def download_price_data(tickers, start, end):
    """Descargar precios de cierre ajustados desde Yahoo Finance."""
    df = yf.download(tickers, start=start, end=end)['Close']
    return df if isinstance(df, pd.DataFrame) else df.to_frame()


def compute_tech_indicators(price_df):
    """Calcular indicadores técnicos diarios por ticker."""
    features_list = []
    for ticker in price_df.columns:
        series = price_df[ticker]
        df_ind = pd.DataFrame(index=series.index)

        # Retornos logarítmicos
        logret = np.log(series / series.shift(1))
        df_ind[f"{ticker}_logret"] = logret

        # Diferencia porcentual SMA200
        sma200 = series.rolling(window=200, min_periods=200).mean()
        df_ind[f"{ticker}_RelSMA200"] = (series - sma200) / sma200

        # ROC10
        roc = (series / series.shift(10)) - 1
        df_ind[f"{ticker}_ROC10"] = roc

        # RSI14 normalizado
        rsi = ta.momentum.RSIIndicator(close=series, window=14).rsi() / 100.0
        df_ind[f"{ticker}_RSI14"] = rsi

        # VOL21 volatilidad histórica
        vol = logret.rolling(window=21, min_periods=21).std()
        df_ind[f"{ticker}_VOL21"] = vol

        features_list.append(df_ind)

    return pd.concat(features_list, axis=1)


def download_macro_data(api_key, start, end, index_ref):
    """Descargar datos macroeconómicos diarios: CPI, GDPC1, FEDFUNDS, VIX, DX."""
    fred = Fred(api_key=api_key)
    macro_df = pd.DataFrame(index=index_ref)

    cpi_m = fred.get_series('CPIAUCSL', observation_start=start, observation_end=end)
    cpi_logm = np.log(cpi_m / cpi_m.shift(1))
    cpi_logm = cpi_logm.reindex(index_ref).ffill()
    macro_df['CPIAUCSL_logm'] = cpi_logm

    gdp_q = fred.get_series('GDPC1', observation_start=start, observation_end=end)
    gdp_logq = np.log(gdp_q / gdp_q.shift(1))
    gdp_logq = gdp_logq.reindex(index_ref).ffill()
    macro_df['GDPC1_logq'] = gdp_logq

    fed = fred.get_series('FEDFUNDS', observation_start=start, observation_end=end)
    fed = fed.reindex(index_ref).ffill()
    macro_df['FEDFUNDS'] = fed

    vix = yf.download('^VIX', start=start, end=end)['Close']
    vix = vix.reindex(index_ref).ffill()
    macro_df['VIX'] = vix

    dx = yf.download('DX-Y.NYB', start=start, end=end)['Close']
    dx_log = np.log(dx / dx.shift(1))
    dx_log = dx_log.reindex(index_ref).ffill()
    macro_df['DX_log'] = dx_log

    return macro_df


def get_combined_data(tickers, start, end, api_key):
    """Combinar datos técnicos y macroeconómicos en un DataFrame limpio."""
    price_df = download_price_data(tickers, start, end)
    tech_df = compute_tech_indicators(price_df)
    macro_df = download_macro_data(api_key, start, end, price_df.index)
    combined = pd.concat([tech_df, macro_df], axis=1).dropna()
    return combined