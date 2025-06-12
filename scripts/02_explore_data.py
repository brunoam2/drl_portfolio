# scripts/02_explore_data.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Columnas a procesar
ASSET_RETURNS_COLS = ['SPY_LogReturn', 'TLT_LogReturn', 'GLD_LogReturn']
MACRO_COLS = ['CPIAUCSL_LogReturnMonthly', 'FederalFundsRate', 'DX_LogReturn']

# Rutas de entrada y salida
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / 'datasets' / 'combined_data.csv'
OUT  = BASE / 'results' / 'explore_data'
OUT.mkdir(parents=True, exist_ok=True)

def plot_and_save(series, title, filename, ylabel, kind='line'):
    """Helper to plot a pandas Series or DataFrame and save the figure."""
    ax = series.plot(kind=kind, figsize=(8,4), title=title)
    if kind != 'kde':
        ax.set_xlabel('Fecha')
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(OUT / filename)
    plt.close()

# Carga y sanity check
df = pd.read_csv(DATA, index_col=0, parse_dates=True)
print(f"Datos: {df.shape[0]} filas x {df.shape[1]} columnas")
assert not df.isna().any().any(), "Error: hay valores NaN en combined_data"

# 1) Stats de log-retornos de activos
asset_logret = df[ASSET_RETURNS_COLS]
stats = asset_logret.describe().T[['mean','std']]
stats['skew'] = asset_logret.skew()
stats.to_csv(OUT / '01_stats_logreturn.csv')

# 2) KDE conjunta de log-retornos de activos
plot_and_save(asset_logret, '02 KDE log-retornos diarios', '02_kde_logret.png', 'Densidad', kind='kde')

# 3) Curva acumulada conjunta de activos
plot_and_save(np.exp(asset_logret.cumsum()), '03 Valor acumulado de activos', '03_cum_returns.png', 'Acumulado (base 1)')

# 4) Gráficas individuales de datos macro
for col in MACRO_COLS:
    if col == 'DX_LogReturn':
        plot_and_save(np.exp(df[col].cumsum()), f'04 {col} acumulado', f'04_macro_{col}_cum.png', 'Acumulado (base 1)')
    else:
        plot_and_save(df[col], f'04 {col}', f'04_macro_{col}.png', 'Valor')

# 5) Correlación conjunta de retornos de activos y datos macroeconómicos
corr_df = pd.concat([asset_logret, df[MACRO_COLS]], axis=1).corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('05 Correlación activos vs macroeconómicos')
plt.tight_layout()
plt.savefig(OUT / '05_corr_activos_macro.png')
plt.close()

print("Exploración mínima completada. Salidas en:", OUT)