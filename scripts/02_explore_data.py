# scripts/02_explore_data.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Rutas
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / 'datasets' / 'combined_data.csv'
OUT  = BASE / 'results' / 'explore_data'
OUT.mkdir(parents=True, exist_ok=True)

# Carga y sanity check
df = pd.read_csv(DATA, index_col=0, parse_dates=True)
print(f"Datos: {df.shape[0]} filas x {df.shape[1]} columnas")
assert not df.isna().any().any(), "Error: hay valores NaN en combined_data"

# 1) Stats de log-retornos
logret = df.filter(like='_logret')
stats = logret.describe().T[['mean','std']]
stats['skew'] = logret.skew()
stats.to_csv(OUT / '01_stats_logret.csv')

# 2) KDE conjunta de log-retornos
plt.figure(figsize=(8,4))
for c in logret.columns:
    sns.kdeplot(logret[c].dropna(), label=c.replace('_logret',''))
plt.title('02 KDE log-retornos diarios')
plt.xlabel('Log-retorno')
plt.legend()
plt.tight_layout()
plt.savefig(OUT / '02_kde_logret.png')
plt.close()

# 3) Curva acumulada conjunta
cum = np.exp(logret.cumsum())
plt.figure(figsize=(8,4))
for c in cum.columns:
    plt.plot(cum.index, cum[c], label=c.replace('_logret',''))
plt.title('03 Valor acumulado de activos')
plt.xlabel('Fecha')
plt.ylabel('Acumulado (base 1)')
plt.legend()
plt.tight_layout()
plt.savefig(OUT / '03_cum_returns.png')
plt.close()

# 4) Evolución acumulada de macros (log-retornos)
all_log_macro = ['CPIAUCSL_logm', 'GDPC1_logq', 'DX_log']
for col in all_log_macro:
    if col in df:
        plt.figure(figsize=(8,4))
        plt.plot(df.index, df[col].cumsum(), label=col)
        plt.title(f'04 {col} acumulado')
        plt.xlabel('Fecha')
        plt.ylabel('Acumulado log-retornos')
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT / f'04_macro_{col}_cum.png')
        plt.close()

# 5) Evolución niveles de macros
all_level_macro = ['FEDFUNDS', 'VIX']
for col in all_level_macro:
    if col in df:
        plt.figure(figsize=(8,4))
        plt.plot(df.index, df[col], label=col)
        plt.title(f'05 {col} nivel')
        plt.xlabel('Fecha')
        plt.ylabel('Nivel')
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT / f'05_macro_{col}_level.png')
        plt.close()

# 6) Correlación conjunta de log-retornos de activos y datos macroeconómicos
# Selección de columnas macroeconómicas definidas previamente
macro_cols = [col for col in all_log_macro + all_level_macro if col in df.columns]
# Concatenar log-retornos de activos y datos macroeconómicos
combined_df = pd.concat([logret, df[macro_cols]], axis=1)
# Cálculo de la matriz de correlación
corr_combined = combined_df.corr()
# Plot heatmap de correlaciones
plt.figure(figsize=(12,10))
sns.heatmap(corr_combined, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('06 Correlación conjunta activos y datos macroeconómicos')
plt.tight_layout()
plt.savefig(OUT / '06_corr_activos_macro.png')
plt.close()

print("Exploración mínima completada. Salidas en:", OUT)