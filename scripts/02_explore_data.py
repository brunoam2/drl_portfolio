import os
os.makedirs("results/explore_data", exist_ok=True)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import matplotlib.dates as mdates

mpl.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "gray",
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "grid.color": "gray",
    "grid.linestyle": "--",
    "grid.alpha": 0.6
})

df = pd.read_csv("datasets/combined_data.csv", index_col=0, parse_dates=True)
assets = ['SPY', 'TLT', 'GLD']

df[assets].plot(figsize=(12, 6), grid=True, linewidth=1.5)
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=30)
plt.title("Evolución de precios de cierre")
plt.ylabel("Precio")
plt.xlabel("Fecha")
# plt.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.legend(loc='upper left', frameon=True)
plt.tight_layout()
plt.savefig("results/explore_data/price_evolution.png")
plt.close()

arith_returns = df[assets].pct_change().fillna(0)
(1 + arith_returns).cumprod().plot(figsize=(12, 6), grid=True, linewidth=1.5)
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=30)
plt.title("Retornos aritméticos acumulados")
plt.ylabel("Crecimiento acumulado")
plt.xlabel("Fecha")
# plt.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.legend(loc='upper left', frameon=True)
plt.tight_layout()
plt.savefig("results/explore_data/arith_returns_cumprod.png")
plt.close()

log_returns = np.log(df[assets] / df[assets].shift(1)).fillna(0)
log_returns.cumsum().plot(figsize=(12, 6), grid=True, linewidth=1.5)
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=30)
plt.title("Retornos logarítmicos acumulados")
plt.ylabel("Log-rendimiento acumulado")
plt.xlabel("Fecha")
# plt.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.legend(loc='upper left', frameon=True)
plt.tight_layout()
plt.savefig("results/explore_data/log_returns_cumsum.png")
plt.close()




# Gráficos individuales de indicadores macroeconómicos
plt.figure(figsize=(6, 5))
plt.plot(df.index, df["CPIAUCSL_Filled"], label="CPI (Filled)", color="tab:blue")
plt.title("Evolución del CPI")
plt.xlabel("Fecha")
plt.ylabel("Índice de precios")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("results/explore_data/cpi_evolucion.png", dpi=300)
plt.close()

plt.figure(figsize=(6, 5))
plt.plot(df.index, df["FederalFundsRate_Filled"], label="Tipo de interés (Fed Funds)", color="tab:orange")
plt.title("Evolución del tipo de interés de la Fed")
plt.xlabel("Fecha")
plt.ylabel("Tasa (%)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("results/explore_data/fedfunds_evolucion.png", dpi=300)
plt.close()

plt.figure(figsize=(6, 5))
plt.plot(df.index, df["DX_Filled"], label="Índice Dólar (DX)", color="tab:green")
plt.title("Evolución del índice del dólar (DX)")
plt.xlabel("Fecha")
plt.ylabel("Índice")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("results/explore_data/dx_evolucion.png", dpi=300)
plt.close()