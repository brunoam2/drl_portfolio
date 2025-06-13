import os
import sys
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Añadimos el project root al path para que `import src` funcione
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

INPUT_DIR = "datasets"
OUTPUT_DIR = "results/explore_data"

data = pd.read_csv(os.path.join(INPUT_DIR, "combined_data.csv"), index_col=0, parse_dates=True)

numeric_data = data.select_dtypes(include=["number"])

# Variables de activos
asset_vars = [c for c in numeric_data.columns if any(c.startswith(t) for t in ["SPY", "TLT", "GLD"])]

# Subgrupos técnicos
logret_vars  = [c for c in asset_vars if c.endswith("_LogReturn")]
arithret_vars= [c for c in asset_vars if c.endswith("_Return")]
sma_vars     = [c for c in asset_vars if c.endswith("_RelSMA200")]
rsi_vars     = [c for c in asset_vars if c.endswith("_RSI14")]
vol_vars     = [c for c in asset_vars if c.endswith("_Vol21")]

#
# Definir variables log-retornos de activos
logret_vars = [
    c for c in numeric_data.columns
    if c.endswith("_LogReturn") and c.split("_")[0] in ("SPY", "TLT", "GLD")
]

# Definir variables macro (excluye todo lo que comience con ticker de activos o sea nombre exacto de ticker)
macro_vars = [
    c for c in numeric_data.columns
    if c not in logret_vars
    and not any(c == pref or c.startswith(pref + "_") for pref in ("SPY", "TLT", "GLD"))
]

# Estadísticas descriptivas para todas las variables numéricas
numeric_data.describe().to_csv(os.path.join(OUTPUT_DIR, "describe_all.csv"))

# Matrices de correlación y gráficos

# Matriz de correlación global de todas las variables numéricas
corr_all = numeric_data.corr()
corr_all.to_csv(os.path.join(OUTPUT_DIR, "corr_all.csv"))
plt.figure(figsize=(12,10))
sns.heatmap(corr_all, annot=False, cmap="coolwarm")
plt.title("Correlación global de variables")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "corr_all.png"))
plt.close()

# Correlación cruzada entre la media de LogReturns de activos y variables macro
logret_vars = [
    c for c in numeric_data.columns
    if c.endswith("_LogReturn") and c.split("_")[0] in ("SPY", "TLT", "GLD")
]
macro_vars = [
    c for c in numeric_data.columns
    if c not in logret_vars and not any(c.startswith(pref + "_") for pref in ("SPY", "TLT", "GLD"))
]

asset_mean = numeric_data[logret_vars].mean(axis=1)
cross_corr = numeric_data[macro_vars].apply(lambda col: asset_mean.corr(col))
cross_corr.to_csv(os.path.join(OUTPUT_DIR, "corr_asset_macro.csv"))

plt.figure(figsize=(10,6))
cross_corr.plot(kind='bar')
plt.title("Correlación Cruzada entre media de LogReturns de activos y Variables Macro")
plt.ylabel("Correlación")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "corr_asset_macro.png"))
plt.close()
