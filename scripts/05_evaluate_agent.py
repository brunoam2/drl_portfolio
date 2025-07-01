import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Asegurarnos de que src esté en el path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.train import evaluate_agent, get_model_class, parse_model_name
from src.portfolio_env import PortfolioEnv
from src.data import load_price_data
from src.metrics import metrics
from src.benchmarks import run_benchmarks

# Variables necesarias (ejemplo o placeholders)
model_id = "SAC_lb30_reb1_tc0"
model_path = f"models/{model_id}.zip"
test_start = "2022-01-01"
test_end = "2024-12-31"
train_start = "2006-01-01"
train_end = "2021-12-31"

# 1. Carga de datos
df = load_price_data("datasets/combined_data.csv", start_date="2006-01-01", end_date="2024-12-31")
df.index = pd.to_datetime(df.index)  # Ensure datetime index for slicing
df_train = df.loc[train_start:train_end].reset_index(drop=False)
df_test = df.loc[test_start:test_end].reset_index(drop=False)

# 2. Crear entorno de evaluación (adaptado de train.py)
model_name, lookback, rebalance_freq, transaction_cost = parse_model_name(model_id)
test_env = PortfolioEnv(df_test, lookback, rebalance_freq, transaction_cost)

# 3. Cargar el modelo entrenado
model_class = get_model_class(model_name)
model = model_class.load(model_path, env=test_env)

# 4. Evaluación agente
eval_metrics = evaluate_agent(model, test_env)
eval_metrics.name = model_id

# 5. Benchmarks: SPY, MM‑Varianza, Min‑Var, Max‑Diversif, Min‑Correl, Risk‑Parity
bench_metrics = run_benchmarks(df_train, df_test)

# 6. Guardar métricas en CSV
all_metrics = pd.concat([pd.DataFrame([eval_metrics.to_dict()]), bench_metrics.drop(columns=['curves', 'weights'], errors='ignore')], axis=0)
# Métricas combinadas listas para guardar
out_dir = "results/model_evaluation"
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, f"{model_id}_evaluation.csv")
all_metrics.to_csv(csv_path, index=True, float_format="%.6f")

# 7. Visualización: evolución del valor de portfolio
plt.figure(figsize=(10, 6))
# agente
wealth_agent = np.exp(np.array(test_env.history["accumulated_rewards"]))
agent_dates = df_test["Date"].iloc[-len(test_env.history["accumulated_rewards"]):].reset_index(drop=True)
plt.plot(agent_dates, wealth_agent, label=model_id)
# benchmarks desde run_benchmarks, suponemos que también devuelve curvas
for name, curve in bench_metrics.curves.items():
    curve_trimmed = curve.reset_index(drop=True)
    if len(curve_trimmed) > len(agent_dates):
        curve_trimmed = curve_trimmed[-len(agent_dates):].reset_index(drop=True)
        agent_dates_trimmed = agent_dates.reset_index(drop=True)
    elif len(curve_trimmed) < len(agent_dates):
        agent_dates_trimmed = agent_dates[-len(curve_trimmed):].reset_index(drop=True)
    else:
        agent_dates_trimmed = agent_dates.reset_index(drop=True)
    plt.plot(agent_dates_trimmed, curve_trimmed, label=name)
plt.legend()
plt.title("Evolución del valor de cartera vs benchmarks")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"{model_id}_wealth_comparison.png"))
plt.close()

# 8. Visualización: reparto de pesos del agente
plt.figure(figsize=(8, 6))
weights = np.vstack(test_env.history["weights"])
mean_weights = weights.mean(axis=0)
assets = test_env.asset_names
plt.bar(assets, mean_weights)
plt.title("Peso medio asignado por el agente")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"{model_id}_avg_weights.png"))
plt.close()

# 9. Visualización: evolución de los pesos en el tiempo
weights_array = np.vstack(test_env.history["weights"])
plt.figure(figsize=(10, 6))
for i, asset in enumerate(test_env.asset_names):
    plt.plot(agent_dates, weights_array[:, i], label=asset)
plt.title("Evolución temporal de los pesos del portafolio")
plt.xlabel("Fecha")
plt.ylabel("Peso")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"{model_id}_weights_evolution.png"))
plt.close()

# 10. Visualización: evolución de los pesos medios acumulados
cumulative_mean_weights = np.cumsum(weights_array, axis=0) / np.arange(1, len(weights_array) + 1).reshape(-1, 1)
plt.figure(figsize=(10, 6))
for i, asset in enumerate(test_env.asset_names):
    plt.plot(agent_dates, cumulative_mean_weights[:, i], label=asset)
plt.title("Evolución de los pesos medios acumulados")
plt.xlabel("Fecha")
plt.ylabel("Peso medio acumulado")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"{model_id}_cumulative_avg_weights.png"))
plt.close()

print("Evaluación completada.")