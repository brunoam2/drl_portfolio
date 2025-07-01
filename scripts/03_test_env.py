import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Determinar ruta raíz del proyecto y asegurar que src está en el PATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.portfolio_env import PortfolioEnv

# Cargar datos combinados desde datasets
combined_csv_path = os.path.join(PROJECT_ROOT, "datasets", "combined_data.csv")
combined_data = pd.read_csv(combined_csv_path, index_col=0, parse_dates=True)

# Definir políticas baseline
baseline_policies = {
    "equal_weights": np.array([1/4, 1/4, 1/4, 1/4]),
    "spy_only": np.array([1.0, 0.0, 0.0, 0.0]),
}

# Crear directorio results si no existe
results_dir = os.path.join(PROJECT_ROOT, "results")
os.makedirs(results_dir, exist_ok=True)

# Crear entorno para todo el rango de datos
env = PortfolioEnv(combined_data=combined_data, lookback=60, rebalance_freq=10, transaction_cost=0.001)

print("Probando el entorno...")

portfolio_curves = {}
for name, weights in baseline_policies.items():
    obs, _ = env.reset()
    done = False
    cumulative_rewards = []
    dates = []

    while not done:
        obs, reward, terminated, truncated, info = env.step(weights)
        cumulative_rewards.append(info["accumulated_rewards"])
        dates.append(env.history["dates"][-1])
        done = terminated or truncated

    portfolio_values = np.exp(cumulative_rewards)
    portfolio_curves[name] = portfolio_values

# Mostrar todos los resultados en un solo gráfico
plt.figure(figsize=(10, 6))
for name, values in portfolio_curves.items():
    plt.plot(dates, values, label=name)
plt.title("Valor del portafolio - estrategias baseline")
plt.xlabel("Fecha")
plt.ylabel("Valor del portafolio")
plt.legend()
plt.grid(True)
plt.show()

print("Prueba finalizada.")