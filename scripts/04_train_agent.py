"""
04_train_agent.py

Entrenamiento SAC con evaluaciones periódicas y guardado del mejor modelo
basado en rendimiento sobre validación.
"""

import os
import sys
import time
import numpy as np
import pandas as pd

# Asegurarnos de que src esté en el path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.portfolio_env import PortfolioEnv
from src.evaluation import evaluate_performance


# --- User parameters (edit these) ---
LOOKBACK = 60
REBALANCE_FREQ = 10
TRANSACTION_COST = 0
TOTAL_TIMESTEPS = 100000
EVAL_FREQ = 5000
TRAIN_START = "2006-01-01"
TRAIN_END = "2016-12-31"
VAL_START = "2017-01-01"
VAL_END = "2019-12-31"
MODEL = "SAC"

# Mapeo de nombre de modelo a la clase correspondiente de SB3
from stable_baselines3 import SAC, DDPG, TD3, PPO
MODEL_MAP = {"SAC": SAC, "DDPG": DDPG, "TD3": TD3, "PPO": PPO}
ModelClass = MODEL_MAP.get(MODEL)
if ModelClass is None:
    raise ValueError(f"Algoritmo desconocido: {MODEL}")

def load_data(path, start, end):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df[(df.index >= start) & (df.index <= end)]

def evaluate_agent(env, model):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
    returns = pd.Series(env.history["rewards"])
    turnovers = pd.Series(env.history["turnovers"])
    cumulative_returns = pd.Series(env.history["accumulated_reward"])
    return evaluate_performance(returns, turnovers, cumulative_returns)

def baseline_metrics(df_val):
    # Buy & Hold SPY
    spy_ret = df_val["SPY_LogReturn"]
    returns = spy_ret.reset_index(drop=True)
    turnovers = pd.Series(np.zeros(len(returns)))
    cumulative_returns = returns.cumsum()
    return evaluate_performance(returns, turnovers, cumulative_returns)

def main():

    # Directorios
    models_dir = "models"
    results_dir = "results/model_training"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Nombre de política y identificador del modelo
    policy = "MlpPolicy"
    model_id = f"{MODEL}_{policy}_lb{LOOKBACK}_reb{REBALANCE_FREQ}_tc{TRANSACTION_COST}"

    # Mostrar parámetros de configuración
    print("=== Parámetros de configuración ===")
    print(f"LOOKBACK: {LOOKBACK}")
    print(f"REBALANCE_FREQ: {REBALANCE_FREQ}")
    print(f"TRANSACTION_COST: {TRANSACTION_COST}")
    print(f"TRAIN period: {TRAIN_START} to {TRAIN_END}")
    print(f"VAL period: {VAL_START} to {VAL_END}")
    print(f"TOTAL_TIMESTEPS: {TOTAL_TIMESTEPS}")
    print(f"EVAL_FREQ: {EVAL_FREQ}")
    print("-------------------------------")

    # Carga y splits
    data_path = "datasets/combined_data.csv"
    df_train = load_data(data_path, TRAIN_START, TRAIN_END)
    df_val = load_data(data_path, VAL_START, VAL_END)

    # Mostrar baseline SPY Buy & Hold
    bh_metrics = baseline_metrics(df_val)

    print("=== Baseline SPY Buy & Hold ===")
    for metric, value in bh_metrics.items():
        print(f"{metric}: {value:.4f}")
    print("-------------------------------")

    # Mostrar validación inicial
    val_env = PortfolioEnv(df_val, LOOKBACK, REBALANCE_FREQ, TRANSACTION_COST, observation_mode="mlp")
    model = ModelClass(policy, val_env, verbose=0)
    init_metrics = evaluate_agent(val_env, model)

    print("=== Validación inicial (modelo sin entrenar) ===")
    for metric, value in init_metrics.items():
        print(f"{metric}: {value:.4f}")
    print("-------------------------------")
    records = []
    records.append({"step": 0, **init_metrics.to_dict()})

    # Entrenamiento con evaluaciones periódicas
    train_env = PortfolioEnv(df_train, LOOKBACK, REBALANCE_FREQ, TRANSACTION_COST, observation_mode="mlp")
    model = ModelClass(policy, train_env, verbose=0)

    best_return = -np.inf
    start_time = time.time()
    
    for step in range(EVAL_FREQ, TOTAL_TIMESTEPS + 1, EVAL_FREQ):
        model.learn(total_timesteps=EVAL_FREQ, reset_num_timesteps=False)
        metrics = evaluate_agent(val_env, model)
        record = {"step": step, **metrics.to_dict()}
        records.append(record)
        # Mostrar evaluación periódica
        print(f"=== Evaluación en step {step} ===")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        print("-------------------------------")
        # Guardar mejor modelo (excluyendo step 0)
        if record["cumulative_return"] > best_return:
            best_return = record["cumulative_return"]
            model.save(os.path.join(models_dir, f"{model_id}.zip"))

    elapsed = time.time() - start_time

    # Evaluación final
    final_metrics = evaluate_agent(val_env, model)
    records.append({"step": TOTAL_TIMESTEPS, **final_metrics.to_dict()})

    # Guardado de métricas
    df_records = pd.DataFrame(records)
    df_records.to_csv(os.path.join(results_dir, f"{model_id}_metrics.csv"), index=False)

    # Informar ubicaciones
    print(f"Training time (s): {elapsed:.1f}")

if __name__ == "__main__":
    main()