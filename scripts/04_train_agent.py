#!/usr/bin/env python3
"""
04_train_agent.py

Entrenamiento SAC, evaluación periódica y registro de métricas.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from stable_baselines3 import SAC

# Ajustar ruta para importar módulos desde src
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from stable_baselines3.sac.policies import MlpPolicy
from src.portfolio_env import PortfolioEnv
from src.evaluation import evaluate_agent, buy_and_hold_metrics

# ------------------ Parámetros de usuario ------------------
LOOKBACK = 60
REBALANCE_FREQ = 10
TRANSACTION_COST = 0
TOTAL_TIMESTEPS = 100000
STEP_EVAL = 5000

TRAIN_START = "2006-01-03"
TRAIN_END   = "2017-12-29"
VAL_START   = "2018-01-02"
VAL_END     = "2020-12-31"
# -----------------------------------------------------------

def load_data(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    train = df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)]
    val   = df[(df.index >= VAL_START)   & (df.index <= VAL_END)]
    return train, val

def main():
    # Mostrar parámetros de entrenamiento
    print("=== Parámetros de entrenamiento ===")
    print(f"LOOKBACK: {LOOKBACK}")
    print(f"REBALANCE_FREQ: {REBALANCE_FREQ}")
    print(f"TRANSACTION_COST: {TRANSACTION_COST}")
    print(f"TOTAL_TIMESTEPS: {TOTAL_TIMESTEPS}")
    print(f"STEP_EVAL: {STEP_EVAL}")
    print(f"TRAIN period: {TRAIN_START} to {TRAIN_END}")
    print(f"VAL period: {VAL_START} to {VAL_END}")
    print("==============================")

    # Directorio para modelos y métricas
    model_dir = os.path.join(PROJECT_ROOT, "results", "model_training")
    os.makedirs(model_dir, exist_ok=True)
    # Nombre base para archivos
    file_base = f"SAC_mlp_{LOOKBACK}_{REBALANCE_FREQ}_{TRANSACTION_COST}"
    best_cum_return = -np.inf

    data_path = os.path.join(PROJECT_ROOT, "datasets", "combined_data.csv")
    df_train, df_val = load_data(data_path)

    # Crear entornos
    train_env = PortfolioEnv(df_train, LOOKBACK, REBALANCE_FREQ, TRANSACTION_COST,
                             seed=None, observation_mode="mlp")
    val_env   = PortfolioEnv(df_val,   LOOKBACK, REBALANCE_FREQ, TRANSACTION_COST,
                             seed=None, observation_mode="mlp")

    # Instanciar SAC con MlpPolicy
    model = SAC(MlpPolicy, train_env, verbose=0)

    metrics = []

    # Evaluación inicial (paso 0)
    cum_ret, sharpe, sortino, mdd, turnover = evaluate_agent(val_env, model)
    metrics.append({
        "step": 0,
        "cum_return": cum_ret,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "turnover": turnover
    })
    print(f"Step 0 -- cum_return: {cum_ret:.4f}, sharpe: {sharpe:.3f}, sortino: {sortino:.3f}, mdd: {mdd:.3f}, turnover: {turnover:.4f}")
    # Guardar mejor modelo si mejora cum_return
    if cum_ret > best_cum_return:
        best_cum_return = cum_ret
        model_path = os.path.join(model_dir, file_base + ".zip")
        model.save(model_path)
        print(f"Nuevo mejor modelo guardado en {model_path} (cum_return={cum_ret:.4f})")

    # Bucle de entrenamiento y evaluación periódica
    start_time = time.time()
    for step in range(STEP_EVAL, TOTAL_TIMESTEPS + 1, STEP_EVAL):
        model.learn(total_timesteps=STEP_EVAL, reset_num_timesteps=False)
        cum_ret, sharpe, sortino, mdd, turnover = evaluate_agent(val_env, model)
        metrics.append({
            "step": step,
            "cum_return": cum_ret,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": mdd,
            "turnover": turnover
        })
        print(f"Step {step} -- cum_return: {cum_ret:.4f}, sharpe: {sharpe:.3f}, sortino: {sortino:.3f}, mdd: {mdd:.3f}, turnover: {turnover:.4f}")
        # Guardar mejor modelo si mejora cum_return
        if cum_ret > best_cum_return:
            best_cum_return = cum_ret
            model_path = os.path.join(model_dir, file_base + ".zip")
            model.save(model_path)
            print(f"Nuevo mejor modelo guardado en {model_path} (cum_return={cum_ret:.4f})")

    elapsed = time.time() - start_time
    print(f"\nTotal training time: {elapsed:.1f} segundos\n")

    # Comparación final con SPY Buy & Hold
    final = metrics[-1]
    print("Final DRL metrics:", final)
    bh_c, bh_sh, bh_so, bh_m = buy_and_hold_metrics(df_val)
    print(f"SPY Buy & Hold -- cum_return: {bh_c:.4f}, sharpe: {bh_sh:.3f}, sortino: {bh_so:.3f}, mdd: {bh_m:.3f}")

    # Guardar evolución de métricas
    pd.DataFrame(metrics).to_csv(os.path.join(model_dir, file_base + ".csv"), index=False)
    print(f"Métricas guardadas en {os.path.join(model_dir, file_base + '.csv')}")

if __name__ == "__main__":
    main()