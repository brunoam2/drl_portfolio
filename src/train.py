import os
import sys
import pandas as pd
import numpy as np
import time
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
from sb3_contrib import RecurrentPPO

# Asegurarnos de que src esté en el path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.portfolio_env import PortfolioEnv
from src.metrics import metrics
from src.data import load_price_data

def evaluate_agent(model, env):
    """
    Evalúa el modelo en un único episodio y devuelve las métricas financieras.
    """
    obs, _ = env.reset()
    state = None
    done = False
    episode_start = True

    while not done:
        if hasattr(model.policy, 'lstm'):
            action, state = model.predict(obs, state=state, episode_start=episode_start, deterministic=True)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_start = done

    return metrics(
        pd.Series(env.history['portfolio_net_returns']),
        pd.Series(env.history['turnovers']),
        pd.Series(env.history['accumulated_rewards'])
    )


def print_metrics(title, metrics):
    """
    Print evaluation metrics.

    :param title: Title for the metrics printout.
    :param metrics: Dictionary of metrics.
    """
    print(title)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


import re

def parse_model_name(user_model):
    """
    Extrae los parámetros del identificador de modelo en formato 'SAC_lb30_reb5_tc0'.

    :param user_model: Cadena con el identificador del modelo.
    :return: model_name, lookback, rebalance_freq, transaction_cost
    """
    parts = user_model.split('_')
    model_name = parts[0].upper()
    lookback = int(re.search(r'\d+', parts[1]).group())
    rebalance_freq = int(re.search(r'\d+', parts[2]).group())
    transaction_cost = float(re.search(r'\d*\.?\d+', parts[3]).group())
    return model_name, lookback, rebalance_freq, transaction_cost


def get_model_class(model_name):
    """
    Get the model class from the model name.

    :param model_name: The name of the model.
    :return: The model class.
    """
    MODEL_MAP = {
        'PPO': PPO,
        'A2C': A2C,
        'DDPG': DDPG,
        'SAC': SAC,
        'TD3': TD3,
        'RECURRENTPPO': RecurrentPPO
    }
    return MODEL_MAP.get(model_name)


def train_and_evaluate(model_id, train_start, train_end, val_start, val_end, total_timesteps, eval_freq, seed, policy_kwargs=None):
    """
    Train and evaluate the model using the provided configuration.
    """
    model_name, lookback, rebalance_freq, transaction_cost = parse_model_name(model_id)
    model_class = get_model_class(model_name)

    df_train = load_price_data("datasets/combined_data.csv", train_start, train_end)
    df_val = load_price_data("datasets/combined_data.csv", val_start, val_end)

    train_env = PortfolioEnv(df_train, lookback, rebalance_freq, transaction_cost)
    val_env = PortfolioEnv(df_val, lookback, rebalance_freq, transaction_cost)

    if model_name == 'RECURRENTPPO':
        policy_type = 'MlpLstmPolicy'
    else:
        policy_type = 'MlpPolicy'

    def build_model():
        if policy_kwargs is not None:
            return model_class(policy_type, train_env, verbose=0, seed=seed, policy_kwargs=policy_kwargs)
        else:
            return model_class(policy_type, train_env, verbose=0, seed=seed)

    start_time = time.time()

    # --- Baseline con SPY (buy and hold) ---
    spy_returns = df_val["SPY_Return"]
    spy_log_cum_rewards = np.log1p(spy_returns).cumsum()
    spy_metrics = metrics(
        pd.Series(spy_returns),
        pd.Series([0] * len(spy_returns)),  # Sin turnover
        pd.Series(spy_log_cum_rewards)
    )
    print_metrics("=== Baseline SPY Buy & Hold ===", spy_metrics)

    # --- Evaluación inicial del modelo sin entrenar ---
    untrained_model = build_model()
    untrained_metrics = evaluate_agent(untrained_model, val_env)
    print_metrics("=== Validación inicial (modelo sin entrenar) ===", untrained_metrics)

    model = build_model()
    model.set_random_seed(seed)

    os.makedirs("models", exist_ok=True)
    best_return = -np.inf

    training_results = []

    timestep = 0
    while timestep < total_timesteps:
        model.learn(total_timesteps=eval_freq, reset_num_timesteps=False)
        timestep += eval_freq
        eval_metrics = evaluate_agent(model, val_env)
        print_metrics(f"=== Evaluación en el paso {timestep} ===", eval_metrics)

        training_results.append({"timestep": timestep, **eval_metrics.to_dict()})

        if eval_metrics["final_portfolio_value"] > best_return:
            best_return = eval_metrics["final_portfolio_value"]
            model.save(f"models/{model_id}.zip")
            print(f"Nuevo mejor modelo guardado en el paso {timestep} con retorno final: {best_return:.4f}")

    end_time = time.time()
    print(f"Tiempo de entrenamiento (s): {end_time - start_time:.1f}")

    # Guardar métricas de entrenamiento por paso en CSV
    results_df = pd.DataFrame(training_results)
    results_path = os.path.join("results", "model_training", f"{model_id}.csv")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results_df.to_csv(results_path, index=False)

    print("=== Evaluación del mejor modelo guardado ===")
    best_model = model_class.load(f"models/{model_id}.zip", env=val_env)
    best_metrics = evaluate_agent(best_model, val_env)
    print_metrics("=== Mejor modelo guardado ===", best_metrics)
