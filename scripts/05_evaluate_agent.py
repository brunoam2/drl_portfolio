

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# project root for src imports
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.portfolio_env import PortfolioEnv
from src.evaluation import evaluate_performance
from stable_baselines3 import SAC

# === CONFIGURATION (match training) ===
LOOKBACK = 60
REBALANCE_FREQ = 10
TRANSACTION_COST = 0.001

def evaluate_agent(env, model):
    """Run the agent through env and collect history, then compute metrics."""
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
    # build metrics from history
    returns = pd.Series(env.history["rewards"])
    turnovers = pd.Series(env.history.get("turnovers", []))
    cumulative_returns = pd.Series(env.history["accumulated_reward"])
    return evaluate_performance(returns, turnovers, cumulative_returns)

def main():
    data_path = "datasets/combined_data.csv"
    model_path = os.path.join("models", "SAC_MlpPolicy_lb60_reb10_tc0.zip")  # update to actual filename
    start_date = "2019-01-01"
    end_date = "2024-01-01"

    # load data
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df_eval = df.loc[start_date:end_date]

    # create environment
    env = PortfolioEnv(
        combined_data=df_eval,
        lookback=LOOKBACK,
        rebalance_freq=REBALANCE_FREQ,
        transaction_cost=TRANSACTION_COST,
        observation_mode="mlp",
    )

    # load trained model
    model = SAC.load(model_path, env=env)

    # evaluate and print metrics
    metrics = evaluate_agent(env, model)
    print("=== Evaluation 2019-2024 ===")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    # extract weight history
    dates = pd.to_datetime(env.history["dates"])
    weights = np.vstack(env.history["weights"])
    df_weights = pd.DataFrame(weights, index=dates, columns=env.asset_names)

    # save metrics and weights
    os.makedirs("results", exist_ok=True)
    metrics.to_frame("value").to_csv("results/eval_metrics_2019_2024.csv")
    df_weights.to_csv("results/eval_weights_2019_2024.csv")

    # plot weight evolution
    plt.figure(figsize=(10, 6))
    for col in df_weights.columns:
        plt.plot(df_weights.index, df_weights[col], label=col)
    plt.title("Portfolio Weights Evolution 2019-2024")
    plt.xlabel("Date")
    plt.ylabel("Weight")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/weights_evolution_2019_2024.png")
    plt.show()

if __name__ == "__main__":
    main()