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
LOOKBACK = 30
REBALANCE_FREQ = 10
TRANSACTION_COST = 0

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
    transaction_costs = pd.Series(env.history.get("costs", []))
    cumulative_costs = transaction_costs.cumsum()
    portfolio_value = np.exp(cumulative_returns)
    return evaluate_performance(returns, turnovers, cumulative_returns), cumulative_costs, portfolio_value

def main():
    data_path = "datasets/combined_data.csv"
    model_path = "models/SAC_MlpPolicy_lb30_reb10_tc0.zip"
    start_date = "2022-01-01"
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
    metrics, cumulative_costs, portfolio_value = evaluate_agent(env, model)
    print("=== Evaluation 2022-2024 ===")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    # extract weight history
    dates = pd.to_datetime(env.history["dates"])
    weights = np.vstack(env.history["weights"])
    df_weights = pd.DataFrame(weights, index=dates, columns=env.asset_names)

    # save metrics and weights
    os.makedirs("results", exist_ok=True)
    metrics.to_frame("value").to_csv("results/model_evaluation/eval_metrics_2022_2024.csv")
    df_weights.to_csv("results/model_evaluation/eval_weights_2022_2024.csv")
    pd.Series(cumulative_costs.values, index=dates[1:], name="cumulative_costs").to_frame().to_csv("results/model_evaluation/eval_cumulative_costs_2022_2024.csv")

    returns = pd.Series(env.history["rewards"])
    transaction_costs = pd.Series(env.history.get("costs", []))

    # align returns and transaction_costs index with dates[1:]
    returns.index = dates[1:]
    transaction_costs.index = dates[1:]

    # guardar CSV con reward y transaction cost diario
    df_daily = pd.DataFrame({
        "reward": returns,
        "transaction_cost": transaction_costs,
        "reward_minus_cost": returns - transaction_costs
    })
    df_daily.to_csv("results/model_evaluation/reward_and_transaction_cost_2022_2024.csv")

    # plot weight evolution
    plt.figure(figsize=(10, 6))
    for col in df_weights.columns:
        plt.plot(df_weights.index, df_weights[col], label=col)
    plt.title("Portfolio Weights Evolution 2022-2024")
    plt.xlabel("Date")
    plt.ylabel("Weight")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/model_evaluation/weights_evolution_2022_2024.png")
    plt.show()

    # plot cumulative transaction costs and portfolio value
    plt.figure(figsize=(10, 6))
    plt.plot(dates[1:], cumulative_costs, label="Cumulative Transaction Costs")
    plt.plot(dates, portfolio_value, label="Portfolio Value (exp of cumulative log-returns)")
    plt.title("Portfolio Value and Transaction Costs 2022-2024")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/model_evaluation/value_and_costs_2022_2024.png")
    plt.show()

    # plot cumulative daily return and cumulative transaction cost
    cumulative_daily_return = returns.cumsum()
    cumulative_daily_cost = transaction_costs.cumsum()

    plt.figure(figsize=(10, 6))
    plt.plot(dates, cumulative_daily_return, label="Cumulative Daily Return")
    plt.plot(dates[1:], cumulative_daily_cost, label="Cumulative Transaction Cost")
    plt.title("Cumulative Daily Return vs Transaction Cost")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/model_evaluation/cumulative_return_vs_cost_2022_2024.png")
    plt.show()

if __name__ == "__main__":
    main()