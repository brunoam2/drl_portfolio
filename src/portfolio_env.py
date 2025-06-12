# src/portfolio_env.py

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

# Añadimos el project root al path para que `import src` funcione
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.observation_builder import build_observation_mlp, build_observation_cnn


class PortfolioEnv(gym.Env):
    """
    Entorno de cartera para SPY, TLT, GLD y CASH.
    observation_mode puede ser "mlp" o "cnn" para usar el builder correspondiente.
    Recompensa = log-return portafolio − coste.
    """

    def __init__(
        self,
        combined_data: pd.DataFrame,
        lookback: int,
        rebalance_freq: int,
        transaction_cost: float,
        seed: int,
        observation_mode: str,
    ):
        super().__init__()

        if seed is not None:
            np.random.seed(seed)

        self.combined_data = combined_data.copy()

        self.lookback = lookback
        self.rebalance_freq = rebalance_freq
        self.transaction_cost = transaction_cost

        self.observation_mode = observation_mode
        if observation_mode == "mlp":
            self.observation_builder = build_observation_mlp
        elif observation_mode == "cnn":
            self.observation_builder = build_observation_cnn
        else:
            raise ValueError(f"Modo de observación desconocido: {observation_mode}")

        self.asset_names = ["SPY", "TLT", "GLD", "CASH"]
        self.n_assets = len(self.asset_names)
        # Columnas macroeconómicas a utilizar en los builders
        self.macro_cols = ["CPIAUCSL_logm", "GDPC1_logq", "FEDFUNDS", "VIX", "DX_log"]

        self.all_dates = combined_data.index.to_list()
        self.end_step = len(self.all_dates) - 1

        # Inicialización del entorno
        obs, _ = self.reset()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )

    def reset(self, **kwargs):
        """
        Reinicia entorno y devuelve la observación inicial.
        """
        self.current_step = self.lookback - 1
        # Peso inicial: 100% en CASH
        self.current_weights = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

        self.history = {
            "weights": [self.current_weights.copy()],
            "accumulated_reward": [0.0],
            "rewards": [],
            "dates": [self.all_dates[self.current_step]],
        }

        self.done = False

        obs = self.observation_builder(
            combined_data=self.combined_data,
            step=self.current_step,
            lookback=self.lookback,
            asset_names=self.asset_names,
            macro_cols=self.macro_cols,
        )
        return obs, {}

    def step(self, action):
        """
        Avanza un paso en el entorno, calcula recompensa y siguiente observación.
        """
        # Normalizar pesos recibidos para garantizar suma 1
        
        action = np.array(action, dtype=float)
        action = np.clip(action, 0.0, None)
        if action.sum() > 0:
            new_weights = action / action.sum()
        else:
            # Fallback: 100% en CASH si recibimos un vector de ceros
            new_weights = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

        step_index = self.current_step - (self.lookback - 1)
        cost = 0.0
        turnover = 0.0
        # Coste solo en rebalanceo
        if step_index % self.rebalance_freq == 0:
            turnover = np.sum(np.abs(new_weights - self.current_weights)) / 2.0
            cost = self.transaction_cost * turnover
        else:
            new_weights = self.current_weights.copy()

        date = self.all_dates[self.current_step]
        asset_returns = []
        for asset in self.asset_names:
            if asset == "CASH":
                asset_returns.append(0.0)
            else:
                asset_returns.append(self.combined_data.loc[date, f"{asset}_logret"])
        asset_returns = np.array(asset_returns)

        portfolio_return = np.dot(new_weights, asset_returns)
        step_reward = portfolio_return - cost

        self.history["rewards"].append(step_reward)
        cumulative = self.history["accumulated_reward"][-1] + step_reward
        self.history["accumulated_reward"].append(cumulative)

        self.current_weights = new_weights
        self.history["weights"].append(self.current_weights.copy())
        self.history["dates"].append(date)

        self.current_step += 1
        if self.current_step > self.end_step:
            self.done = True

        if not self.done:
            obs = self.observation_builder(
                combined_data=self.combined_data,
                step=self.current_step,
                lookback=self.lookback,
                asset_names=self.asset_names,
                macro_cols=self.macro_cols,
            )
        else:
            obs = None

        info = {
            "accumulated_reward": cumulative,
            "transaction_cost": cost,
            "turnover": turnover,
            "weights": self.current_weights.copy(),
            "reward": step_reward,
        }

        terminated = self.done
        truncated = False

        return obs, step_reward, terminated, truncated, info