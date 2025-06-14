# src/portfolio_env.py

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

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
        observation_mode: str = "mlp",
    ):
        super().__init__()

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
        self.all_dates = combined_data.index.to_list()

        self.initial_weights = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

        self.reset()
        self.end_step = len(self.all_dates) - 1

        sample_obs = self.observation_builder(
            combined_data=self.combined_data,
            step=self.current_step,
            lookback=self.lookback,
            asset_names=self.asset_names,
        )
        obs_shape = sample_obs.shape

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)

    def reset(self, **kwargs):
        """
        Reinicia entorno y devuelve la observación inicial.
        """
        self.current_step = self.lookback - 1
        self.current_weights = self.initial_weights.copy()

        self.history = {
            "weights": [self.current_weights.copy()],
            "accumulated_reward": [0.0],
            "rewards": [],
            "turnovers": [0.0],
            "dates": [self.all_dates[self.current_step]],
        }

        self.done = False

        obs = self.observation_builder(
            combined_data=self.combined_data,
            step=self.current_step,
            lookback=self.lookback,
            asset_names=self.asset_names,
        )
        return obs, {}

    def step(self, action):
        """
        Avanza un paso en el entorno, calcula recompensa y siguiente observación.
        """
        step_index = self.current_step - (self.lookback - 1)
        date = self.all_dates[self.current_step]
        asset_returns = []

        # Determinar si es momento de rebalancear
        if step_index % self.rebalance_freq == 0:
            action = np.clip(action, 0, None)
            new_weights = action / action.sum()

            turnover = np.sum(np.abs(new_weights - self.current_weights)) / 2.0
            cost = self.transaction_cost * turnover
        else:
            new_weights = self.current_weights.copy()
            turnover = 0.0
            cost = 0.0

        self.history["turnovers"].append(turnover)

        # Calcular retorno del portafolio
        asset_returns = []
        for asset in self.asset_names:
            if asset == "CASH":
                asset_returns.append(0.0)
            else:
                asset_returns.append(self.combined_data.loc[date, f"{asset}_Return"])
        asset_returns = np.array(asset_returns)
        portfolio_return = np.dot(new_weights, asset_returns) - cost
        step_reward = np.log1p(portfolio_return)

        self.history["rewards"].append(step_reward)
        cumulative = self.history["accumulated_reward"][-1] + step_reward
        self.history["accumulated_reward"].append(cumulative)

        # Aplicar drift de mercado: ajustar pesos según el retorno de cada activo
        drifted_weights = new_weights * (1 + asset_returns)
        drifted_weights = drifted_weights / drifted_weights.sum()

        self.current_weights = drifted_weights.copy()
        self.history["weights"].append(self.current_weights.copy())
        
        # Actualizar paso y verificar si se ha alcanzado el final
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
            )
        else:
            obs = None

        info = {
            "date" : date,
            "reward": step_reward,
            "accumulated_reward": cumulative,
            "weights": self.current_weights.copy(),
            "turnover": turnover,
            "transaction_cost": cost,
        }

        terminated = self.done
        truncated = False

        return obs, step_reward, terminated, truncated, info