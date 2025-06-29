# src/portfolio_env.py

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from src.observation_builder import build_observation_mlp

class PortfolioEnv(gym.Env):
    """
    Entorno personalizado de Gym para simular una cartera de inversión con activos SPY, TLT, GLD y liquidez (CASH).
    Utiliza una representación MLP de observación. La recompensa se define como el log-retorno neto de la cartera.
    """

    def __init__(
        self,
        combined_data: pd.DataFrame,
        lookback: int,
        rebalance_freq: int,
        transaction_cost: float,
    ):
        super().__init__()

        self.combined_data = combined_data.copy()
        self.lookback = lookback
        self.rebalance_freq = rebalance_freq
        self.transaction_cost = transaction_cost

        self.observation_builder = build_observation_mlp

        self.asset_names = ["SPY", "TLT", "GLD", "CASH"]
        self.n_assets = len(self.asset_names)
        self.all_dates = combined_data.index.to_list()

        self.initial_weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)

        self.reset()
        self.end_step = len(self.all_dates) - 1

        sample_obs = self.observation_builder(
            combined_data=self.combined_data,
            step=self.current_step,
            lookback=self.lookback,
            current_weights = self.current_weights, 
        )
        obs_shape = sample_obs.shape

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)

    def reset(self, **kwargs):
        """
        Reinicia el entorno y devuelve la observación inicial.
        """
        self.current_step = self.lookback - 1
        self.current_weights = self.initial_weights.copy()

        self.history = {
            "weights": [self.current_weights.copy()],
            "portfolio_net_returns": [],
            "rewards": [],
            "accumulated_rewards": [0.0],
            "turnovers": [0.0],
            "costs":[],
            "dates": [self.all_dates[self.current_step]],
        }

        self.done = False

        obs = self.observation_builder(
            combined_data=self.combined_data,
            step=self.current_step,
            lookback=self.lookback,
            current_weights = self.current_weights, 
        )
        return obs, {}

    def step(self, action):
        """
        Avanza un paso en el entorno, calcula la recompensa y genera la siguiente observación.
        """

        # Tomar la decisión de rebalanceo cada `rebalance_freq` pasos
        if (self.current_step - (self.lookback - 1)) % self.rebalance_freq == 0:
            action = np.clip(action, 0, None)
            new_weights = action / np.sum(action) if np.sum(action) > 0 else self.current_weights.copy()
            turnover = np.sum(np.abs(new_weights - self.current_weights)) / 2.0
            cost = self.transaction_cost * turnover
        else:
            new_weights = self.current_weights.copy()
            turnover = 0.0
            cost = 0.0

        self.history["turnovers"].append(turnover)
        self.history["costs"].append(cost)

        # Actualizar el paso actual y verificar si se ha alcanzado el final
        self.current_step += 1
        if self.current_step > self.end_step:
            self.current_step = self.end_step
            self.done = True

        date = self.all_dates[self.current_step]
        self.history["dates"].append(date)
        step_index = self.current_step - (self.lookback - 1)

        # Calcular retorno bruto del portafolio como suma ponderada de retornos de activos
        asset_returns = []
        for asset in self.asset_names:
            if asset == "CASH":
                asset_returns.append(0.0)
            else:
                asset_returns.append(self.combined_data.loc[date, f"{asset}_Return"])
        asset_returns = np.array(asset_returns)
        portfolio_gross_return = np.dot(new_weights, asset_returns)
        portfolio_net_return = portfolio_gross_return - cost
        step_reward = np.log1p(portfolio_net_return)

        self.history["portfolio_net_returns"].append(portfolio_gross_return)
        self.history["rewards"].append(step_reward)
        accumulated_rewards = np.sum(self.history["rewards"])
        self.history["accumulated_rewards"].append(accumulated_rewards)

        # Ajustar pesos según retorno de activos para reflejar evolución del portafolio (drift)
        drifted_weights = new_weights * (1 + asset_returns)
        drifted_weights = drifted_weights / drifted_weights.sum()

        self.current_weights = drifted_weights.copy()
        self.history["weights"].append(self.current_weights.copy())
        
        # Construir la observación para el siguiente paso
        if not self.done:
            obs = self.observation_builder(
                combined_data=self.combined_data,
                step=self.current_step,
                lookback=self.lookback,
                current_weights = self.current_weights,
            )
        else:
            obs = None

        info = {
            "date" : date,
            "reward": step_reward,
            "accumulated_rewards": accumulated_rewards,
            "weights": self.current_weights.copy(),
            "turnover": turnover,
            "transaction_cost": cost,
        }

        terminated = self.done
        truncated = False

        return obs, step_reward, terminated, truncated, info