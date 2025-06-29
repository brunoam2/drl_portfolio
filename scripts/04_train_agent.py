"""
04_train_agent.py

Entrenamiento SAC con evaluaciones periódicas y guardado del mejor modelo
basado en rendimiento sobre validación.
"""

import os
import sys
import random
import numpy as np
import torch

# Asegurarnos de que src esté en el path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.train import train_and_evaluate

# --- User parameters (edit these) ---
# Rango temporal para entrenamiento y validación
TRAIN_START = "2006-01-01"
TRAIN_END = "2016-12-31"
VAL_START = "2017-01-01"
VAL_END = "2021-12-31"

# Identificador del modelo con formato estándar
MODEL_ID = "SAC_lb30_reb10_tc0_LeakyReLU,"

# Otros parámetros
TOTAL_TIMESTEPS = 150000
EVAL_FREQ = 5000
SEED = 69

# Establecer la semilla global
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Entrenamiento y evaluación
train_and_evaluate(
    model_id=MODEL_ID,
    train_start=TRAIN_START,
    train_end=TRAIN_END,
    val_start=VAL_START,
    val_end=VAL_END,
    total_timesteps=TOTAL_TIMESTEPS,
    eval_freq=EVAL_FREQ,
    seed=SEED,
    policy_kwargs={
        "activation_fn": torch.nn.Tanh
    }
)