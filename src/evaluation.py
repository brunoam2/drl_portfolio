import numpy as np

def compute_sharpe_sortino(log_rets):
    """
    Calcula Sharpe y Sortino anuales a partir de retornos logarítmicos diarios.
    """
    if len(log_rets) > 1:
        mean = np.mean(log_rets)
        std = np.std(log_rets) + 1e-8
        sharpe = mean / std * np.sqrt(252)
        downside = log_rets[log_rets < 0]
        sortino = (mean / (np.std(downside) + 1e-8) * np.sqrt(252)) if len(downside) > 0 else np.nan
    else:
        sharpe, sortino = np.nan, np.nan
    return sharpe, sortino

def compute_max_drawdown(equity):
    """
    Calcula el máximo drawdown de la serie de valores de portafolio.
    """
    high_water = np.maximum.accumulate(equity)
    drawdowns = (high_water - equity) / high_water
    return np.max(drawdowns)

def evaluate_agent(env, model):
    """
    Evalúa un agente entrenado en un entorno Gym de portafolio:
    - Ejecuta episodios hasta done.
    - Devuelve cum_ret, sharpe, sortino, max_drawdown, turnover.
    """
    obs, _ = env.reset()
    done = False
    cum_rewards = []
    turnovers = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        cum_rewards.append(info["accumulated_reward"])
        turnovers.append(info.get("turnover", 0.0))
    equity = np.exp(cum_rewards)
    log_rets = np.diff(np.log(equity))
    cum_return = equity[-1] - 1.0
    sharpe, sortino = compute_sharpe_sortino(log_rets)
    mdd = compute_max_drawdown(equity)
    avg_turnover = float(np.mean(turnovers)) if turnovers else 0.0
    return cum_return, sharpe, sortino, mdd, avg_turnover

def buy_and_hold_metrics(df):
    """
    Calcula métricas para la estrategia Buy & Hold sobre SPY:
    - df debe contener la columna 'SPY_logret'.
    """
    spy_lr = df["SPY_logret"].values
    equity = np.empty(len(spy_lr) + 1, dtype=float)
    equity[0] = 1.0
    equity[1:] = np.cumprod(np.exp(spy_lr))
    cum_return = equity[-1] - 1.0
    log_rets = np.diff(np.log(equity))
    sharpe, sortino = compute_sharpe_sortino(log_rets)
    mdd = compute_max_drawdown(equity)
    return cum_return, sharpe, sortino, mdd