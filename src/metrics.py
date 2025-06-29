import numpy as np
import pandas as pd

def metrics(portfolio_net_returns: pd.Series, turnovers: pd.Series, accumulated_rewards: pd.Series, periods_per_year: int = 252) -> pd.Series:
    """Calcula métricas de rendimiento financiero como Sharpe, Sortino, drawdown y turnover."""

    # Ratio de Sharpe anualizado sobre retornos aritméticos netos
    mean_annual = portfolio_net_returns.mean() * periods_per_year
    vol_annual = portfolio_net_returns.std() * np.sqrt(periods_per_year)
    sharpe_ratio = mean_annual / vol_annual if vol_annual != 0 else np.nan

    # Ratio de Sortino anualizado sobre retornos aritméticos netos
    downside = portfolio_net_returns[portfolio_net_returns < 0]
    downside_deviation = downside.std() * np.sqrt(periods_per_year)
    sortino_ratio = mean_annual / downside_deviation if downside_deviation != 0 else np.nan

    # Máximo drawdown sobre la curva de riqueza total
    wealth_curve = np.exp(accumulated_rewards)
    peaks = wealth_curve.cummax()
    drawdowns = (wealth_curve - peaks) / peaks
    max_drawdown = drawdowns.min()

    # Valor final de la curva de riqueza (valor final del portfolio)
    final_portfolio_value = (wealth_curve).iloc[-1]

    # Promedio diario de turnover
    average_daily_turnover = turnovers.mean()

    metrics = {
        'final_portfolio_value': final_portfolio_value,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'average_daily_turnover': average_daily_turnover,
    }
    return pd.Series(metrics)