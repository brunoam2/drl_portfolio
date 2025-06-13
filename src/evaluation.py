import numpy as np
import pandas as pd

def evaluate_performance(returns: pd.Series, turnovers: pd.Series, cumulative_returns: pd.Series, periods_per_year: int = 252) -> pd.Series:
    """
    Calcula métricas de desempeño:
    - cumulative_return (a partir de la serie de log-retornos convertida internamente)
    - sharpe_ratio (anualizado, tasa libre de riesgo cero)
    - sortino_ratio (anualizado, tasa libre de riesgo cero)
    - max_drawdown
    - average_daily_turnover
    """
    # Convertir log-retornos y log-retornos acumulados a retornos aritméticos
    arith_returns = np.expm1(returns)
    arith_cum_returns = np.expm1(cumulative_returns)

    # Ratio de Sharpe anualizado sobre retornos aritméticos
    mean_annual = arith_returns.mean() * periods_per_year
    vol_annual = arith_returns.std() * np.sqrt(periods_per_year)
    sharpe_ratio = mean_annual / vol_annual if vol_annual != 0 else np.nan

    # Ratio de Sortino anualizado sobre retornos aritméticos
    downside = arith_returns[arith_returns < 0]
    downside_deviation = downside.std() * np.sqrt(periods_per_year)
    sortino_ratio = mean_annual / downside_deviation if downside_deviation != 0 else np.nan

    # Convertir serie acumulada de log-retornos a retornos aritméticos

    # Curva de riqueza: 1 + retornos acumulados aritméticos
    wealth = 1 + arith_cum_returns
    peaks = wealth.cummax()
    drawdowns = (wealth - peaks) / peaks
    max_drawdown = drawdowns.min()

    # Retorno acumulado final en escala aritmética
    arith_cumulative_return = arith_cum_returns.iloc[-1]

    # Promedio diario de turnover
    average_daily_turnover = turnovers.mean()

    metrics = {
        'cumulative_return': arith_cumulative_return,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'average_daily_turnover': average_daily_turnover,
    }
    return pd.Series(metrics)