import numpy as np
import pandas as pd

from src.metrics import metrics


def run_benchmarks(df_train, df_test):
    prices_train = df_train[["SPY", "TLT", "GLD"]]
    returns_train = prices_train.pct_change().dropna()
    cov_matrix = compute_covariance(returns_train)
    mean_returns = returns_train.mean()

    prices_test = df_test[["SPY", "TLT", "GLD"]]
    returns_test = prices_test.pct_change().dropna()

    benchmarks = {}

    # 1. SPY Buy & Hold
    spy_returns = df_test["SPY"].pct_change().dropna()
    spy_curve = (1 + spy_returns).cumprod()
    benchmarks["SPY Buy & Hold"] = spy_curve

    # 2. Mínima varianza
    inv_vol = 1 / np.sqrt(np.diag(cov_matrix))
    min_var_weights = inv_vol / inv_vol.sum()
    min_var_curve = compute_portfolio_curve(returns_test, min_var_weights)
    benchmarks["Mínima varianza"] = min_var_curve

    # 3. Media-varianza (Sharpe max)
    weights_mv = mean_variance_weights(mean_returns, cov_matrix)
    mv_curve = compute_portfolio_curve(returns_test, weights_mv)
    benchmarks["Media-varianza"] = mv_curve

    # 4. Máxima diversificación
    divers_weights = max_diversification_weights(cov_matrix)
    divers_curve = compute_portfolio_curve(returns_test, divers_weights)
    benchmarks["Máxima diversificación"] = divers_curve

    # 5. Mínima correlación
    decor_weights = min_correlation_weights(cov_matrix)
    decor_curve = compute_portfolio_curve(returns_test, decor_weights)
    benchmarks["Mínima correlación"] = decor_curve

    # 6. Paridad de riesgo
    risk_parity_weights = risk_parity_allocation(cov_matrix)
    rp_curve = compute_portfolio_curve(returns_test, risk_parity_weights)
    benchmarks["Paridad de riesgo"] = rp_curve

    # Compilación de métricas
    metrics_df = []
    curves = {}
    for name, curve in benchmarks.items():
        returns = curve.pct_change().fillna(0)
        log_cum_rewards = np.log1p(returns).cumsum()
        met = metrics(returns, pd.Series([0.0] * len(returns)), log_cum_rewards)
        met.name = name
        metrics_df.append(met)
        curves[name] = curve

    df_out = pd.DataFrame(metrics_df)
    df_out.curves = curves  # para uso posterior opcional

    return df_out

def compute_portfolio_curve(returns, weights):
    weighted_returns = (returns * weights).sum(axis=1)
    return (1 + weighted_returns).cumprod()

def mean_variance_weights(mean_returns, cov_matrix, risk_aversion=1.0):
    inv_cov = np.linalg.inv(cov_matrix)
    weights = inv_cov @ mean_returns.values
    return weights / weights.sum()

def max_diversification_weights(cov_matrix):
    from scipy.optimize import minimize
    sigma = np.sqrt(np.diag(cov_matrix))

    def diversification_ratio(weights):
        numerator = np.dot(weights, sigma)
        denominator = np.sqrt(weights.T @ cov_matrix @ weights)
        return -numerator / denominator  # negative because we minimize

    n = len(sigma)
    x0 = np.ones(n) / n
    bounds = [(0, 1) for _ in range(n)]
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    result = minimize(diversification_ratio, x0, bounds=bounds, constraints=constraints)
    return result.x

def min_correlation_weights(cov_matrix):
    corr_matrix = cov_matrix / np.outer(np.sqrt(np.diag(cov_matrix)), np.sqrt(np.diag(cov_matrix)))
    mean_corr = corr_matrix.mean(axis=1)
    inv_corr = 1 / mean_corr
    return inv_corr / inv_corr.sum()

def risk_parity_allocation(cov_matrix):
    from scipy.optimize import minimize

    def risk_budget_objective_error(weights, args):
        cov = args
        port_var = weights.T @ cov @ weights
        sigma = np.sqrt(port_var) if port_var > 0 else 1e-8
        marginal_contrib = cov @ weights
        risk_contrib = weights * marginal_contrib / sigma
        target = np.ones_like(weights) / len(weights)
        return np.sum((risk_contrib - target * sigma) ** 2)

    n = cov_matrix.shape[0]
    x0 = np.ones(n) / n
    bounds = [(0.0, 1.0) for _ in range(n)]
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    result = minimize(risk_budget_objective_error, x0, args=(cov_matrix,), method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x


# --- Utility functions ---
def compute_covariance(returns):
    return returns.cov()