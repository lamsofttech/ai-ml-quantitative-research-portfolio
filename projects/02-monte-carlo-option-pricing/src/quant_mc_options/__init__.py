"""Vectorized Monte Carlo option pricing and risk analysis for quantitative research."""

from .black_scholes import Greeks, bs_greeks, bs_price
from .gbm import simulate_paths, simulate_terminal
from .monte_carlo import MCPriceResult, price_european
from .risk import MCGreeks, gbm_parametric_var, historical_cvar, historical_var, mc_greeks

__all__ = [
    "Greeks",
    "bs_greeks",
    "bs_price",
    "simulate_paths",
    "simulate_terminal",
    "MCPriceResult",
    "price_european",
    "MCGreeks",
    "gbm_parametric_var",
    "historical_cvar",
    "historical_var",
    "mc_greeks",
]
