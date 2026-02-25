"""Batman Strategy Engine — Core analytics and backtesting."""

from engine.strategy import (
    generate_strategy,
    Regime,
    RiskProfile,
    ButterflyPosition,
    BatmanPosition,
    StrategyRecommendation,
    compute_trap_score,
    round_to_strike,
)
from engine.backtester import (
    run_backtest,
    optimize_parameters,
    generate_synthetic_data,
    load_csv_data,
    BacktestSummary,
    OptimizationResult,
    DailyBar,
)
from engine.data_fetcher import (
    fetch_and_cache,
    load_cached_data,
    get_data_info,
)

__all__ = [
    "generate_strategy",
    "run_backtest",
    "optimize_parameters",
    "generate_synthetic_data",
    "load_csv_data",
    "fetch_and_cache",
    "load_cached_data",
    "get_data_info",
    "Regime",
    "RiskProfile",
    "ButterflyPosition",
    "BatmanPosition",
    "StrategyRecommendation",
    "BacktestSummary",
    "OptimizationResult",
    "DailyBar",
]
