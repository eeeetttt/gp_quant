"""
gp-quant 股票量化交易框架
"""
__version__ = "0.1.0"
__author__ = "gp-quant team"

from .data.fetcher import create_fetcher
from .data.processor import create_processor
from .data.storage import create_storage
from .strategy.base import Strategy, Signal, Order, Position
from .strategy.indicators import TechnicalIndicators
from .backtest.engine import BacktestEngine, Trade, Portfolio
from .ml.features import FeatureEngineer, FeatureConfig
from .ml.trainer import ModelTrainer, TrainConfig
from .ml.predictor import ModelPredictor
from .harness.engine import (
    HarnessEngine,
    RiskConfig,
    ScheduleConfig,
    FixedFractionSizer,
    PaperExecutionEngine,
)

__all__ = [
    # 数据模块
    "create_fetcher",
    "create_processor",
    "create_storage",

    # 策略模块
    "Strategy",
    "Signal",
    "Order",
    "Position",
    "TechnicalIndicators",

    # 回测模块
    "BacktestEngine",
    "Trade",
    "Portfolio",

    # 机器学习模块
    "FeatureEngineer",
    "FeatureConfig",
    "ModelTrainer",
    "TrainConfig",
    "ModelPredictor",

    # 交易调度模块
    "HarnessEngine",
    "RiskConfig",
    "ScheduleConfig",
    "FixedFractionSizer",
    "PaperExecutionEngine",
]
