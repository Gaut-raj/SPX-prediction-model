"""Configuration models for the stock prediction pipeline."""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class DataConfig:
    """Market-data extraction and feature configuration."""

    tickers: Tuple[str, ...] = ("^GSPC",)
    start_date: date = date(2015, 1, 1)
    end_date: date = date.today()
    interval: str = "1d"
    lookback_window: int = 60
    train_ratio: float = 0.8
    minimum_rows: int = 120


@dataclass(frozen=True)
class ModelConfig:
    """LSTM architecture and training hyper-parameters."""

    lstm_units: Tuple[int, ...] = (64, 32)
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    epochs: int = 30
    batch_size: int = 32
    validation_ratio: float = 0.1
    random_seed: int = 42


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    log_level: str = "INFO"
    artifact_directory: Path = Path("artifacts")


DEFAULT_CONFIG = AppConfig()