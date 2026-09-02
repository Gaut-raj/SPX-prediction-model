"""End-to-end stock prediction ETL orchestration."""

import asyncio
import logging
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf

from config import AppConfig, DEFAULT_CONFIG
from extractor import MarketDataExtractor
from model import LSTMPriceModel
from transformer import FeatureTransformer

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    """Prediction outputs and evaluation metrics for one ticker."""

    ticker: str
    dates: pd.DatetimeIndex
    predictions: np.ndarray
    actuals: np.ndarray
    test_mae: float


class StockPredictionPipeline:
    """Coordinate extraction, feature engineering, training, and evaluation."""

    def __init__(self, config: AppConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO))
        random.seed(config.model.random_seed)
        np.random.seed(config.model.random_seed)
        tf.random.set_seed(config.model.random_seed)
        self.extractor = MarketDataExtractor()
        self.transformer = FeatureTransformer()

    def run(self, ticker: str) -> PipelineResult:
        """Run a complete chronological training and test evaluation for a ticker."""
        try:
            raw = asyncio.run(
                self.extractor.fetch(
                    ticker,
                    self.config.data.start_date,
                    self.config.data.end_date,
                    self.config.data.interval,
                )
            )
            featured = self.transformer.transform(raw)
            if len(featured) < self.config.data.minimum_rows:
                raise ValueError(
                    f"Ticker {ticker} has {len(featured)} usable rows; "
                    f"at least {self.config.data.minimum_rows} are required"
                )
            features, targets, scaler = self.transformer.create_windows(
                featured, self.config.data.lookback_window
            )
            split_index = int(len(features) * self.config.data.train_ratio)
            if split_index < 1 or split_index >= len(features):
                raise ValueError("train_ratio must create non-empty train and test sets")
            model = LSTMPriceModel(self.config.model)
            model.fit(features[:split_index], targets[:split_index])
            predictions = model.predict_original_values(features[split_index:], scaler, self.transformer)
            actuals = self.transformer.inverse_target_scale(targets[split_index:], scaler)
            dates = pd.DatetimeIndex(
                featured.index[self.config.data.lookback_window + split_index :]
            )
            if len(dates) != len(predictions):
                raise RuntimeError("Prediction dates do not align with test predictions")
            mae = float(np.mean(np.abs(predictions - actuals)))
            LOGGER.info("%s test MAE: %.4f", ticker, mae)
            return PipelineResult(ticker, dates, predictions, actuals, mae)
        except Exception as error:
            LOGGER.exception("Pipeline failed for %s", ticker)
            raise RuntimeError(f"End-to-end prediction pipeline failed for {ticker}") from error


if __name__ == "__main__":
    result = StockPredictionPipeline(DEFAULT_CONFIG).run("^GSPC")
    logging.getLogger(__name__).info("Completed %s with %d test predictions", result.ticker, len(result.predictions))