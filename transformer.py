"""Feature engineering and supervised temporal data preparation."""

import logging
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

LOGGER = logging.getLogger(__name__)
FEATURE_COLUMNS: tuple[str, ...] = (
    "Close",
    "Daily_Return",
    "Log_Return",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "Rolling_Volatility_5",
    "Volume",
)


class FeatureTransformer:
    """Compute technical features and chronological LSTM windows."""

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a cleaned frame containing engineered model features."""
        try:
            frame = data.copy().sort_index()
            required = {"Close", "Volume"}
            missing = required.difference(frame.columns)
            if missing:
                raise ValueError(f"Input data is missing columns: {sorted(missing)}")
            close = frame["Close"].astype(float)
            delta = close.diff()
            gains = delta.clip(lower=0).rolling(14, min_periods=14).mean()
            losses = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
            relative_strength = gains / losses.replace(0, np.nan)
            frame["RSI_14"] = 100 - (100 / (1 + relative_strength))
            frame["RSI_14"] = frame["RSI_14"].fillna(100.0).clip(0, 100)
            ema_fast = close.ewm(span=12, adjust=False, min_periods=12).mean()
            ema_slow = close.ewm(span=26, adjust=False, min_periods=26).mean()
            frame["MACD"] = ema_fast - ema_slow
            frame["MACD_Signal"] = frame["MACD"].ewm(span=9, adjust=False, min_periods=9).mean()
            frame["Daily_Return"] = close.pct_change()
            frame["Log_Return"] = np.log(close / close.shift(1))
            frame["Rolling_Volatility_5"] = frame["Log_Return"].rolling(5, min_periods=5).std()
            frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=list(FEATURE_COLUMNS))
            if frame.empty:
                raise ValueError("Feature engineering produced no complete rows")
            return frame
        except Exception as error:
            LOGGER.exception("Feature transformation failed")
            raise RuntimeError("Unable to calculate financial features") from error

    def create_windows(
        self,
        data: pd.DataFrame,
        lookback_window: int,
        target_column: str = "Close",
    ) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        """Scale features and construct 3D chronological input windows."""
        try:
            if lookback_window < 1:
                raise ValueError("lookback_window must be positive")
            if target_column not in FEATURE_COLUMNS:
                raise ValueError(f"Unsupported target column: {target_column}")
            values = data.loc[:, list(FEATURE_COLUMNS)].astype(float).to_numpy()
            if len(values) <= lookback_window:
                raise ValueError("Not enough rows to construct an LSTM window")
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(values)
            target_index = FEATURE_COLUMNS.index(target_column)
            features = np.asarray(
                [scaled[index - lookback_window:index] for index in range(lookback_window, len(scaled))],
                dtype=np.float32,
            )
            targets = np.asarray(
                [scaled[index, target_index] for index in range(lookback_window, len(scaled))],
                dtype=np.float32,
            )
            return features, targets, scaler
        except Exception as error:
            LOGGER.exception("Window construction failed")
            raise RuntimeError("Unable to prepare temporal model windows") from error

    @staticmethod
    def inverse_target_scale(
        scaled_values: Sequence[float], scaler: MinMaxScaler, target_column: str = "Close"
    ) -> np.ndarray:
        """Inverse-transform one target feature without fabricating other features."""
        target_index = FEATURE_COLUMNS.index(target_column)
        values = np.zeros((len(scaled_values), len(FEATURE_COLUMNS)), dtype=float)
        values[:, target_index] = np.asarray(scaled_values, dtype=float)
        return scaler.inverse_transform(values)[:, target_index]