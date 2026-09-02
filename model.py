"""TensorFlow LSTM model and prediction utilities."""

import logging
from typing import Optional

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam

from config import ModelConfig
from transformer import FeatureTransformer

LOGGER = logging.getLogger(__name__)


class LSTMPriceModel:
    """Multivariate stacked LSTM for next-step normalized close prediction."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.network: Optional[Sequential] = None

    def build(self, input_shape: tuple[int, int]) -> Sequential:
        """Build and compile a safely shape-aware sequential network."""
        if len(input_shape) != 2 or min(input_shape) < 1:
            raise ValueError("input_shape must be (timesteps, feature_count), both positive")
        network = Sequential(name="multivariate_lstm")
        for index, units in enumerate(self.config.lstm_units):
            network.add(
                LSTM(
                    units,
                    return_sequences=index < len(self.config.lstm_units) - 1,
                    input_shape=input_shape if index == 0 else None,
                )
            )
            network.add(Dropout(self.config.dropout_rate))
        network.add(Dense(1))
        network.compile(optimizer=Adam(learning_rate=self.config.learning_rate), loss="mse")
        self.network = network
        return network

    def fit(self, features: np.ndarray, targets: np.ndarray) -> object:
        """Train without shuffling to preserve temporal order."""
        if features.ndim != 3 or targets.ndim != 1:
            raise ValueError("features must be 3D and targets must be 1D")
        if len(features) != len(targets):
            raise ValueError("features and targets must have equal sample counts")
        if self.network is None:
            self.build((features.shape[1], features.shape[2]))
        try:
            return self.network.fit(
                features,
                targets,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                validation_split=self.config.validation_ratio,
                shuffle=False,
                verbose=0,
            )
        except Exception as error:
            LOGGER.exception("LSTM training failed")
            raise RuntimeError("Unable to train LSTM model") from error

    def predict_original_values(
        self, features: np.ndarray, scaler: MinMaxScaler, transformer: FeatureTransformer
    ) -> np.ndarray:
        """Predict and reverse-scale normalized close values into dollars."""
        if self.network is None:
            raise RuntimeError("Model must be built or trained before prediction")
        try:
            normalized = self.network.predict(features, verbose=0).reshape(-1)
            return transformer.inverse_target_scale(normalized, scaler)
        except Exception as error:
            LOGGER.exception("Prediction or inverse scaling failed")
            raise RuntimeError("Unable to produce original-scale predictions") from error