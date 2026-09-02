"""Compatibility entry point for the modular stock prediction pipeline."""

from pipeline import StockPredictionPipeline


if __name__ == "__main__":
    StockPredictionPipeline().run("^GSPC")

