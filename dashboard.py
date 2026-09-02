"""Local web server for the S&P 500 prediction frontend."""

import logging
from pathlib import Path

from flask import Flask, jsonify, render_template

from config import DEFAULT_CONFIG
from pipeline import StockPredictionPipeline

LOGGER = logging.getLogger(__name__)
FRONTEND_DIRECTORY = Path(__file__).parent / "frontend"
app = Flask(__name__, template_folder=str(FRONTEND_DIRECTORY), static_folder=str(FRONTEND_DIRECTORY))


@app.get("/")
def index() -> str:
    """Serve the dashboard frontend."""
    return render_template("index.html")


@app.post("/api/predictions")
def predictions() -> tuple[object, int] | object:
    """Execute the S&P 500 pipeline and return chart-ready JSON."""
    try:
        result = StockPredictionPipeline(DEFAULT_CONFIG).run("^GSPC")
        points = [
            {
                "date": date.isoformat(),
                "actual": float(actual),
                "predicted": float(predicted),
            }
            for date, actual, predicted in zip(result.dates, result.actuals, result.predictions)
        ]
        return jsonify(
            {
                "ticker": result.ticker,
                "mae": result.test_mae,
                "points": points,
            }
        )
    except Exception as error:
        LOGGER.exception("Dashboard prediction run failed")
        return jsonify({"error": str(error)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)