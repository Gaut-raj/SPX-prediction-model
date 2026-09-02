# S&P 500 Prediction Model

This project downloads daily S&P 500 index data from Yahoo Finance (`^GSPC`),
calculates technical features, and trains a multivariate LSTM to predict the
closing value. It also includes a small Flask server and browser dashboard for
reviewing the held-out test predictions.

The model is for experimentation and learning. It is not a trading system, and
its predictions should not be treated as financial advice. Yahoo Finance data
is downloaded at run time, so the dashboard needs an internet connection.

## Files

- `config.py` contains data and model settings.
- `extractor.py` downloads market data with retries.
- `transformer.py` calculates features and creates LSTM windows.
- `model.py` defines and trains the Keras model.
- `pipeline.py` runs extraction, transformation, training, and evaluation.
- `dashboard.py` serves the browser frontend and prediction API.
- `frontend/` contains the HTML, CSS, and JavaScript dashboard.

## Setup

Install the Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the pipeline without the dashboard:

```powershell
python pipeline.py
```

Start the dashboard server:

```powershell
python dashboard.py
```

Open `http://127.0.0.1:5000` in a browser, then click **Run prediction**. Each
run downloads the data and trains the model before returning the test results.
