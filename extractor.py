"""Reliable asynchronous market-data extraction backed by yfinance."""

import asyncio
import logging
from datetime import date
from typing import Optional

import pandas as pd
import yfinance as yf

LOGGER = logging.getLogger(__name__)


class MarketDataExtractor:
    """Download historical OHLCV data with bounded retries."""

    def __init__(self, max_retries: int = 3, backoff_seconds: float = 2.0) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    async def fetch(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch one ticker asynchronously, retrying transient failures."""
        if not ticker.strip():
            raise ValueError("ticker must not be empty")

        for attempt in range(1, self.max_retries + 1):
            try:
                LOGGER.info("Fetching %s, attempt %d/%d", ticker, attempt, self.max_retries)
                data = await asyncio.to_thread(
                    yf.download,
                    ticker,
                    start=start_date.isoformat(),
                    end=end_date.isoformat(),
                    interval=interval,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
                normalized = self._normalize_columns(data)
                if normalized.empty:
                    raise ValueError(f"No market data returned for ticker {ticker}")
                normalized["Ticker"] = ticker
                return normalized
            except Exception as error:
                LOGGER.warning("Market data attempt failed for %s: %s", ticker, error)
                if attempt == self.max_retries:
                    raise RuntimeError(
                        f"Unable to fetch market data for {ticker} after {self.max_retries} attempts"
                    ) from error
                await asyncio.sleep(self.backoff_seconds * (2 ** (attempt - 1)))
        raise RuntimeError(f"Unreachable extraction state for {ticker}")

    async def fetch_many(
        self,
        tickers: tuple[str, ...],
        start_date: date,
        end_date: date,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """Fetch multiple tickers concurrently and return them by symbol."""
        results = await asyncio.gather(
            *(self.fetch(ticker, start_date, end_date, interval) for ticker in tickers),
            return_exceptions=True,
        )
        output: dict[str, pd.DataFrame] = {}
        failures: list[str] = []
        for ticker, result in zip(tickers, results):
            if isinstance(result, Exception):
                failures.append(f"{ticker}: {result}")
            else:
                output[ticker] = result
        if failures:
            raise RuntimeError("One or more ticker downloads failed: " + "; ".join(failures))
        return output

    @staticmethod
    def _normalize_columns(data: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Normalize yfinance's possible single- and multi-level columns."""
        if data is None or data.empty:
            return pd.DataFrame()
        normalized = data.copy()
        if isinstance(normalized.columns, pd.MultiIndex):
            normalized.columns = normalized.columns.get_level_values(0)
        normalized.columns = [str(column) for column in normalized.columns]
        normalized.index.name = "Date"
        required = {"Open", "High", "Low", "Close", "Volume"}
        missing = required.difference(normalized.columns)
        if missing:
            raise ValueError(f"Downloaded data is missing required columns: {sorted(missing)}")
        return normalized.sort_index()