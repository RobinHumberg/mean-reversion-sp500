"""Lädt Open + Close (adjustiert) aller aktuellen S&P-500-Mitglieder (data/mr_sp500_repo/sp500.csv) + SPY + ^IRX ab 1998.

Run:  .venv/bin/python download_sp500_ohlc.py
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
START, CHUNK = "1998-01-01", 40


def main() -> None:
    meta = pd.read_csv(DATA / "mr_sp500_repo" / "sp500.csv")
    tickers = sorted(meta["Symbol"].str.replace(".", "-", regex=False).unique().tolist())
    want = tickers + ["SPY", "^IRX"]
    opens, closes, failed = [], [], []
    for i in range(0, len(want), CHUNK):
        chunk = want[i:i + CHUNK]
        for attempt in range(3):
            try:
                raw = yf.download(chunk, start=START, auto_adjust=True, progress=False, threads=True)
                opens.append(raw["Open"])
                closes.append(raw["Close"])
                print(f"chunk {i // CHUNK + 1}/{-(-len(want) // CHUNK)} ok ({raw['Close'].shape[1]} Ticker)")
                break
            except Exception as e:  # noqa: BLE001
                print(f"  retry {attempt + 1}: {e}")
                time.sleep(5)
        else:
            failed += chunk
    def cat(fr):
        x = pd.concat(fr, axis=1)
        x = x.loc[:, ~x.columns.duplicated()].sort_index()
        x.index = pd.to_datetime(x.index).tz_localize(None)
        return x
    o, c = cat(opens), cat(closes)
    print(f"\nOpen {o.shape}, Close {c.shape}, Fehler: {failed}, Zeitraum {c.index[0].date()}..{c.index[-1].date()}")
    pd.to_pickle({"open": o, "close": c}, DATA / "sp500_ohlc_1998.pkl")


if __name__ == "__main__":
    main()
