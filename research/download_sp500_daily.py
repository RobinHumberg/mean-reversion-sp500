"""Lädt tägliche (split-/dividendenadjustierte) Schlusskurse für das S&P-500-Einzelaktienuniversum
(Large Cap aus data/universe_clean.csv) + SPY + ^IRX (13-Wochen-T-Bill, Cash-Rendite) und cached sie.

Run:  .venv/bin/python download_sp500_daily.py
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
START = "1998-01-01"
CHUNK = 40


def main() -> None:
    uni = pd.read_csv(DATA / "universe_clean.csv")
    tickers = sorted(uni.loc[uni["Segment"] == "Large Cap", "Ticker"].tolist())
    want = tickers + ["SPY", "^IRX"]

    frames, failed = [], []
    for i in range(0, len(want), CHUNK):
        chunk = want[i:i + CHUNK]
        for attempt in range(3):
            try:
                raw = yf.download(chunk, start=START, auto_adjust=True, progress=False, threads=True)
                close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
                frames.append(close)
                print(f"chunk {i // CHUNK + 1}/{-(-len(want) // CHUNK)}: {close.shape[1]} Spalten, "
                      f"{close.index[0].date()} .. {close.index[-1].date()}")
                break
            except Exception as e:  # noqa: BLE001
                print(f"  retry {attempt + 1}: {e}")
                time.sleep(5)
        else:
            failed += chunk

    px = pd.concat(frames, axis=1)
    px = px.loc[:, ~px.columns.duplicated()].sort_index()
    px.index = pd.to_datetime(px.index).tz_localize(None)
    empty = [c for c in px.columns if px[c].notna().sum() < 300]
    print(f"\nTicker gesamt: {px.shape[1]} | zu wenig Daten: {len(empty)} {empty[:20]} | Chunk-Fehler: {failed}")
    px.to_pickle(DATA / "daily_close_sp500.pkl")
    print(f"gespeichert: {DATA / 'daily_close_sp500.pkl'}  shape={px.shape}")


if __name__ == "__main__":
    main()
