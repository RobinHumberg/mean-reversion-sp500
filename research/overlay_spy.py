"""RSI(2)-Strategy (run.py, Strategy 0071) als Overlay auf den S&P 500 (SPY).

Konstruktion
------------
Kapital wird gleichgewichtet auf die 4 Sleeves (SPY, QQQ, DIA, IWM) aufgeteilt (je 25%).
Pro Sleeve gilt (Signal aus run.py, Entscheidung am Close t, Halten ab t+1):
    Signal = 1 (Strategy im Markt)  -> Sleeve-Kapital liegt im Sleeve-ETF
    Signal = 0 (Strategy flat)      -> Sleeve-Kapital liegt im SPY  (= S&P-Overlay)
Das Portfolio ist damit immer zu 100% investiert (kein Cash).

Kosten: 2 bps pro Seite (IBKR_LIQUID_ETF) auf den Turnover der Gewichte, d.h. ein
Wechsel SPY -> QQQ kostet beide Seiten (SPY verkaufen + QQQ kaufen).

Run:  .venv/bin/python overlay_spy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

QUANTLAB_SRC = Path("/Users/robinhumberg/Documents/Backtests/src")
sys.path.insert(0, str(QUANTLAB_SRC))

from quantlab.costs import IBKR_LIQUID_ETF  # noqa: E402
from quantlab.data import get_prices  # noqa: E402
from quantlab.metrics import (compute_metrics, max_drawdown,  # noqa: E402
                              max_drawdown_duration)

HERE = Path(__file__).resolve().parent
OUT = HERE / "overlay_results"
OUT.mkdir(exist_ok=True)

START = "1999-01-01"
SLEEVES = ["SPY", "QQQ", "DIA", "IWM"]
BASE = "SPY"
UNIVERSES = {
    "4 Indizes": SLEEVES,
    "16 Equity-ETFs": SLEEVES + ["MDY", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB",
                                 "EFA", "EEM"],
    "18 inkl. GLD, TLT": SLEEVES + ["MDY", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB",
                                    "EFA", "EEM", "GLD", "TLT"],
}
MODES = {"equal": "je Sleeve 1/N", "all_in": "Alles-rein"}
FIRST = pd.Timestamp("2000-05-26")  # Start des jüngsten der 4 Basis-ETFs (IWM) -> gleiche Periode überall
RF = 0.02
TDAYS = 252
COST_SIDE = (IBKR_LIQUID_ETF.slippage_bps + IBKR_LIQUID_ETF.regulatory_bps) / 1e4


# ---------------------------------------------------------------- Signal (aus run.py)
def rsi(close: pd.Series, period: int = 2) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    ag = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    al = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = ag / al.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def build_signal(df: pd.DataFrame, entry_th: float = 10, trend: int = 200,
                 sma_exit: int = 5) -> pd.Series:
    close = df["Close"]
    r = rsi(close, 2)
    c, rt = close.to_numpy(), r.to_numpy()
    st = close.rolling(trend).mean().to_numpy()
    se = close.rolling(sma_exit).mean().to_numpy()
    sig = np.zeros(len(c))
    in_pos = False
    for i in range(len(c)):
        if not in_pos:
            if not np.isnan(st[i]) and rt[i] < entry_th and c[i] > st[i]:
                in_pos = True
        elif c[i] > se[i]:
            in_pos = False
        sig[i] = 1.0 if in_pos else 0.0
    return pd.Series(sig, index=close.index)


# ---------------------------------------------------------------- Overlay-Engine
def build_overlay(px: dict, assets: list, mode: str = "equal", sig_cache: dict = None) -> dict:
    """mode="equal":  jedes der N Sleeves hat 1/N Kapital; flat -> sein Anteil liegt im SPY.
    mode="all_in": aktive Sleeves teilen das GESAMTE Kapital zu gleichen Teilen; keins aktiv -> 100% SPY."""
    idx = px[BASE].index
    ret = pd.DataFrame({t: px[t]["Close"].reindex(idx).pct_change() for t in assets}).fillna(0.0)
    sig = pd.DataFrame({t: (sig_cache[t] if sig_cache and t in sig_cache else
                            build_signal(px[t])).reindex(idx).fillna(0.0) for t in assets})
    held = sig.shift(1).fillna(0.0)  # Entscheidung Close t -> gehalten ab t+1

    w = pd.DataFrame(0.0, index=idx, columns=assets)
    if mode == "equal":
        for t in assets:
            w[t] += held[t] / len(assets)
            w[BASE] += (1.0 - held[t]) / len(assets)
    elif mode == "all_in":
        n_act = held.sum(axis=1)
        for t in assets:
            w[t] += (held[t] / n_act.where(n_act > 0, 1.0))
        w[BASE] += (n_act == 0).astype(float)
    else:
        raise ValueError(mode)

    turnover = w.diff().abs().sum(axis=1)
    turnover.iloc[0] = w.iloc[0].abs().sum()
    cost = turnover * COST_SIDE
    gross = (w * ret).sum(axis=1)
    net = gross - cost
    return {"ret": ret, "sig": sig, "held": held, "w": w, "net": net, "gross": gross,
            "turnover": turnover, "cost": cost}


# ---------------------------------------------------------------- Metriken
def underwater_days(r: pd.Series) -> int:
    eq = (1 + r).cumprod()
    uw = (eq / eq.cummax() - 1 < -1e-9).to_numpy()
    longest = cur = 0
    for f in uw:
        cur = cur + 1 if f else 0
        longest = max(longest, cur)
    return longest


def full_metrics(r: pd.Series, bench: pd.Series) -> dict:
    m = compute_metrics(r, risk_free_annual=RF)
    ex = r - bench
    cov = np.cov(r, bench, ddof=1)
    beta = cov[0, 1] / cov[1, 1]
    rf_d = (1 + RF) ** (1 / TDAYS) - 1
    alpha_d = (r.mean() - rf_d) - beta * (bench.mean() - rf_d)
    te = ex.std(ddof=1) * np.sqrt(TDAYS)
    up, dn = bench > 0, bench < 0
    q = {
        "Total Return": m["total_return"], "CAGR": m["cagr"], "Ann. Vol": m["annual_volatility"],
        "Sharpe (rf 2%)": m["sharpe"], "Sortino (rf 2%)": m["sortino"], "Calmar": m["calmar"],
        "Max Drawdown": m["max_drawdown"], "Max-DD-Dauer (Tage)": m["max_drawdown_duration_days"],
        "Bester Tag": r.max(), "Schlechtester Tag": r.min(),
        "Bester Monat": (1 + r).resample("ME").prod().sub(1).max(),
        "Schlechtester Monat": (1 + r).resample("ME").prod().sub(1).min(),
        "% positive Tage": (r > 0).mean(),
        "Skew (täglich)": r.skew(), "Kurtosis (täglich, excess)": r.kurt(),
        "VaR 95% (täglich)": r.quantile(0.05),
        "CVaR 95% (täglich)": r[r <= r.quantile(0.05)].mean(),
        "Beta zu SPY": beta, "Korrelation zu SPY": r.corr(bench),
        "Alpha p.a. (CAPM)": alpha_d * TDAYS,
        "Excess CAGR vs SPY": m["cagr"] - compute_metrics(bench, RF)["cagr"],
        "Tracking Error": te,
        "Information Ratio": ex.mean() * TDAYS / te if te > 0 else np.nan,
        "Upside Capture": r[up].mean() / bench[up].mean(),
        "Downside Capture": r[dn].mean() / bench[dn].mean(),
    }
    return q


def yearly(r: pd.Series) -> pd.Series:
    return (1 + r).groupby(r.index.year).prod() - 1


def rolling_outperf(r: pd.Series, b: pd.Series, years: int) -> dict:
    n = years * TDAYS
    ra = (1 + r).rolling(n).apply(np.prod, raw=True) - 1
    rb = (1 + b).rolling(n).apply(np.prod, raw=True) - 1
    d = (ra - rb).dropna()
    return {"Anteil Fenster Overlay > SPY": float((d > 0).mean()),
            "Ø Excess": float(d.mean()), "Min Excess": float(d.min()), "Max Excess": float(d.max())}


def fmt(k: str, v: float) -> str:
    if k in ("Sharpe (rf 2%)", "Sortino (rf 2%)", "Calmar", "Beta zu SPY", "Korrelation zu SPY",
             "Information Ratio", "Skew (täglich)", "Kurtosis (täglich, excess)"):
        return f"{v:8.2f}"
    if "Tage" in k and "%" not in k:
        return f"{v:8.0f}"
    if k in ("Upside Capture", "Downside Capture"):
        return f"{v * 100:7.1f}%"
    return f"{v * 100:+7.2f}%"


def main() -> None:
    all_t = sorted({t for u in UNIVERSES.values() for t in u})
    px = {t: get_prices(t, start=START, force_refresh=True) for t in all_t}
    for t in all_t:
        print(f"  {t:4} {px[t].index[0].date()} .. {px[t].index[-1].date()}")
    sig_cache = {t: build_signal(px[t]) for t in all_t}

    bench_full = px[BASE]["Close"].pct_change().fillna(0.0)
    bench = bench_full[bench_full.index > FIRST]
    m_bh = full_metrics(bench, bench)

    results = {}
    for uname, assets in UNIVERSES.items():
        for mode, mlab in MODES.items():
            o = build_overlay(px, assets, mode, sig_cache)
            r = o["net"][o["net"].index > FIRST]
            results[(uname, mode)] = {"o": o, "r": r, "m": full_metrics(r, bench)}

    print("\n" + "=" * 118)
    print(f"RSI(2)-Overlay auf SPY | {bench.index[0].date()} .. {bench.index[-1].date()} | {len(bench)} Handelstage")
    print("=" * 118)

    # ---- Vergleichstabelle
    cols = ["CAGR", "Excess CAGR vs SPY", "Ann. Vol", "Sharpe (rf 2%)", "Sortino (rf 2%)", "Calmar",
            "Max Drawdown", "Max-DD-Dauer (Tage)", "Alpha p.a. (CAPM)", "Beta zu SPY", "Tracking Error",
            "Information Ratio", "Upside Capture", "Downside Capture", "Schlechtester Monat"]
    short = ["CAGR", "ExcCAGR", "Vol", "Sharpe", "Sortino", "Calmar", "MaxDD", "DD-Tage", "Alpha", "Beta",
             "TE", "IR", "UpCap", "DnCap", "WorstMon"]
    print(f"  {'Variante':36}" + "".join(f"{c:>9}" for c in short))
    def row(lab, m):
        cells = []
        for c in cols:
            v = m[c]
            if c in ("Sharpe (rf 2%)", "Sortino (rf 2%)", "Calmar", "Beta zu SPY", "Information Ratio"):
                cells.append(f"{v:>9.2f}")
            elif c == "Max-DD-Dauer (Tage)":
                cells.append(f"{v:>9.0f}")
            elif "Capture" in c:
                cells.append(f"{v * 100:>8.1f}%")
            else:
                cells.append(f"{v * 100:>+8.2f}%")
        print(f"  {lab:36}" + "".join(cells))
    row("SPY Buy & Hold", m_bh)
    for (uname, mode), d in results.items():
        row(f"{uname} | {MODES[mode]}", d["m"])

    # ---- Aktivität / Kosten
    print("\n--- Aktivität & Kosten ---")
    print(f"  {'Variante':36}{'%Ø nicht-SPY':>13}{'%Tage 100%SPY':>15}{'Turnover/J':>12}{'Kosten p.a.':>13}"
          f"{'CAGR brutto':>13}{'Exc. brutto':>13}")
    for (uname, mode), d in results.items():
        o, r = d["o"], d["r"]
        w = o["w"].loc[r.index]
        yrs = len(r) / TDAYS
        mg = compute_metrics(o["gross"].loc[r.index], RF)
        print(f"  {uname + ' | ' + MODES[mode]:36}{(1 - w[BASE]).mean() * 100:>12.1f}%"
              f"{(w[BASE] > 1 - 1e-9).mean() * 100:>14.1f}%{o['turnover'].loc[r.index].sum() / yrs:>11.1f}x"
              f"{o['cost'].loc[r.index].sum() / yrs * 100:>12.3f}%{mg['cagr'] * 100:>+12.2f}%"
              f"{(mg['cagr'] - m_bh['CAGR']) * 100:>+12.2f}%")

    # ---- Kalenderjahre (Excess vs SPY)
    yb = yearly(bench)
    labs = list(results)
    ydf = pd.DataFrame({"SPY": yb})
    for k in labs:
        ydf[f"{k[0]}|{k[1]}"] = yearly(results[k]["r"]) - yb
    print("\n--- Kalenderjahre: SPY-Rendite und Excess der Varianten (%-Punkte) ---")
    print(f"  {'Jahr':6}{'SPY':>8}" + "".join(f"{(str(i + 1)):>8}" for i in range(len(labs))))
    for i, k in enumerate(labs):
        print(f"    ({i + 1}) = {k[0]} | {MODES[k[1]]}")
    for y, rw in ydf.iterrows():
        print(f"  {y:<6}{rw['SPY'] * 100:>+7.1f}%" + "".join(f"{v * 100:>+7.1f}%" for v in rw.iloc[1:]))
    print(f"  {'Jahre>SPY':14}" + "".join(f"{int((ydf.iloc[:, i + 1] > 0).sum()):>5}/{len(ydf)}" for i in range(len(labs))))

    # ---- Rolling
    print("\n--- Rolling: Anteil der Fenster mit Overlay > SPY / Ø Excess ---")
    print(f"  {'Variante':36}" + "".join(f"{str(y) + 'J':>16}" for y in (1, 3, 5, 10)))
    for k, d in results.items():
        cells = []
        for yrs in (1, 3, 5, 10):
            x = rolling_outperf(d["r"], bench, yrs)
            cells.append(f"{x['Anteil Fenster Overlay > SPY'] * 100:>6.1f}%/{x['Ø Excess'] * 100:>+6.2f}%")
        print(f"  {k[0] + ' | ' + MODES[k[1]]:36}" + "".join(f"{c:>16}" for c in cells))

    # ---- Zeit-Split
    mid = bench.index[len(bench) // 2]
    print(f"\n--- Zeit-Split an {mid.date()}: CAGR / Sharpe (1. Hälfte | 2. Hälfte) ---")
    for lab, r in [("SPY Buy & Hold", bench)] + [(f"{k[0]} | {MODES[k[1]]}", d["r"]) for k, d in results.items()]:
        a = compute_metrics(r[r.index <= mid], RF)
        b = compute_metrics(r[r.index > mid], RF)
        print(f"  {lab:36}{a['cagr'] * 100:>+7.2f}% / {a['sharpe']:>5.2f}   |   {b['cagr'] * 100:>+7.2f}% / {b['sharpe']:>5.2f}")

    # ---- Trades pro Asset (größtes Universum), Signal ist unabhängig vom Modus
    big = max(UNIVERSES.values(), key=len)
    o = results[(max(UNIVERSES, key=lambda u: len(UNIVERSES[u])), "equal")]["o"]
    ridx = bench.index
    print("\n--- Strategy-Trades pro Asset (größtes Universum; vs. SPY in derselben Haltedauer) ---")
    print(f"  {'Asset':6}{'Trades':>7}{'Win%':>7}{'Ø Hold':>8}{'Ø Trade':>9}{'Ø Trade vs SPY':>16}{'%Zeit aktiv':>13}")
    for t in big:
        h = o["held"][t].loc[ridx]
        rt, rs = o["ret"][t].loc[ridx], o["ret"][BASE].loc[ridx]
        grp = (h.diff().fillna(h) != 0).cumsum()
        tr = pd.DataFrame([{"hold": len(g), "ret": (1 + rt.loc[g.index]).prod() - 1,
                            "spy": (1 + rs.loc[g.index]).prod() - 1}
                           for _, g in h[h > 0].groupby(grp[h > 0])])
        if tr.empty:
            continue
        print(f"  {t:6}{len(tr):>7}{(tr['ret'] > 0).mean() * 100:>6.0f}%{tr['hold'].mean():>8.1f}"
              f"{tr['ret'].mean() * 100:>8.2f}%{(tr['ret'] - tr['spy']).mean() * 100:>15.2f}%{h.mean() * 100:>12.1f}%")

    # ---- Speichern
    rows = {"SPY Buy & Hold": m_bh}
    rows.update({f"{k[0]} | {MODES[k[1]]}": d["m"] for k, d in results.items()})
    pd.DataFrame(rows).to_csv(OUT / "metrics_variants.csv")
    ydf.to_csv(OUT / "yearly_excess_variants.csv")
    daily = pd.DataFrame({"SPY": bench})
    for k, d in results.items():
        daily[f"{k[0]}|{k[1]}"] = d["r"]
    daily.to_csv(OUT / "daily_variants.csv")
    plot(daily)
    print(f"\n  gespeichert in {OUT}")


def plot(daily: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eq = (1 + daily).cumprod()
    fig, ax = plt.subplots(3, 1, figsize=(13, 14), gridspec_kw={"height_ratios": [3, 1.5, 2]})
    for c in eq.columns:
        kw = dict(color="black", lw=2) if c == "SPY" else dict(lw=1.1, ls="-" if c.endswith("equal") else "--")
        ax[0].plot(eq[c], label=c, **kw)
    ax[0].set_yscale("log")
    ax[0].set_title("Equity (log)")
    ax[0].legend(fontsize=7, ncol=2)
    ax[0].grid(alpha=.3)
    for c in eq.columns:
        if c != "SPY":
            ax[1].plot(eq[c] / eq["SPY"], lw=1.1, ls="-" if c.endswith("equal") else "--", label=c)
    ax[1].set_title("Relative Equity (Variante / SPY)")
    ax[1].grid(alpha=.3)
    for c in eq.columns:
        ax[2].plot(eq[c] / eq[c].cummax() - 1, lw=1.0, color="black" if c == "SPY" else None)
    ax[2].set_title("Drawdown")
    ax[2].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT / "overlay_variants.png", dpi=130)


if __name__ == "__main__":
    main()
