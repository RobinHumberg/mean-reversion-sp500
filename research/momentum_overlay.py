"""RSI(2)-Strategy (run.py) + Time-Series-Momentum (Moskowitz/Ooi/Pedersen, JFE 2012) auf S&P-500-Einzelaktien.

Bausteine
---------
1) TSMOM-Buch (long-only Adaption des Papers, Eq. 5):
     Signal_i(t)  = 1[ Überrendite_i über die letzten L Monate > 0 ]          (Paper: Vorzeichen, L = 12)
     Größe_i(t)   = (1/S_t) * 40% / sigma_i(t)                                (Paper: 40%-Vol-Ziel pro Position)
     sigma_i(t)   = ex-ante Vol, EWMA der quadr. Tagesrenditen, Schwerpunkt 60 Tage (Paper Eq. 1)
     Rebalancing  = monatlich (Monatsletzter Handelstag), Halten 1 Monat (k = 12, h = 1)
     Brutto-Exposure auf 100% gedeckelt (kein Leverage); Rest = Cash (T-Bill ^IRX).
2) RSI(2)-Sleeves: jede Aktie hat 1/N_t des Kapitals. Signal aus run.py (RSI(2)<10, Close>SMA200,
   Exit Close>SMA5). Aktiv -> Sleeve-Kapital liegt in der Aktie. Flat -> Sleeve-Kapital liegt im
   "Idle-Topf" (hier: TSMOM-Buch; als Referenz: SPY oder Cash).

Look-ahead: alle Signale mit Info bis Close t, gehalten ab t+1. Kosten pro Seite auf |Δ Zielgewicht|.
Out-of-sample: IS bis 2012-12-31, OOS ab 2013-01-01. Parameter kommen aus dem Paper bzw. run.py und
werden NICHT auf IS/OOS getunt.

Run:  .venv/bin/python momentum_overlay.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "momentum_results"
OUT.mkdir(exist_ok=True)

# ------------------------------------------------------------------ Konstanten
LOOKBACK_M = 12            # Paper: k = 12 Monate
LB_DAYS = {3: 63, 6: 126, 9: 189, 12: 252}
VOL_COM = 60               # Paper: Schwerpunkt der EWMA-Gewichte = 60 Tage
PER_STOCK_VOL = 0.40       # Paper: 40% ex-ante Vol pro Position
MAX_GROSS = 1.0            # kein Leverage
TDAYS = 252
COST_STOCK = 5e-4          # 5 bps pro Seite (Einzelaktien, S&P 500)
COST_SPY = 2e-4            # 2 bps pro Seite
EVAL_START = pd.Timestamp("2000-01-01")
IS_END = pd.Timestamp("2012-12-31")
N_BOOT, BLOCK = 5000, 20


# ------------------------------------------------------------------ Daten
def load():
    px = pd.read_pickle(HERE / "data" / "daily_close_sp500.pkl").sort_index()
    irx = px.pop("^IRX").ffill().bfill()
    spy = px.pop("SPY")
    rf = (irx / 100.0 / TDAYS).clip(lower=0.0)  # Tages-Cash-Rendite (einfache Näherung)
    return px, spy, rf


# ------------------------------------------------------------------ RSI(2)-Signal (identisch zu run.py)
def rsi(close: pd.Series, period: int = 2) -> pd.Series:
    delta = close.diff()
    gain, loss = delta.clip(lower=0.0), -delta.clip(upper=0.0)
    ag = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    al = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = ag / al.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def rsi_signal(close: pd.Series, entry_th: float = 10, trend: int = 200, sma_exit: int = 5) -> pd.Series:
    close = close.dropna()
    r = rsi(close, 2).tolist()
    c = close.tolist()
    st = close.rolling(trend).mean().tolist()
    se = close.rolling(sma_exit).mean().tolist()
    sig, in_pos = [0.0] * len(c), False
    for i in range(len(c)):
        if not in_pos:
            if st[i] == st[i] and r[i] < entry_th and c[i] > st[i]:
                in_pos = True
        elif c[i] > se[i]:
            in_pos = False
        sig[i] = 1.0 if in_pos else 0.0
    return pd.Series(sig, index=close.index)


# ------------------------------------------------------------------ TSMOM-Buch
def tsmom_book(px: pd.DataFrame, rf: pd.Series, lookback_m: int = LOOKBACK_M, inv_vol: bool = True):
    """Zielgewichte (dates x stocks), bereits um 1 Tag verschoben (= gehaltene Gewichte)."""
    L = LB_DAYS[lookback_m]
    ret = px.pct_change(fill_method=None)
    sigma = ret.ewm(com=VOL_COM, min_periods=VOL_COM).std() * np.sqrt(TDAYS)
    cum_rf = (1 + rf).cumprod()
    excess = px / px.shift(L) - 1.0 - (cum_rf / cum_rf.shift(L) - 1.0).to_numpy()[:, None]
    elig = px.notna() & px.shift(L).notna() & sigma.notna()

    me = px.index.to_series().groupby([px.index.year, px.index.month]).max().to_numpy()  # Monatsletzter Handelstag
    S = elig.loc[me].sum(axis=1).replace(0, np.nan)
    long_ = (excess.loc[me] > 0) & elig.loc[me]
    if inv_vol:
        raw = (PER_STOCK_VOL / sigma.loc[me]).where(long_, 0.0).div(S, axis=0)
    else:  # gleichgewichtet unter den positiven Signalen, Exposure = Anteil positiver Aktien
        raw = long_.astype(float).div(S, axis=0)
    gross = raw.sum(axis=1)
    raw = raw.mul(np.minimum(1.0, MAX_GROSS / gross.replace(0, np.nan)).fillna(0.0), axis=0)

    w = raw.reindex(px.index).ffill().fillna(0.0)
    return w.shift(1).fillna(0.0), elig, ret


# ------------------------------------------------------------------ Portfolio-Engine
def run_portfolio(w: pd.DataFrame, ret: pd.DataFrame, rf: pd.Series, cost_vec: pd.Series) -> dict:
    """w = gehaltene Gewichte je Spalte (Summe <= 1, Rest Cash zu rf)."""
    r = ret.reindex(columns=w.columns).fillna(0.0)
    gross = (w * r).sum(axis=1)
    cash_w = 1.0 - w.sum(axis=1)
    turn = w.diff().abs()
    turn.iloc[0] = w.iloc[0].abs()
    cost = (turn * cost_vec.reindex(w.columns).to_numpy()[None, :]).sum(axis=1)
    net = gross + cash_w * rf.reindex(w.index) - cost
    return {"ret": net, "w": w, "turnover": turn.sum(axis=1), "cost": cost, "exposure": w.sum(axis=1)}


# ------------------------------------------------------------------ Metriken
def seg_metrics(r: pd.Series, bench: pd.Series, rf: pd.Series) -> dict:
    r, bench, rf = r.align(bench, join="inner")[0], bench.reindex(r.index), rf.reindex(r.index)
    ex, bex = r - rf, bench - rf
    n, yrs = len(r), len(r) / TDAYS
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    down = ex[ex < 0].std(ddof=1)
    beta = np.cov(r, bench, ddof=1)[0, 1] / bench.var(ddof=1)
    alpha = (ex.mean() - beta * bex.mean()) * TDAYS
    te = (r - bench).std(ddof=1) * np.sqrt(TDAYS)
    up, dn = bench > 0, bench < 0
    uw, longest = (dd < -1e-9).to_numpy(), 0
    cur = 0
    for f in uw:
        cur = cur + 1 if f else 0
        longest = max(longest, cur)
    bcagr = (1 + bench).prod() ** (1 / yrs) - 1
    return {
        "Total Return": eq.iloc[-1] - 1, "CAGR": cagr, "Vol": r.std(ddof=1) * np.sqrt(TDAYS),
        "Sharpe": ex.mean() / ex.std(ddof=1) * np.sqrt(TDAYS),
        "Sortino": ex.mean() / down * np.sqrt(TDAYS), "MaxDD": dd.min(),
        "Calmar": cagr / abs(dd.min()), "DD-Tage": longest,
        "Worst Day": r.min(), "Worst Month": ((1 + r).resample("ME").prod() - 1).min(),
        "Beta": beta, "Alpha p.a.": alpha, "Excess CAGR": cagr - bcagr, "TE": te,
        "IR": (r - bench).mean() * TDAYS / te if te > 0 else np.nan,
        "UpCap": r[up].mean() / bench[up].mean(), "DnCap": r[dn].mean() / bench[dn].mean(),
    }


def sharpe_diff_bootstrap(r: pd.Series, b: pd.Series, rf: pd.Series, seed: int = 0) -> dict:
    """Zirkulärer Block-Bootstrap (Block 20 Tage) auf Δ Sharpe = Sharpe(r) - Sharpe(b), gepaart."""
    ex = (r - rf).to_numpy()
    bx = (b - rf).to_numpy()
    n = len(ex)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / BLOCK))
    starts = rng.integers(0, n, size=(N_BOOT, nb))
    idx = ((starts[:, :, None] + np.arange(BLOCK)[None, None, :]) % n).reshape(N_BOOT, -1)[:, :n]
    sr = ex[idx].mean(1) / ex[idx].std(1, ddof=1)
    sb = bx[idx].mean(1) / bx[idx].std(1, ddof=1)
    d = (sr - sb) * np.sqrt(TDAYS)
    obs = (ex.mean() / ex.std(ddof=1) - bx.mean() / bx.std(ddof=1)) * np.sqrt(TDAYS)
    return {"obs": obs, "lo": np.percentile(d, 2.5), "hi": np.percentile(d, 97.5), "p_le0": float((d <= 0).mean())}


def yearly(r: pd.Series) -> pd.Series:
    return (1 + r).groupby(r.index.year).prod() - 1


FMT = {"Total Return": "pct0", "CAGR": "pct", "Vol": "pct", "Sharpe": "f2", "Sortino": "f2", "MaxDD": "pct1",
       "Calmar": "f2", "DD-Tage": "int", "Worst Day": "pct1", "Worst Month": "pct1", "Beta": "f2",
       "Alpha p.a.": "pct", "Excess CAGR": "pct", "TE": "pct", "IR": "f2", "UpCap": "pct0", "DnCap": "pct0"}


def f(k: str, v: float) -> str:
    t = FMT[k]
    if v != v:
        return "     n/a"
    return {"pct0": f"{v * 100:7.0f}%", "pct": f"{v * 100:+7.2f}%", "pct1": f"{v * 100:+7.1f}%",
            "f2": f"{v:8.2f}", "int": f"{v:8.0f}"}[t]


# ------------------------------------------------------------------ Main
def main() -> None:
    px, spy, rf = load()
    stocks = list(px.columns)
    print(f"Universum: {len(stocks)} S&P-500-Aktien | {px.index[0].date()} .. {px.index[-1].date()}")

    ret = px.pct_change(fill_method=None)
    ret["SPY"] = spy.pct_change()
    ret_full = ret.copy()

    # ---- RSI(2)-Signale je Aktie
    sig = pd.DataFrame({t: rsi_signal(px[t]) for t in stocks}).reindex(px.index).fillna(0.0)

    # ---- TSMOM-Buch (Paper-Spezifikation) + Robustheitsvarianten
    w_mom, elig, _ = tsmom_book(px, rf, 12, True)
    books = {"TSMOM 12m (Paper)": w_mom}
    for m in (3, 6, 9):
        books[f"TSMOM {m}m"] = tsmom_book(px, rf, m, True)[0]
    books["TSMOM 12m, gleichgew."] = tsmom_book(px, rf, 12, False)[0]

    # ---- RSI-Sleeves: Kapital 1/N_t je zulässiger Aktie
    E1 = elig.shift(1, fill_value=False).astype(bool)
    N = E1.sum(axis=1).replace(0, np.nan)
    held = (sig.shift(1).fillna(0.0) * E1).fillna(0.0)
    w_rsi = held.div(N, axis=0).fillna(0.0)
    idle = (1.0 - w_rsi.sum(axis=1)).clip(lower=0.0)

    cost_stock = pd.Series(COST_STOCK, index=stocks)
    cost_vec = pd.concat([cost_stock, pd.Series({"SPY": COST_SPY})])

    def with_spy(w):  # Spalte SPY anhängen
        w = w.copy()
        w["SPY"] = 0.0
        return w

    port = {}
    # Benchmarks
    w_spy = with_spy(pd.DataFrame(0.0, index=px.index, columns=stocks))
    w_spy["SPY"] = 1.0
    port["SPY Buy & Hold"] = run_portfolio(w_spy, ret_full, rf, cost_vec)
    ew = E1.astype(float).div(N, axis=0).fillna(0.0)
    port["EW-Survivors (Benchmark)"] = run_portfolio(with_spy(ew), ret_full, rf, cost_vec)

    # Referenzen / Hauptstrategien
    port["RSI(2)-Sleeves, Rest Cash"] = run_portfolio(with_spy(w_rsi), ret_full, rf, cost_vec)
    w = with_spy(w_rsi)
    w["SPY"] = idle
    port["RSI(2)-Sleeves + Rest SPY (alt)"] = run_portfolio(w, ret_full, rf, cost_vec)
    for name, wm in books.items():
        port[name] = run_portfolio(with_spy(wm), ret_full, rf, cost_vec)
    wc = with_spy(w_rsi + wm_mul(books["TSMOM 12m (Paper)"], idle))
    port["RSI(2)-Sleeves + Rest TSMOM 12m"] = run_portfolio(wc, ret_full, rf, cost_vec)
    for m in (3, 6, 9):
        wc2 = with_spy(w_rsi + wm_mul(books[f"TSMOM {m}m"], idle))
        port[f"RSI(2)-Sleeves + Rest TSMOM {m}m"] = run_portfolio(wc2, ret_full, rf, cost_vec)

    # ---- Evaluationszeiträume
    idx_all = px.index[px.index >= EVAL_START]
    segs = {"VOLL": idx_all, "IS (bis 2012)": idx_all[idx_all <= IS_END], "OOS (ab 2013)": idx_all[idx_all > IS_END]}
    bench_name = "SPY Buy & Hold"
    HEAD = "RSI(2)-Sleeves + Rest TSMOM 12m"

    cols = ["CAGR", "Vol", "Sharpe", "Sortino", "MaxDD", "Calmar", "DD-Tage", "Worst Month", "Beta", "Alpha p.a.",
            "Excess CAGR", "TE", "IR", "UpCap", "DnCap"]
    results = {}
    for sname, ix in segs.items():
        print("\n" + "=" * 150)
        print(f"{sname}: {ix[0].date()} .. {ix[-1].date()}  ({len(ix)} Tage, {len(ix) / TDAYS:.1f} Jahre) | "
              f"Sharpe = (r - T-Bill) / Vol")
        print("=" * 150)
        print(f"  {'Strategie':36}" + "".join(f"{c[:9]:>9}" for c in cols))
        for name, p in port.items():
            m = seg_metrics(p["ret"].loc[ix], port[bench_name]["ret"].loc[ix], rf)
            results[(sname, name)] = m
            print(f"  {name:36}" + "".join(f"{f(c, m[c]):>9}" for c in cols))

    # ---- Aktivität & Kosten
    print("\n--- Aktivität & Kosten (Gesamtzeitraum ab 2000) ---")
    print(f"  {'Strategie':36}{'Ø Exposure':>11}{'Ø Cash':>8}{'Turnover/J':>11}{'Kosten p.a.':>12}{'Ø #Positionen':>15}")
    for name, p in port.items():
        ww = p["w"].loc[idx_all]
        print(f"  {name:36}{ww.sum(axis=1).mean() * 100:>10.1f}%{(1 - ww.sum(axis=1)).mean() * 100:>7.1f}%"
              f"{p['turnover'].loc[idx_all].sum() / (len(idx_all) / TDAYS):>10.1f}x"
              f"{p['cost'].loc[idx_all].sum() / (len(idx_all) / TDAYS) * 100:>11.3f}%"
              f"{(ww > 1e-9).sum(axis=1).mean():>15.0f}")

    # ---- Signifikanz OOS
    oos = segs["OOS (ab 2013)"]
    print("\n--- OOS: Δ Sharpe vs. SPY (Block-Bootstrap, 5000 Ziehungen, Block 20 Tage) ---")
    print(f"  {'Strategie':36}{'Δ Sharpe':>10}{'95%-KI':>20}{'P(Δ<=0)':>10}")
    for name in port:
        if name == bench_name:
            continue
        d = sharpe_diff_bootstrap(port[name]["ret"].loc[oos], port[bench_name]["ret"].loc[oos], rf.loc[oos])
        print(f"  {name:36}{d['obs']:>+10.2f}   [{d['lo']:+6.2f}, {d['hi']:+6.2f}]{d['p_le0']:>10.3f}")

    # ---- Vergleichsgröße: auf die Volatilität des SPY skaliert (Leverage k, Finanzierung zum T-Bill-Satz)
    # r_skaliert = rf + k * (r - rf),  k = Vol(SPY) / Vol(Strategie) im jeweiligen Segment (Ex-post-Skalierung,
    # nur als Rechengröße für den Rendite-Vergleich bei gleichem Risiko; Sharpe bleibt unverändert).
    print("\n--- Auf SPY-Volatilität skaliert (k = Vol_SPY / Vol_Strategie) ---")
    print(f"  {'Strategie':36}{'Segment':>15}{'k':>6}{'CAGR':>9}{'SPY CAGR':>10}{'Sharpe':>8}{'SPY Sharpe':>11}{'MaxDD':>9}{'SPY MaxDD':>10}")
    for name in ("TSMOM 12m (Paper)", "TSMOM 3m", "TSMOM 12m, gleichgew.", HEAD, "RSI(2)-Sleeves + Rest TSMOM 3m"):
        for sname in ("IS (bis 2012)", "OOS (ab 2013)"):
            ix = segs[sname]
            rr, bb, rff = port[name]["ret"].loc[ix], port[bench_name]["ret"].loc[ix], rf.loc[ix]
            k = bb.std() / rr.std()
            rs = rff + k * (rr - rff)
            ms, mb = seg_metrics(rs, bb, rff), seg_metrics(bb, bb, rff)
            print(f"  {name:36}{sname:>15}{k:>6.2f}{ms['CAGR'] * 100:>+8.2f}%{mb['CAGR'] * 100:>+9.2f}%"
                  f"{ms['Sharpe']:>8.2f}{mb['Sharpe']:>11.2f}{ms['MaxDD'] * 100:>+8.1f}%{mb['MaxDD'] * 100:>+9.1f}%")

    # ---- Kalenderjahre
    yb = yearly(port[bench_name]["ret"].loc[idx_all])
    show = [bench_name, "EW-Survivors (Benchmark)", "TSMOM 12m (Paper)", HEAD]
    print("\n--- Kalenderjahre ---")
    print(f"  {'Jahr':6}" + "".join(f"{s[:20]:>22}" for s in show))
    ys = {s: yearly(port[s]["ret"].loc[idx_all]) for s in show}
    for y in yb.index:
        print(f"  {y:<6}" + "".join(f"{ys[s][y] * 100:>+21.1f}%" for s in show))
    for s in show[1:]:
        print(f"  Jahre > SPY: {s:36}{(ys[s] > yb).sum()} / {len(yb)}  (OOS-Jahre ab 2013: "
              f"{(ys[s].loc[2013:] > yb.loc[2013:]).sum()} / {len(yb.loc[2013:])})")

    # ---- Speichern
    rows = []
    for (sname, name), m in results.items():
        rows.append({"Segment": sname, "Strategie": name, **m})
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame({k: v["ret"] for k, v in port.items()}).to_csv(OUT / "daily_returns.csv")
    pd.DataFrame(ys).to_csv(OUT / "yearly_returns.csv")
    plot(port, idx_all, bench_name, HEAD)
    print(f"\n  gespeichert in {OUT}")


def wm_mul(book: pd.DataFrame, idle: pd.Series) -> pd.DataFrame:
    """Idle-Kapital (Anteil) wird in das TSMOM-Buch investiert (Buch-Gewichte skaliert mit Idle)."""
    return book.mul(idle, axis=0)


def plot(port, ix, bench, head) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    show = [bench, "EW-Survivors (Benchmark)", "TSMOM 12m (Paper)", head]
    sty = {bench: dict(color="black", lw=2), show[1]: dict(color="#7F8C8D", lw=1.2, ls="--"),
           show[2]: dict(color="#E67E22", lw=1.3), head: dict(color="#2E86C1", lw=1.8)}
    fig, ax = plt.subplots(3, 1, figsize=(13, 14), gridspec_kw={"height_ratios": [3, 1.5, 1.8]})
    for s in show:
        e = (1 + port[s]["ret"].loc[ix]).cumprod()
        ax[0].plot(e, label=s, **sty[s])
        ax[1].fill_between(e.index, e / e.cummax() - 1, 0, alpha=.25, color=sty[s]["color"])
    ax[0].set_yscale("log")
    for a in ax[:2]:
        a.axvline(IS_END, color="red", ls=":", lw=1)
        a.grid(alpha=.3)
    ax[0].set_title("Equity (log) — rote Linie: Beginn OOS (2013)")
    ax[0].legend()
    ax[1].set_title("Drawdown")
    eh = (1 + port[head]["ret"].loc[ix]).cumprod()
    eb = (1 + port[bench]["ret"].loc[ix]).cumprod()
    ax[2].plot(eh / eb, color="#2E86C1")
    ax[2].axvline(IS_END, color="red", ls=":", lw=1)
    ax[2].set_title(f"Relative Equity: {head} / SPY")
    ax[2].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT / "momentum_overlay.png", dpi=130)


if __name__ == "__main__":
    main()
