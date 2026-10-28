"""Test der Mean-Reversion-Strategie aus github.com/RobinHumberg/mean-reversion-sp500 (+ Long/Short + weitere Ansätze)
gegen den S&P 500 (SPY, Total Return).

Konventionen (für ALLE Strategien identisch)
--------------------------------------------
* Signal aus Info bis Close t-1  ->  Handel am Open t  ->  Rendite Open t -> Open t+1 (repo: "next_open").
* Kosten pro Seite auf |Δ Gewicht|: Aktien 5 bps, SPY / EW-Index 1 bp (wie im Repo).
* Cash (Gewicht < 1) verzinst mit ^IRX; Leverage (Cash < 0) zahlt ^IRX + 0,5 % p.a.; Short-Leg zahlt 0,5 % p.a. Leihegebühr.
* Universum: heutige S&P-500-Mitglieder (sp500.csv), PIT-Korrektur: handelbar erst ab "Date added".
  Entfernte/delistete Aktien fehlen (Survivorship-Bias), siehe Ausgabe/README.
* Sharpe = (r - T-Bill)/Vol, annualisiert.  Zeitraum: IS = 2000-2012, OOS = 2013-2026.
  (Die Repo-Parameter wurden auf Daten ab 2012 gewählt -> für die Repo-Strategie ist 2000-2012 das echte Hold-out.)

Run:  .venv/bin/python sp500_research.py
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from momentum_overlay import seg_metrics, sharpe_diff_bootstrap, yearly

HERE = Path(__file__).resolve().parent
OUT = HERE / "sp500_results"
OUT.mkdir(exist_ok=True)

TDAYS = 252
COST_STOCK, COST_IDX = 5e-4, 1e-4
BORROW, LEV_SPREAD = 0.005, 0.005
EVAL_START, IS_END = pd.Timestamp("2000-01-01"), pd.Timestamp("2012-12-31")
SLOTS = 20


# ------------------------------------------------------------------ Daten
def load():
    d = pd.read_pickle(HERE / "data" / "sp500_ohlc_1998.pkl")
    o, c = d["open"], d["close"]
    irx = c.pop("^IRX").ffill().bfill()
    o = o.drop(columns=["^IRX"])
    meta = pd.read_csv(HERE / "data" / "mr_sp500_repo" / "sp500.csv")
    meta["Symbol"] = meta["Symbol"].str.replace(".", "-", regex=False)
    added = pd.to_datetime(meta.set_index("Symbol")["Date added"], errors="coerce")
    stocks = [t for t in c.columns if t not in ("SPY",)]
    elig = pd.DataFrame(False, index=c.index, columns=stocks)
    for t in stocks:
        a = added.get(t, pd.NaT)
        ok = c[t].notna()
        elig[t] = ok & (c.index >= a) if pd.notna(a) else ok
    rf = (irx / 100.0 / TDAYS).clip(lower=0.0)
    return o, c, rf, elig, stocks


# ------------------------------------------------------------------ MR-Signal (aus dem Repo)
def state_machine(entry, exit_, force_exit, min_hold, index, columns):
    n_days, n_st = entry.shape
    state = np.zeros((n_days, n_st), dtype=bool)
    held = np.zeros(n_st, dtype=bool)
    last_signal = np.full(n_st, -10**9, dtype=np.int64)
    for t in range(n_days):
        sig = entry[t]
        exit_now = (exit_[t] & ((t - last_signal) >= min_hold)) | force_exit[t]
        held = np.where(sig, True, np.where(exit_now, False, held))
        last_signal = np.where(sig, t, last_signal)
        state[t] = held
    return pd.DataFrame(state.astype(float), index=index, columns=columns)


def mr_state(close, elig, length=10, entry=-8.0, exit_=0.0, trend=200, top_q=0.02, min_hold=3, side="long"):
    score = (close.pct_change(length, fill_method=None) * 100).where(elig)
    sma = close.rolling(trend).mean()
    if side == "long":
        ok = close > sma
        rank = score.rank(axis=1, pct=True)
        ent, ex = (score < entry) & ok & (rank <= top_q), score > exit_
    else:  # spiegelbildlich: überkauft, unterhalb SMA, ranghöchste
        ok = close < sma
        rank = score.rank(axis=1, pct=True, ascending=False)
        ent, ex = (score > -entry) & ok & (rank <= top_q), score < -exit_
    force = ~ok | score.isna()
    return state_machine(ent.values, ex.values, force.values, min_hold, ent.index, ent.columns)


def sleeve(state: pd.DataFrame, mode: str, k: int = SLOTS) -> pd.DataFrame:
    """Gehaltene Sleeve-Gewichte (Signal t-1 -> Position t). all_in: 100% auf aktive Namen; slots: 1/max(n,K)."""
    pos = state.shift(1).fillna(0.0)
    n = pos.sum(axis=1)
    if mode == "all_in":
        return pos.div(n.replace(0, np.nan), axis=0).fillna(0.0)
    return pos.div(np.maximum(n, k), axis=0)


# ------------------------------------------------------------------ Querschnitts-Faktoren (monatlich)
def month_ends(index):
    return index.to_series().groupby([index.year, index.month]).max().to_numpy()


def cs_book(factor: pd.DataFrame, elig: pd.DataFrame, q: float, side: str) -> pd.DataFrame:
    """Monatlich: gleichgewichtet die oberen (long) bzw. unteren (short) q-Anteile des Faktors unter zulässigen Aktien."""
    me = month_ends(factor.index)
    f = factor.where(elig).loc[me]
    r = f.rank(axis=1, pct=True)
    sel = (r >= 1 - q) if side == "high" else (r <= q)
    sel = sel & f.notna()
    w = sel.astype(float).div(sel.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return w.reindex(factor.index).ffill().fillna(0.0).shift(1).fillna(0.0)


# ------------------------------------------------------------------ Engine
def run(w: pd.DataFrame, R: pd.DataFrame, rf: pd.Series, costs: pd.Series) -> dict:
    w = w.reindex(columns=R.columns).fillna(0.0)
    gross = (w * R.fillna(0.0)).sum(axis=1)
    cash = 1.0 - w.sum(axis=1)
    cash_ret = np.where(cash >= 0, cash * rf.reindex(w.index), cash * (rf.reindex(w.index) + LEV_SPREAD / TDAYS))
    short_cost = (-w.clip(upper=0.0)).sum(axis=1) * BORROW / TDAYS
    turn = w.diff().abs()
    turn.iloc[0] = w.iloc[0].abs()
    cost = (turn * costs.reindex(w.columns).to_numpy()[None, :]).sum(axis=1)
    return {"ret": gross + cash_ret - cost - short_cost, "w": w, "turnover": turn.sum(axis=1), "cost": cost}


def fmt_row(m):
    return (f"{m['CAGR'] * 100:>+8.2f}%{m['Vol'] * 100:>8.1f}%{m['Sharpe']:>8.2f}{m['MaxDD'] * 100:>+8.1f}%"
            f"{m['Beta']:>6.2f}{m['Alpha p.a.'] * 100:>+8.2f}%{m['Excess CAGR'] * 100:>+8.2f}%{m['TE'] * 100:>7.1f}%{m['IR']:>7.2f}")


HDR = f"{'CAGR':>9}{'Vol':>9}{'Sharpe':>8}{'MaxDD':>9}{'Beta':>6}{'Alpha':>9}{'ExcCAGR':>9}{'TE':>8}{'IR':>7}"


def main() -> None:
    o, c, rf_full, elig, stocks = load()
    print(f"Universum: {len(stocks)} Aktien (PIT), {c.index[0].date()} .. {c.index[-1].date()}")
    print(f"Ø zulässige Aktien: 2000: {elig.loc['2000'].sum(axis=1).mean():.0f} | 2012: {elig.loc['2012'].sum(axis=1).mean():.0f} | "
          f"2026: {elig.loc['2026'].sum(axis=1).mean():.0f}")

    # Open->Open-Renditen, Spalte t = Halten von Open t bis Open t+1
    R = o.shift(-1) / o - 1.0
    R = R.iloc[:-1]
    cols_stock = stocks
    R_stock = R[cols_stock].where(elig.reindex(R.index))
    R_all = R_stock.copy()
    R_all["SPY"] = R["SPY"]
    R_all["EWIDX"] = R_stock.mean(axis=1)
    idx = R_all.index
    rf = rf_full.reindex(idx)
    costs = pd.Series(COST_STOCK, index=R_all.columns)
    costs["SPY"], costs["EWIDX"] = COST_IDX, COST_IDX
    close = c[cols_stock].reindex(idx)
    el = elig.reindex(idx)
    spy_c = c["SPY"].reindex(idx)

    def Z():
        return pd.DataFrame(0.0, index=idx, columns=R_all.columns)

    def base(asset):
        w = Z()
        w[asset] = 1.0
        return w

    def overlay(long_sl=None, short_sl=None, idle="SPY", idle_w=None):
        """Idle-Kapital im Basis-Asset; Long-Sleeve ersetzt Idle; Short-Sleeve wird als negative Gewichte addiert."""
        w = Z()
        if long_sl is not None:
            w[cols_stock] = long_sl[cols_stock].reindex(idx).fillna(0.0)
            used = long_sl.sum(axis=1).reindex(idx).fillna(0.0)
        else:
            used = pd.Series(0.0, index=idx)
        if idle_w is not None:  # Basis ist ein Aktienbuch (z. B. Momentum)
            w[cols_stock] = w[cols_stock] + idle_w[cols_stock].reindex(idx).fillna(0.0).mul(1.0 - used, axis=0)
        else:
            w[idle] = 1.0 - used
        if short_sl is not None:
            w[cols_stock] = w[cols_stock] - short_sl[cols_stock].reindex(idx).fillna(0.0)
        return w

    # ---- Bausteine
    L_state = mr_state(close, el, side="long")
    S_state = mr_state(close, el, side="short")
    L_all, L_slot = sleeve(L_state, "all_in"), sleeve(L_state, "slots")
    S_slot = sleeve(S_state, "slots")

    ret_c = close.pct_change(fill_method=None)
    mom = close.shift(21) / close.shift(252) - 1.0                         # 12-1 Monats-Momentum
    vol126 = ret_c.rolling(126, min_periods=126).std()
    mom_long = cs_book(mom, el, 0.10, "high")
    mom_short = cs_book(mom, el, 0.10, "low")
    lowvol = cs_book(vol126, el, 0.20, "low")

    trend_state = (spy_c > spy_c.rolling(200).mean()).astype(float).shift(1).fillna(0.0)
    spy_ret = spy_c.pct_change()
    sig_m = (0.15 / (spy_ret.rolling(21).std() * np.sqrt(TDAYS))).clip(upper=1.5)
    volman = sig_m.loc[month_ends(idx)].reindex(idx).ffill().shift(1).fillna(0.0)

    strat = {}
    strat["SPY Buy & Hold"] = base("SPY")
    strat["EW-Index PIT (Repo-Benchmark)"] = base("EWIDX")

    # --- Repo-Ansatz
    strat["A  MR Repo (Idle EW-Index, all-in)"] = overlay(L_all, idle="EWIDX")
    strat["A2 MR Repo (Idle SPY, all-in)"] = overlay(L_all, idle="SPY")
    strat["A3 MR long (Idle SPY, 20 Slots)"] = overlay(L_slot, idle="SPY")
    strat["A4 MR long (Idle SPY, 10 Slots)"] = overlay(sleeve(L_state, "slots", 10), idle="SPY")
    strat["A5 MR long (Idle SPY, 5 Slots)"] = overlay(sleeve(L_state, "slots", 5), idle="SPY")
    # --- Long/Short
    strat["B  MR Long/Short (SPY-Idle, 20+20 Slots)"] = overlay(L_slot, S_slot, idle="SPY")
    mn = overlay(L_slot, S_slot, idle="SPY")
    mn_only = (L_slot.reindex(idx).fillna(0.0)).copy()
    mn_w = Z()
    mn_w[cols_stock] = L_slot[cols_stock].reindex(idx).fillna(0.0) - S_slot[cols_stock].reindex(idx).fillna(0.0)
    strat["B2 SPY + MR marktneutral (Portable Alpha)"] = base("SPY") + mn_w
    strat["B3 MR marktneutral allein (Cash)"] = mn_w
    # --- andere Ansätze
    mw = Z(); mw[cols_stock] = mom_long
    strat["C  XS-Momentum 12-1 long (Top-Dezil)"] = mw
    mls = Z(); mls[cols_stock] = 0.5 * mom_long - 0.5 * mom_short
    strat["C2 SPY + XS-Momentum L/S (0,5/0,5)"] = base("SPY") + mls
    lw = Z(); lw[cols_stock] = lowvol
    strat["D  Low-Vol long (unteres Quintil)"] = lw
    tw = Z(); tw["SPY"] = trend_state
    strat["E  SPY Trend (>SMA200 sonst Cash)"] = tw
    vw = Z(); vw["SPY"] = volman
    strat["F  SPY Vol-Target 15% (max 1,5x)"] = vw
    mom_book = Z(); mom_book[cols_stock] = mom_long
    strat["G  MR long (20 Slots) + Idle Momentum-Buch"] = overlay(L_slot, idle_w=mom_book)
    # --- Kombinationen
    strat["H  SPY + MR marktneutral + Momentum L/S"] = base("SPY") + mn_w + mls
    strat["H2 Trend-SPY + MR marktneutral + Momentum L/S"] = tw + mn_w + mls
    strat["I  Idle Momentum + MR L/S Overlay"] = overlay(L_slot, S_slot, idle_w=mom_book)

    port = {k: run(w, R_all, rf, costs) for k, w in strat.items()}

    bn = "SPY Buy & Hold"
    ix_all = idx[idx >= EVAL_START]
    segs = {"IS 2000-2012": ix_all[ix_all <= IS_END], "OOS 2013-2026": ix_all[ix_all > IS_END], "VOLL 2000-2026": ix_all}
    res = {}
    for sname, ix in segs.items():
        print("\n" + "=" * 140)
        print(f"{sname}: {ix[0].date()} .. {ix[-1].date()} ({len(ix) / TDAYS:.1f} J.) | Benchmark = SPY Total Return | Sharpe über T-Bill")
        print("=" * 140)
        print(f"  {'Strategie':46}{HDR}")
        for k, p in port.items():
            m = seg_metrics(p["ret"].loc[ix], port[bn]["ret"].loc[ix], rf.loc[ix])
            res[(sname, k)] = m
            print(f"  {k:46}{fmt_row(m)}")

    # ---- Exposure / Kosten
    print("\n--- Exposure & Kosten (2000-2026) ---")
    print(f"  {'Strategie':46}{'Ø netto':>8}{'Ø brutto':>9}{'max brutto':>11}{'Turnover/J':>11}{'Kosten p.a.':>12}")
    for k, p in port.items():
        ww = p["w"].loc[ix_all]
        print(f"  {k:46}{ww.sum(axis=1).mean() * 100:>7.0f}%{ww.abs().sum(axis=1).mean() * 100:>8.0f}%"
              f"{ww.abs().sum(axis=1).max() * 100:>10.0f}%{p['turnover'].loc[ix_all].sum() / (len(ix_all) / TDAYS):>10.1f}x"
              f"{p['cost'].loc[ix_all].sum() / (len(ix_all) / TDAYS) * 100:>11.3f}%")

    # ---- Signifikanz: Δ Sharpe vs SPY
    print("\n--- Δ Sharpe vs. SPY (Block-Bootstrap 20 Tage, 5000 Ziehungen; P = Anteil Ziehungen mit Δ<=0) ---")
    print(f"  {'Strategie':46}{'IS Δ':>8}{'P':>7}{'OOS Δ':>9}{'95%-KI OOS':>20}{'P':>7}")
    boot = {}
    for k in port:
        if k == bn:
            continue
        a = sharpe_diff_bootstrap(port[k]["ret"].loc[segs["IS 2000-2012"]], port[bn]["ret"].loc[segs["IS 2000-2012"]], rf.loc[segs["IS 2000-2012"]])
        b = sharpe_diff_bootstrap(port[k]["ret"].loc[segs["OOS 2013-2026"]], port[bn]["ret"].loc[segs["OOS 2013-2026"]], rf.loc[segs["OOS 2013-2026"]])
        boot[k] = (a, b)
        print(f"  {k:46}{a['obs']:>+8.2f}{a['p_le0']:>7.3f}{b['obs']:>+9.2f}   [{b['lo']:+6.2f}, {b['hi']:+6.2f}]{b['p_le0']:>7.3f}")

    # ---- Repo-Replikation (Zeitraum wie im Repo ab 08/2012, rf 2,5 % konstant, EW-Index als Benchmark)
    print("\n--- Repo-Replikation: ab 2012-08, Sharpe mit rf=2,5% konstant (README: Index CAGR 15,9 / Strategie 28,8 / Sharpe 0,84 / 0,88) ---")
    ixr = idx[idx >= "2012-08-01"]
    for k in ("EW-Index PIT (Repo-Benchmark)", "A  MR Repo (Idle EW-Index, all-in)", "SPY Buy & Hold", "A2 MR Repo (Idle SPY, all-in)"):
        r = port[k]["ret"].loc[ixr]
        eq = (1 + r).cumprod()
        cagr = eq.iloc[-1] ** (TDAYS / len(r)) - 1
        sh = (r.mean() - 0.025 / TDAYS) / r.std() * np.sqrt(TDAYS)
        ex = r - port["EW-Index PIT (Repo-Benchmark)"]["ret"].loc[ixr]
        print(f"  {k:46} CAGR {cagr * 100:6.1f}%  Vol {r.std() * np.sqrt(TDAYS) * 100:5.1f}%  Sharpe {sh:5.2f}  "
              f"MaxDD {(eq / eq.cummax() - 1).min() * 100:6.1f}%  IR vs EW-Index {ex.mean() / ex.std() * np.sqrt(TDAYS):5.2f}")

    # ---- Nachbarschafts-Robustheit des Repo-Ansatzes (idle SPY)
    print("\n--- Nachbarschaft Repo-MR (Idle SPY, all-in vs 20 Slots): length 5/10 x entry -5/-8/-12 x Trend 100/200 ---")
    nb = []
    for length, entry, trend in itertools.product((5, 10), (-5.0, -8.0, -12.0), (100, 200)):
        st = mr_state(close, el, length=length, entry=entry, trend=trend)
        for mode in ("all_in", "slots"):
            r = run(overlay(sleeve(st, mode), idle="SPY"), R_all, rf, costs)["ret"]
            row = {"mode": mode, "length": length, "entry": entry, "trend": trend}
            for tag, sname in (("IS", "IS 2000-2012"), ("OOS", "OOS 2013-2026")):
                ix = segs[sname]
                m = seg_metrics(r.loc[ix], port[bn]["ret"].loc[ix], rf.loc[ix])
                row[f"{tag}_sharpe"], row[f"{tag}_exc"], row[f"{tag}_ir"] = m["Sharpe"], m["Excess CAGR"], m["IR"]
            nb.append(row)
    nb = pd.DataFrame(nb)
    for mode, g in nb.groupby("mode"):
        for seg in ("IS", "OOS"):
            print(f"  {mode:7} {seg:3}: Sharpe Ø {g[f'{seg}_sharpe'].mean():.2f} [{g[f'{seg}_sharpe'].min():.2f},{g[f'{seg}_sharpe'].max():.2f}] | "
                  f"Excess CAGR Ø {g[f'{seg}_exc'].mean() * 100:+.2f}% [{g[f'{seg}_exc'].min() * 100:+.2f},{g[f'{seg}_exc'].max() * 100:+.2f}] | "
                  f"IR Ø {g[f'{seg}_ir'].mean():+.2f} | Konfigurationen mit Excess>0: {(g[f'{seg}_exc'] > 0).sum()}/12")
    print(f"  (SPY Sharpe IS {res[('IS 2000-2012', bn)]['Sharpe']:.2f} / OOS {res[('OOS 2013-2026', bn)]['Sharpe']:.2f})")

    # ---- Kostensensitivität (Aktien-Kosten pro Seite)
    print("\n--- Kostensensitivität (bps pro Seite auf Aktien; SPY/Index 1 bp) ---")
    print(f"  {'Strategie':40}{'bps':>5}  {'IS CAGR':>9}{'IS Sharpe':>10}{'OOS CAGR':>10}{'OOS Sharpe':>11}{'OOS ΔSharpe':>12}{'P':>7}")
    for k in ("A2 MR Repo (Idle SPY, all-in)", "A4 MR long (Idle SPY, 10 Slots)", "A3 MR long (Idle SPY, 20 Slots)"):
        for bps in (0, 1, 2, 5, 10):
            cs = costs.copy()
            cs[cols_stock] = bps / 1e4
            r = run(strat[k], R_all, rf, cs)["ret"]
            a = seg_metrics(r.loc[segs["IS 2000-2012"]], port[bn]["ret"].loc[segs["IS 2000-2012"]], rf.loc[segs["IS 2000-2012"]])
            b = seg_metrics(r.loc[segs["OOS 2013-2026"]], port[bn]["ret"].loc[segs["OOS 2013-2026"]], rf.loc[segs["OOS 2013-2026"]])
            bb = sharpe_diff_bootstrap(r.loc[segs["OOS 2013-2026"]], port[bn]["ret"].loc[segs["OOS 2013-2026"]], rf.loc[segs["OOS 2013-2026"]])
            print(f"  {k:40}{bps:>5}  {a['CAGR'] * 100:>+8.2f}%{a['Sharpe']:>10.2f}{b['CAGR'] * 100:>+9.2f}%{b['Sharpe']:>11.2f}{bb['obs']:>+12.2f}{bb['p_le0']:>7.3f}")

    # ---- Kalenderjahre
    yb = yearly(port[bn]["ret"].loc[ix_all])
    print("\n--- Kalenderjahre: Excess vs SPY (%-Punkte) ---")
    show = list(port)[1:]
    ys = {k: yearly(port[k]["ret"].loc[ix_all]) - yb for k in show}
    short = [k.split()[0] for k in show]
    print(f"  {'Jahr':6}{'SPY':>8}" + "".join(f"{s:>7}" for s in short))
    for y in yb.index:
        print(f"  {y:<6}{yb[y] * 100:>+7.1f}%" + "".join(f"{ys[k][y] * 100:>+7.1f}" for k in show))
    print(f"  {'>SPY':14}" + "".join(f"{int((ys[k] > 0).sum()):>7d}" for k in show) + f"   (von {len(yb)})")

    # ---- Speichern
    pd.DataFrame([{"Segment": s, "Strategie": k, **m} for (s, k), m in res.items()]).to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame({k: v["ret"] for k, v in port.items()}).to_csv(OUT / "daily_returns.csv")
    nb.to_csv(OUT / "neighborhood.csv", index=False)
    plot(port, ix_all, res, boot)
    print(f"\n  gespeichert in {OUT}")


def plot(port, ix, res, boot) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(port)
    oos = {k: res[("OOS 2013-2026", k)] for k in names}
    top = sorted([k for k in names if k != "SPY Buy & Hold"], key=lambda k: -oos[k]["Sharpe"])[:5]
    fig, ax = plt.subplots(2, 1, figsize=(13, 11), gridspec_kw={"height_ratios": [3, 1.6]})
    for k in ["SPY Buy & Hold"] + top:
        e = (1 + port[k]["ret"].loc[ix]).cumprod()
        ax[0].plot(e, label=k, lw=2.2 if k.startswith("SPY") else 1.2, color="black" if k.startswith("SPY") else None)
        ax[1].plot(e / e.cummax() - 1, lw=1.0, color="black" if k.startswith("SPY") else None)
    ax[0].set_yscale("log")
    ax[0].axvline(IS_END, color="red", ls=":")
    ax[1].axvline(IS_END, color="red", ls=":")
    ax[0].set_title("Equity (log): SPY und die 5 Strategien mit der höchsten OOS-Sharpe — rote Linie = Beginn OOS")
    ax[1].set_title("Drawdown")
    ax[0].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT / "sp500_research.png", dpi=130)


if __name__ == "__main__":
    main()
