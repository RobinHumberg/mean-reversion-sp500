"""50%-Overlay der Repo-Mean-Reversion-Strategie auf den SPY.

Regel
-----
Repo-Signal unverändert (10-Tage-Rendite < -8 %, Close > SMA200, Top-2 % überverkauft, Exit 10-Tage-Rendite > 0,
Mindesthaltedauer 3 Tage, PIT-Universum). Statt das gesamte Kapital in die aktiven Aktien zu legen ("all-in"),
bekommt der MR-Sleeve höchstens `scale` (hier 50 %) des Kapitals:

    w_Aktie_i(t) = scale * pos_i(t) / n(t)        (n(t) = Anzahl aktiver Signale, pos aus Signal t-1)
    w_SPY(t)     = 1 - scale * 1[n(t) > 0]        (kein Signal -> 100 % SPY)

Skalierung scale = 0,5 ist VORAB festgelegt (README-Vorschlag); 0,25 / 0,75 / 1,0 nur zur Sensitivität.
Konventionen/Kosten/Zeiträume wie in sp500_research.py.

Run:  .venv/bin/python overlay50.py
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from momentum_overlay import seg_metrics, sharpe_diff_bootstrap, yearly
from sp500_research import (COST_IDX, COST_STOCK, EVAL_START, IS_END, TDAYS, HDR, fmt_row, load, mr_state, run,
                            sleeve)

HERE = Path(__file__).resolve().parent
OUT = HERE / "sp500_results"
SCALE = 0.5


def main() -> None:
    o, c, rf_full, elig, stocks = load()
    R = (o.shift(-1) / o - 1.0).iloc[:-1]
    R_stock = R[stocks].where(elig.reindex(R.index))
    R_all = R_stock.copy()
    R_all["SPY"] = R["SPY"]
    idx = R_all.index
    rf = rf_full.reindex(idx)
    costs = pd.Series(COST_STOCK, index=R_all.columns)
    costs["SPY"] = COST_IDX
    close, el = c[stocks].reindex(idx), elig.reindex(idx)

    def build(state, scale, k=None):
        sl = sleeve(state, "all_in") * scale
        w = pd.DataFrame(0.0, index=idx, columns=R_all.columns)
        w[stocks] = sl[stocks].reindex(idx).fillna(0.0)
        w["SPY"] = 1.0 - sl.sum(axis=1).reindex(idx).fillna(0.0)
        return w

    state = mr_state(close, el)  # Repo-Standard
    spy_w = pd.DataFrame(0.0, index=idx, columns=R_all.columns)
    spy_w["SPY"] = 1.0
    port = {"SPY Buy & Hold": run(spy_w, R_all, rf, costs)}
    for s in (0.25, 0.5, 0.75, 1.0):
        port[f"MR-Overlay {int(s * 100)}%"] = run(build(state, s), R_all, rf, costs)

    head = f"MR-Overlay {int(SCALE * 100)}%"
    bn = "SPY Buy & Hold"
    ix_all = idx[idx >= EVAL_START]
    segs = {"IS 2000-2012": ix_all[ix_all <= IS_END], "OOS 2013-2026": ix_all[ix_all > IS_END], "VOLL 2000-2026": ix_all}

    res = {}
    for sname, ix in segs.items():
        print("\n" + "=" * 128)
        print(f"{sname}: {ix[0].date()} .. {ix[-1].date()} ({len(ix) / TDAYS:.1f} J.) | Benchmark = SPY | Sharpe über T-Bill")
        print("=" * 128)
        print(f"  {'Strategie':30}{HDR}{'Sortino':>9}{'Calmar':>8}{'UpCap':>7}{'DnCap':>7}")
        for k, p in port.items():
            m = seg_metrics(p["ret"].loc[ix], port[bn]["ret"].loc[ix], rf.loc[ix])
            res[(sname, k)] = m
            print(f"  {k:30}{fmt_row(m)}{m['Sortino']:>9.2f}{m['Calmar']:>8.2f}{m['UpCap'] * 100:>6.0f}%{m['DnCap'] * 100:>6.0f}%")

    print("\n--- Aktivität & Kosten (2000-2026) ---")
    print(f"  {'Strategie':30}{'Ø Aktien-Anteil':>16}{'Ø #Aktien (aktiv)':>19}{'Turnover/J':>12}{'Kosten p.a.':>12}")
    for k, p in port.items():
        ww = p["w"].loc[ix_all]
        stw = ww[stocks]
        act = (stw > 1e-9).sum(axis=1)
        print(f"  {k:30}{stw.sum(axis=1).mean() * 100:>15.1f}%{act[act > 0].mean() if (act > 0).any() else 0:>19.1f}"
              f"{p['turnover'].loc[ix_all].sum() / (len(ix_all) / TDAYS):>11.1f}x"
              f"{p['cost'].loc[ix_all].sum() / (len(ix_all) / TDAYS) * 100:>11.3f}%")
    act_days = (port[head]["w"].loc[ix_all][stocks].sum(axis=1) > 0).mean()
    print(f"  Anteil Tage mit mindestens einem aktiven Signal: {act_days * 100:.1f}%")

    print("\n--- Δ Sharpe vs. SPY (Block-Bootstrap 20 Tage, 5000 Ziehungen; P = Anteil Ziehungen mit Δ<=0) ---")
    print(f"  {'Strategie':30}{'IS Δ':>8}{'95%-KI IS':>18}{'P':>7}{'OOS Δ':>9}{'95%-KI OOS':>19}{'P':>7}")
    for k in port:
        if k == bn:
            continue
        a = sharpe_diff_bootstrap(port[k]["ret"].loc[segs["IS 2000-2012"]], port[bn]["ret"].loc[segs["IS 2000-2012"]], rf.loc[segs["IS 2000-2012"]])
        b = sharpe_diff_bootstrap(port[k]["ret"].loc[segs["OOS 2013-2026"]], port[bn]["ret"].loc[segs["OOS 2013-2026"]], rf.loc[segs["OOS 2013-2026"]])
        print(f"  {k:30}{a['obs']:>+8.2f}  [{a['lo']:+6.2f},{a['hi']:+6.2f}]{a['p_le0']:>7.3f}"
              f"{b['obs']:>+9.2f}  [{b['lo']:+6.2f},{b['hi']:+6.2f}]{b['p_le0']:>7.3f}")

    # ---- Kostensensitivität des 50%-Overlays
    print(f"\n--- Kostensensitivität {head} (bps pro Seite auf Aktien; SPY 1 bp) ---")
    print(f"  {'bps':>5}{'IS CAGR':>10}{'IS Sharpe':>10}{'OOS CAGR':>10}{'OOS Sharpe':>11}{'OOS ExcCAGR':>12}{'OOS ΔSharpe':>12}{'P':>7}")
    w50 = build(state, SCALE)
    for bps in (0, 1, 2, 5, 10, 20):
        cs = costs.copy()
        cs[stocks] = bps / 1e4
        r = run(w50, R_all, rf, cs)["ret"]
        a = seg_metrics(r.loc[segs["IS 2000-2012"]], port[bn]["ret"].loc[segs["IS 2000-2012"]], rf.loc[segs["IS 2000-2012"]])
        b = seg_metrics(r.loc[segs["OOS 2013-2026"]], port[bn]["ret"].loc[segs["OOS 2013-2026"]], rf.loc[segs["OOS 2013-2026"]])
        bb = sharpe_diff_bootstrap(r.loc[segs["OOS 2013-2026"]], port[bn]["ret"].loc[segs["OOS 2013-2026"]], rf.loc[segs["OOS 2013-2026"]])
        print(f"  {bps:>5}{a['CAGR'] * 100:>+9.2f}%{a['Sharpe']:>10.2f}{b['CAGR'] * 100:>+9.2f}%{b['Sharpe']:>11.2f}"
              f"{b['Excess CAGR'] * 100:>+11.2f}%{bb['obs']:>+12.2f}{bb['p_le0']:>7.3f}")

    # ---- Nachbarschaft
    print(f"\n--- Nachbarschaft {head}: length 5/10 x entry -5/-8/-12 x Trend 100/200 ---")
    nb = []
    for length, entry, trend in itertools.product((5, 10), (-5.0, -8.0, -12.0), (100, 200)):
        st = mr_state(close, el, length=length, entry=entry, trend=trend)
        r = run(build(st, SCALE), R_all, rf, costs)["ret"]
        row = {"length": length, "entry": entry, "trend": trend}
        for tag, sname in (("IS", "IS 2000-2012"), ("OOS", "OOS 2013-2026")):
            ix = segs[sname]
            m = seg_metrics(r.loc[ix], port[bn]["ret"].loc[ix], rf.loc[ix])
            row[f"{tag}_cagr"], row[f"{tag}_sharpe"], row[f"{tag}_exc"], row[f"{tag}_ir"] = m["CAGR"], m["Sharpe"], m["Excess CAGR"], m["IR"]
        nb.append(row)
    nb = pd.DataFrame(nb)
    for tag in ("IS", "OOS"):
        print(f"  {tag:3}: Sharpe Ø {nb[f'{tag}_sharpe'].mean():.2f} [{nb[f'{tag}_sharpe'].min():.2f},{nb[f'{tag}_sharpe'].max():.2f}] | "
              f"Excess CAGR Ø {nb[f'{tag}_exc'].mean() * 100:+.2f}% [{nb[f'{tag}_exc'].min() * 100:+.2f},{nb[f'{tag}_exc'].max() * 100:+.2f}] | "
              f"IR Ø {nb[f'{tag}_ir'].mean():+.2f} | Excess>0: {(nb[f'{tag}_exc'] > 0).sum()}/12 | "
              f"Sharpe>SPY: {(nb[f'{tag}_sharpe'] > res[(('IS 2000-2012' if tag == 'IS' else 'OOS 2013-2026'), bn)]['Sharpe']).sum()}/12")
    print(f"  (SPY Sharpe IS {res[('IS 2000-2012', bn)]['Sharpe']:.2f} / OOS {res[('OOS 2013-2026', bn)]['Sharpe']:.2f})")

    # ---- Kalenderjahre + Drawdown-Episoden
    yb = yearly(port[bn]["ret"].loc[ix_all])
    yh = yearly(port[head]["ret"].loc[ix_all])
    print(f"\n--- Kalenderjahre: SPY vs {head} ---")
    print(f"  {'Jahr':6}{'SPY':>9}{head:>20}{'Excess':>9}")
    for y in yb.index:
        print(f"  {y:<6}{yb[y] * 100:>+8.1f}%{yh[y] * 100:>+19.1f}%{(yh[y] - yb[y]) * 100:>+8.1f}")
    print(f"  Jahre > SPY: {(yh > yb).sum()} / {len(yb)} (OOS 2013+: {(yh.loc[2013:] > yb.loc[2013:]).sum()} / {len(yb.loc[2013:])})")

    # ---- Rolling 3J-Excess
    ex = (port[head]["ret"] - port[bn]["ret"]).loc[ix_all]
    roll = (1 + port[head]["ret"].loc[ix_all]).rolling(3 * TDAYS).apply(np.prod, raw=True) - (1 + port[bn]["ret"].loc[ix_all]).rolling(3 * TDAYS).apply(np.prod, raw=True)
    roll = roll.dropna()
    print(f"\n  Rolling 3J (kumuliert): Overlay > SPY in {(roll > 0).mean() * 100:.1f}% der Fenster | Ø {roll.mean() * 100:+.1f} pp | Min {roll.min() * 100:+.1f} | Max {roll.max() * 100:+.1f}")

    pd.DataFrame([{"Segment": s, "Strategie": k, **m} for (s, k), m in res.items()]).to_csv(OUT / "overlay50_metrics.csv", index=False)
    pd.DataFrame({k: v["ret"] for k, v in port.items()}).to_csv(OUT / "overlay50_daily.csv")
    nb.to_csv(OUT / "overlay50_neighborhood.csv", index=False)
    plot(port, ix_all, head, bn)
    print(f"\n  gespeichert in {OUT}")


def plot(port, ix, head, bn) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1, figsize=(13, 13), gridspec_kw={"height_ratios": [3, 1.5, 1.8]})
    for k, st in ((bn, dict(color="black", lw=2)), ("MR-Overlay 25%", dict(color="#95A5A6", lw=1)),
                  (head, dict(color="#2E86C1", lw=1.8)), ("MR-Overlay 100%", dict(color="#E67E22", lw=1, ls="--"))):
        e = (1 + port[k]["ret"].loc[ix]).cumprod()
        ax[0].plot(e, label=k, **st)
        ax[1].plot(e / e.cummax() - 1, lw=1, color=st["color"])
    ax[0].set_yscale("log")
    ax[0].legend()
    ax[0].set_title("Equity (log) — rote Linie: Beginn OOS (2013)")
    ax[1].set_title("Drawdown")
    rel = (1 + port[head]["ret"].loc[ix]).cumprod() / (1 + port[bn]["ret"].loc[ix]).cumprod()
    ax[2].plot(rel, color="#2E86C1")
    ax[2].set_title(f"Relative Equity: {head} / SPY")
    for a in ax:
        a.axvline(IS_END, color="red", ls=":")
        a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT / "overlay50.png", dpi=130)


if __name__ == "__main__":
    main()
