"""Mean Reversion (Repo, 50%-Overlay) + Momentum, mit Crash-Filter nach Daniel & Moskowitz, "Momentum Crashes".

Paper-Bausteine
---------------
* Momentum = Rendite von t-12 bis t-1 Monat (12-1), Querschnitts-Rang unter zulässigen (PIT) Aktien.
* Bear-Market-Indikator I_B (Paper Seite 17): 1, wenn die kumulierte Marktrendite der letzten 24 Monate vor Monat t
  negativ ist, sonst 0. Hier: SPY (Total Return), Monatsende-Info, wirkt ab dem nächsten Handelstag.
* Befund: WML (Winner minus Loser) crasht in Bear-Zuständen (Loser steigen stark, Option-Charakter), Alpha dort negativ.

Varianten (alle mit Repo-MR-Sleeve, 50 % Skalierung, 5 bps/Seite; Konventionen wie sp500_research.py)
-------------------------------------------------------------------------------------------------------
 M0  Basis: 50%-Overlay, Idle = SPY                                              (bisher beste Variante)
 M1  50%-Overlay, Idle = Momentum-Buch (Top-Dezil 12-1, gleichgew., monatlich)    (ohne Crash-Filter)
 M2  wie M1, aber im Bear-Zustand (I_B = 1) Idle = SPY statt Momentum-Buch        (Crash-Filter)
 M3  MR-Einstieg nur bei Gewinnern (Momentum-Rang >= 0,5), Idle = SPY             (Momentum als MR-Filter)
 M4  wie M3, im Bear-Zustand gespiegelt (Rang <= 0,5; Verlierer erholen sich)     (Paper: Loser-Rebound im Bear)
 M5  SPY + 0,5 * WML(0,5 long / 0,5 short), ohne Filter                           (Portable Alpha)
 M6  SPY + 0,5 * WML, im Bear-Zustand aus                                         (Crash-Filter)
 M7  M2 + M6 kombiniert (Momentum-Buch als Idle + WML-Overlay mit Crash-Filter)

Vorab festgelegt, nicht auf IS/OOS getunt. IS 2000-2012, OOS 2013-2026.

Run:  .venv/bin/python momentum_crash_mr.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from momentum_overlay import seg_metrics, sharpe_diff_bootstrap, yearly
from sp500_research import (COST_IDX, COST_STOCK, EVAL_START, HDR, IS_END, TDAYS, cs_book, fmt_row, load, month_ends,
                            run, sleeve, state_machine)

HERE = Path(__file__).resolve().parent
OUT = HERE / "sp500_results"
SCALE = 0.5


def mr_state_masked(close, elig, mask=None, length=10, entry=-8.0, exit_=0.0, trend=200, top_q=0.02, min_hold=3):
    """Repo-MR-Long-Signal; `mask` (bool-DataFrame) schränkt nur die ENTRIES ein."""
    score = (close.pct_change(length, fill_method=None) * 100).where(elig)
    ok = close > close.rolling(trend).mean()
    rank = score.rank(axis=1, pct=True)
    ent = (score < entry) & ok & (rank <= top_q)
    if mask is not None:
        ent = ent & mask
    force = ~ok | score.isna()
    return state_machine(ent.values, (score > exit_).values, force.values, min_hold, ent.index, ent.columns)


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
    spy_c = c["SPY"].reindex(idx)
    me = month_ends(idx)

    # ---- Bear-Indikator (24 Monate ~ 504 Handelstage), Info am Monatsende -> wirkt ab nächstem Tag
    ret24 = spy_c / spy_c.shift(504) - 1.0
    bear_m = (ret24.loc[me] < 0).reindex(idx).ffill()
    bear = bear_m.shift(1, fill_value=False).astype(bool)           # gehaltener Zustand je Handelstag
    print(f"Bear-Zustand (24M-Marktrendite < 0): {bear.loc[EVAL_START:].mean() * 100:.1f}% der Tage seit 2000")

    # ---- Momentum
    mom = close.shift(21) / close.shift(252) - 1.0
    mom_rank = mom.where(el).rank(axis=1, pct=True)
    mom_long = cs_book(mom, el, 0.10, "high")
    mom_short = cs_book(mom, el, 0.10, "low")

    def Z():
        return pd.DataFrame(0.0, index=idx, columns=R_all.columns)

    def assemble(sl, idle_stock_w=None, idle_gate=None):
        """MR-Sleeve (Gewichte auf Aktien, Summe <= SCALE); Idle = SPY oder Momentum-Buch (optional nur wenn gate True)."""
        w = Z()
        w[stocks] = sl[stocks].reindex(idx).fillna(0.0)
        used = sl.sum(axis=1).reindex(idx).fillna(0.0)
        idle = 1.0 - used
        if idle_stock_w is None:
            w["SPY"] = idle
        else:
            g = pd.Series(1.0, index=idx) if idle_gate is None else idle_gate.astype(float)
            w[stocks] = w[stocks] + idle_stock_w[stocks].reindex(idx).fillna(0.0).mul(idle * g, axis=0)
            w["SPY"] = idle * (1.0 - g)
        return w

    base_state = mr_state_masked(close, el)
    sl_base = sleeve(base_state, "all_in") * SCALE
    st_winner = mr_state_masked(close, el, mask=(mom_rank >= 0.5))
    mask_flip = pd.DataFrame(np.where(bear.shift(1, fill_value=False).to_numpy()[:, None], (mom_rank <= 0.5).values,
                                      (mom_rank >= 0.5).values), index=idx, columns=stocks)
    # Maske gilt für Signal am Close t-1  -> Bear-Zustand von t-1: bear.shift(1) (konsistent mit Signal-Zeitpunkt)
    st_flip = mr_state_masked(close, el, mask=mask_flip)

    mw = Z(); mw[stocks] = mom_long
    wml = Z(); wml[stocks] = 0.5 * mom_long - 0.5 * mom_short
    spy_only = Z(); spy_only["SPY"] = 1.0
    gate_normal = ~bear                                              # Momentum-Buch nur im Nicht-Bear-Zustand

    strat = {
        "SPY Buy & Hold": spy_only,
        "M0 MR-Overlay 50% (Idle SPY)": assemble(sl_base),
        "M1 MR 50% + Idle Momentum-Buch": assemble(sl_base, mom_long),
        "M2 MR 50% + Idle Momentum (Crash-Filter)": assemble(sl_base, mom_long, gate_normal),
        "M3 MR 50% nur Gewinner (Idle SPY)": assemble(sleeve(st_winner, "all_in") * SCALE),
        "M4 MR 50% Gewinner / Bear: Verlierer": assemble(sleeve(st_flip, "all_in") * SCALE),
        "M5 SPY + 0,5 WML (ohne Filter)": spy_only + wml * 1.0,
        "M6 SPY + 0,5 WML (Crash-Filter)": spy_only + wml.mul((~bear).astype(float), axis=0),
        "M7 M2 + WML-Overlay (Crash-Filter)": assemble(sl_base, mom_long, gate_normal) + wml.mul((~bear).astype(float), axis=0),
    }
    port = {k: run(w, R_all, rf, costs) for k, w in strat.items()}

    bn = "SPY Buy & Hold"
    ix_all = idx[idx >= EVAL_START]
    segs = {"IS 2000-2012": ix_all[ix_all <= IS_END], "OOS 2013-2026": ix_all[ix_all > IS_END], "VOLL 2000-2026": ix_all}
    res = {}
    for sname, ix in segs.items():
        print("\n" + "=" * 140)
        print(f"{sname}: {ix[0].date()} .. {ix[-1].date()} ({len(ix) / TDAYS:.1f} J.) | Benchmark = SPY | Sharpe über T-Bill")
        print("=" * 140)
        print(f"  {'Strategie':44}{HDR}{'Sortino':>9}{'Calmar':>8}")
        for k, p in port.items():
            m = seg_metrics(p["ret"].loc[ix], port[bn]["ret"].loc[ix], rf.loc[ix])
            res[(sname, k)] = m
            print(f"  {k:44}{fmt_row(m)}{m['Sortino']:>9.2f}{m['Calmar']:>8.2f}")

    print("\n--- Exposure & Kosten (2000-2026) ---")
    print(f"  {'Strategie':44}{'Ø netto':>8}{'Ø brutto':>9}{'max brutto':>11}{'Turnover/J':>11}{'Kosten p.a.':>12}")
    for k, p in port.items():
        ww = p["w"].loc[ix_all]
        print(f"  {k:44}{ww.sum(axis=1).mean() * 100:>7.0f}%{ww.abs().sum(axis=1).mean() * 100:>8.0f}%"
              f"{ww.abs().sum(axis=1).max() * 100:>10.0f}%{p['turnover'].loc[ix_all].sum() / (len(ix_all) / TDAYS):>10.1f}x"
              f"{p['cost'].loc[ix_all].sum() / (len(ix_all) / TDAYS) * 100:>11.3f}%")

    print("\n--- Δ Sharpe vs. SPY (Block-Bootstrap 20 Tage, 5000 Ziehungen) ---")
    print(f"  {'Strategie':44}{'IS Δ':>8}{'P':>7}{'OOS Δ':>9}{'95%-KI OOS':>20}{'P':>7}")
    for k in port:
        if k == bn:
            continue
        a = sharpe_diff_bootstrap(port[k]["ret"].loc[segs["IS 2000-2012"]], port[bn]["ret"].loc[segs["IS 2000-2012"]], rf.loc[segs["IS 2000-2012"]])
        b = sharpe_diff_bootstrap(port[k]["ret"].loc[segs["OOS 2013-2026"]], port[bn]["ret"].loc[segs["OOS 2013-2026"]], rf.loc[segs["OOS 2013-2026"]])
        print(f"  {k:44}{a['obs']:>+8.2f}{a['p_le0']:>7.3f}{b['obs']:>+9.2f}   [{b['lo']:+6.2f}, {b['hi']:+6.2f}]{b['p_le0']:>7.3f}")

    # ---- Momentum-Crash-Diagnose: reiner WML (Dezil long - Dezil short), Monatsrenditen
    wml_pure = Z(); wml_pure[stocks] = mom_long - mom_short
    r_wml = run(wml_pure, R_all, pd.Series(0.0, index=idx), costs)["ret"].loc[ix_all]
    mret = (1 + r_wml).resample("ME").prod() - 1
    bear_month = bear.loc[ix_all].resample("ME").last()
    filt = mret.where(~bear_month, 0.0)
    print("\n--- Momentum-Crash-Diagnose (reiner WML: Top-Dezil - Bottom-Dezil, EW, netto Kosten) ---")
    for lab, s in (("WML ohne Filter", mret), ("WML mit Crash-Filter (Bear -> 0)", filt)):
        ann = s.mean() * 12
        sh = s.mean() / s.std() * np.sqrt(12)
        worst = ", ".join("%s %+.0f%%" % (d.strftime("%Y-%m"), v * 100) for d, v in s.nsmallest(5).items())
        print(f"  {lab:34} Ø {ann * 100:+6.1f}%/J  Sharpe {sh:5.2f}  Schiefe {s.skew():+5.2f}  schlechtester Monat {s.min() * 100:+6.1f}%  "
              f"5 schlechteste Monate: {worst}")
    print(f"  Bear-Monate: {int(bear_month.sum())} von {len(bear_month)} ({bear_month.mean() * 100:.0f}%); "
          f"Ø WML-Monatsrendite im Bear {mret[bear_month].mean() * 100:+.2f}% vs. Nicht-Bear {mret[~bear_month].mean() * 100:+.2f}%")

    # ---- Kalenderjahre (Excess vs SPY) der Hauptvarianten
    yb = yearly(port[bn]["ret"].loc[ix_all])
    show = [k for k in port if k != bn]
    ys = {k: yearly(port[k]["ret"].loc[ix_all]) - yb for k in show}
    print("\n--- Kalenderjahre: Excess vs SPY (%-Punkte) ---")
    print(f"  {'Jahr':6}{'SPY':>8}" + "".join(f"{k.split()[0]:>6}" for k in show))
    for y in yb.index:
        print(f"  {y:<6}{yb[y] * 100:>+7.1f}%" + "".join(f"{ys[k][y] * 100:>+6.1f}" for k in show))
    print(f"  {'>SPY':14}" + "".join(f"{int((ys[k] > 0).sum()):>6d}" for k in show) + f"   (von {len(yb)})")

    pd.DataFrame([{"Segment": s, "Strategie": k, **m} for (s, k), m in res.items()]).to_csv(OUT / "momentum_crash_mr_metrics.csv", index=False)
    pd.DataFrame({k: v["ret"] for k, v in port.items()}).to_csv(OUT / "momentum_crash_mr_daily.csv")
    plot(port, ix_all, res)
    print(f"\n  gespeichert in {OUT}")


def plot(port, ix, res) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(13, 10), gridspec_kw={"height_ratios": [3, 1.6]})
    for k in port:
        e = (1 + port[k]["ret"].loc[ix]).cumprod()
        sp = k.startswith("SPY")
        ax[0].plot(e, label=k, lw=2.2 if sp else 1.0, color="black" if sp else None)
        ax[1].plot(e / e.cummax() - 1, lw=0.9, color="black" if sp else None)
    ax[0].set_yscale("log")
    ax[0].legend(fontsize=7)
    ax[0].set_title("Equity (log) — rote Linie: Beginn OOS (2013)")
    ax[1].set_title("Drawdown")
    for a in ax:
        a.axvline(IS_END, color="red", ls=":")
        a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT / "momentum_crash_mr.png", dpi=130)


if __name__ == "__main__":
    main()
