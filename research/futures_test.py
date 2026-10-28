"""Futures-Test: RSI(2)-Mean-Reversion und Time-Series-Momentum auf einem liquiden Futures-Universum, als Overlay auf den SPY.

Warum ETF-Proxy statt Yahoo-Fortlaufendserien?
  Die Yahoo-"=F"-Serien sind nicht back-adjusted und teils defekt (NG=F: 135 Tage mit |r|>10 %, CL=F: -37,63 $ und +306 %,
  6J=F: Preis 0,00 / +904 %, RTY=F erst ab 2017). Futures-Rendite = Überrendite des Underlyings:
      r_Future  ≈  r_ETF(Total Return) - r_f        (Margin-Collateral in T-Bills verzinst mit r_f)
  Das ist roll-artefaktfrei; Kosten werden futures-typisch (0,5-1 bp pro Seite + Roll) angesetzt.
  Nicht abgebildet: Basis-/Roll-Rendite-Unterschiede einzelner Kontrakte, Margin-Rufe, Tick-Größen.

Instrument -> Proxy:  ES SPY | NQ QQQ | YM DIA | RTY IWM | MFS EFA | MES-EM EEM | ZT SHY | ZF IEI | ZN IEF | ZB TLT |
                      GC GLD | SI SLV | Rohstoff-Korb DBC | Agrar-Korb DBA | 6E FXE | 6B FXB | 6C FXC | 6A FXA | 6J FXY

Sleeves (vorab festgelegt, keine Optimierung):
  TS  = Time-Series-Momentum (Moskowitz/Ooi/Pedersen): sign(12M-Überrendite) * 40%/sigma_i / N, monatlich.
  MR  = RSI(2)-Mean-Reversion (run.py): Long RSI2<10 & Close>SMA200, Exit Close>SMA5;
        Short RSI2>90 & Close<SMA200, Exit Close<SMA5; Größe 40%/sigma_i / N bei Aktivität.
  CO  = TS + MR.
Jeder Sleeve wird ex-ante auf 10 % Vola skaliert (EWMA 60 Tage, monatlich aktualisiert, max. 4x).
Portfolios:  SPY + Sleeve (Overlay, Margin in T-Bills).

Signal Close t-1 -> Handel Open t -> Rendite Open t -> Open t+1. IS 2004-2012, OOS 2013-2026.

Run:  .venv/bin/python futures_test.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from momentum_overlay import seg_metrics, sharpe_diff_bootstrap, yearly
from sp500_research import load, month_ends

HERE = Path(__file__).resolve().parent
OUT = HERE / "sp500_results"
TDAYS = 252
PROXY = {"ES": "SPY", "NQ": "QQQ", "YM": "DIA", "RTY": "IWM", "EFA": "EFA", "EEM": "EEM",
         "ZT": "SHY", "ZF": "IEI", "ZN": "IEF", "ZB": "TLT", "GC": "GLD", "SI": "SLV", "CMDTY": "DBC", "AGRI": "DBA",
         "6E": "FXE", "6B": "FXB", "6C": "FXC", "6A": "FXA", "6J": "FXY"}
CLASS = {"ES": "eq", "NQ": "eq", "YM": "eq", "RTY": "eq", "EFA": "eq", "EEM": "eq", "ZT": "bond", "ZF": "bond",
         "ZN": "bond", "ZB": "bond", "GC": "cmd", "SI": "cmd", "CMDTY": "cmd", "AGRI": "cmd",
         "6E": "fx", "6B": "fx", "6C": "fx", "6A": "fx", "6J": "fx"}
COST_BP = {"eq": 0.5, "bond": 0.5, "cmd": 1.0, "fx": 0.5}   # pro Seite
ROLL_BP_PER_YEAR = 4.0                                      # 4 Rolls x (Ein+Aus) x 0,5 bp pro Einheit Notional
SLEEVE_VOL, LEV_CAP, VOL_COM = 0.10, 4.0, 60
EVAL_START, IS_END = pd.Timestamp("2004-01-01"), pd.Timestamp("2012-12-31")


def get_etfs():
    cache = HERE / "data" / "futures_proxy_etf.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    tk = sorted(set(PROXY.values()))
    raw = yf.download(tk, start="1999-01-01", auto_adjust=True, progress=False, threads=True)
    d = {"open": raw["Open"], "close": raw["Close"]}
    pd.to_pickle(d, cache)
    return d


def rsi2(close: pd.Series) -> pd.Series:
    d = close.diff()
    ag = d.clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
    al = (-d.clip(upper=0)).ewm(alpha=0.5, adjust=False).mean()
    return (100 - 100 / (1 + ag / al.replace(0, np.nan))).fillna(50.0)


def mr_state(close: pd.Series):
    """(long_state, short_state) 0/1 aus run.py-Regeln (spiegelbildlich für Short)."""
    s = close.dropna()
    r, c = rsi2(s).tolist(), s.tolist()
    sma200, sma5 = s.rolling(200).mean().tolist(), s.rolling(5).mean().tolist()
    lg, sh = [0.0] * len(c), [0.0] * len(c)
    L = S = False
    for i in range(len(c)):
        if not L:
            if sma200[i] == sma200[i] and r[i] < 10 and c[i] > sma200[i]:
                L = True
        elif c[i] > sma5[i]:
            L = False
        if not S:
            if sma200[i] == sma200[i] and r[i] > 90 and c[i] < sma200[i]:
                S = True
        elif c[i] < sma5[i]:
            S = False
        lg[i], sh[i] = float(L), float(S)
    return pd.Series(lg, index=s.index), pd.Series(sh, index=s.index)


def engine(w: pd.DataFrame, rex: pd.DataFrame, rf: pd.Series, cost_bp: pd.Series, cost_mult: float = 1.0):
    """ret = rf + Σ w (r_i - rf) - Handelskosten - Roll-Kosten. w = gehaltene Gewichte (bereits verzögert)."""
    w = w.reindex(columns=rex.columns).fillna(0.0)
    gross = (w * rex.fillna(0.0)).sum(axis=1)
    turn = w.diff().abs()
    turn.iloc[0] = w.iloc[0].abs()
    cost = (turn * (cost_bp.reindex(w.columns).to_numpy() * 1e-4 * cost_mult)[None, :]).sum(axis=1)
    roll = w.abs().sum(axis=1) * ROLL_BP_PER_YEAR * 1e-4 * cost_mult / TDAYS
    return rf + gross - cost - roll, turn.sum(axis=1), cost + roll


def vol_target(sleeve_ret: pd.Series, w: pd.DataFrame, target=SLEEVE_VOL):
    """Skaliert Sleeve-Gewichte monatlich mit target/EWMA-Vola (ex-ante, Faktor ab nächstem Tag)."""
    vol = sleeve_ret.ewm(com=VOL_COM, min_periods=VOL_COM).std() * np.sqrt(TDAYS)
    k = (target / vol).clip(upper=LEV_CAP)
    k = k.loc[month_ends(k.index)].reindex(k.index).ffill().shift(1).fillna(0.0)
    return w.mul(k, axis=0), k


def main() -> None:
    o_all, c_all, rf_full, _, _ = load()                         # rf (^IRX) und SPY aus dem S&P-Datensatz
    etf = get_etfs()
    eo, ec = etf["open"], etf["close"]
    inst = [k for k in PROXY]
    idx = eo.index[eo.index >= "1999-06-01"]
    R = (eo.shift(-1) / eo - 1.0).iloc[:-1]
    idx = R.index[R.index >= "1999-06-01"]
    rf = rf_full.reindex(idx).ffill()
    rex = pd.DataFrame({k: R[PROXY[k]].reindex(idx) - rf for k in inst})
    close = pd.DataFrame({k: ec[PROXY[k]].reindex(idx) for k in inst})
    avail = close.notna() & (close.notna().cumsum() > 260)
    cost_bp = pd.Series({k: COST_BP[CLASS[k]] for k in inst})

    ret_c = close.pct_change(fill_method=None)
    sigma = ret_c.ewm(com=VOL_COM, min_periods=VOL_COM).std() * np.sqrt(TDAYS)
    N = avail.shift(1, fill_value=False).sum(axis=1).replace(0, np.nan)

    # ---- TS: 12M-Überrendite (ETF-TR minus kumulierter T-Bill)
    cum_rf = (1 + rf).cumprod()
    exc12 = close / close.shift(252) - 1.0 - (cum_rf / cum_rf.shift(252) - 1.0).to_numpy()[:, None]
    me = month_ends(idx)
    sign = np.sign(exc12).where(avail & sigma.notna())
    w_ts = (sign * 0.40 / sigma).loc[me].reindex(idx).ffill().shift(1).fillna(0.0).div(N, axis=0).fillna(0.0)

    # ---- MR: RSI(2)-Zustände je Instrument
    L = pd.DataFrame({k: mr_state(close[k])[0] for k in inst}).reindex(idx).fillna(0.0)
    S = pd.DataFrame({k: mr_state(close[k])[1] for k in inst}).reindex(idx).fillna(0.0)
    size = (0.40 / sigma).where(avail & sigma.notna(), 0.0)
    w_mr = ((L - S).shift(1).fillna(0.0) * size.shift(1)).div(N, axis=0).fillna(0.0)
    w_mr_long = (L.shift(1).fillna(0.0) * size.shift(1)).div(N, axis=0).fillna(0.0)

    sleeves = {"TS  Time-Series-Momentum (L/S)": w_ts,
               "MR  RSI(2) Mean-Reversion (L/S)": w_mr,
               "MR  RSI(2) nur Long": w_mr_long,
               "CO  TS + MR (L/S)": w_ts + w_mr}

    spy_ret = R["SPY"].reindex(idx)
    # Cash-Basis & SPY (ETF, 1 bp)
    cash = pd.Series(rf.values, index=idx)
    spy_cost = 1e-4
    spy_net = spy_ret.copy()
    spy_net.iloc[0] -= spy_cost
    port = {"SPY Buy & Hold": spy_net}
    info = {}
    for name, w0 in sleeves.items():
        r0, _, _ = engine(w0, rex, rf, cost_bp)
        wv, k = vol_target(r0, w0)
        r_sl, turn, cst = engine(wv, rex, rf, cost_bp)
        sl_excess = r_sl - rf                                          # reine Überrendite des Sleeves
        port[f"{name}  [Sleeve allein, 10% Vola]"] = r_sl
        port[f"SPY + {name}"] = spy_net + sl_excess
        port[f"SPY + 0,5x {name}"] = spy_net + 0.5 * sl_excess
        info[name] = {"turn": turn, "cost": cst, "w": wv, "k": k}

    bn = "SPY Buy & Hold"
    ix_all = idx[idx >= EVAL_START]
    segs = {"IS 2004-2012": ix_all[ix_all <= IS_END], "OOS 2013-2026": ix_all[ix_all > IS_END], "VOLL 2004-2026": ix_all}
    hdr = f"{'CAGR':>9}{'Vol':>9}{'Sharpe':>8}{'MaxDD':>9}{'Beta':>6}{'Alpha':>9}{'ExcCAGR':>9}{'TE':>8}{'IR':>7}"
    res = {}
    for sname, ix in segs.items():
        print("\n" + "=" * 138)
        print(f"{sname}: {ix[0].date()} .. {ix[-1].date()} ({len(ix) / TDAYS:.1f} J.) | Benchmark = SPY | Sharpe über T-Bill")
        print("=" * 138)
        print(f"  {'Strategie':64}{hdr}")
        for k, r in port.items():
            m = seg_metrics(r.loc[ix], port[bn].loc[ix], rf.loc[ix])
            res[(sname, k)] = m
            print(f"  {k:64}{m['CAGR'] * 100:>+8.2f}%{m['Vol'] * 100:>8.1f}%{m['Sharpe']:>8.2f}{m['MaxDD'] * 100:>+8.1f}%"
                  f"{m['Beta']:>6.2f}{m['Alpha p.a.'] * 100:>+8.2f}%{m['Excess CAGR'] * 100:>+8.2f}%{m['TE'] * 100:>7.1f}%{m['IR']:>7.2f}")

    print("\n--- Sleeve-Aktivität & Kosten (2004-2026, nach Vola-Skalierung auf 10 %) ---")
    print(f"  {'Sleeve':36}{'Ø brutto':>9}{'max brutto':>11}{'Ø Hebel k':>10}{'Turnover/J':>11}{'Kosten p.a.':>12}")
    for name, d in info.items():
        w = d["w"].loc[ix_all]
        print(f"  {name:36}{w.abs().sum(axis=1).mean() * 100:>8.0f}%{w.abs().sum(axis=1).max() * 100:>10.0f}%"
              f"{d['k'].loc[ix_all].mean():>10.2f}{d['turn'].loc[ix_all].sum() / (len(ix_all) / TDAYS):>10.1f}x"
              f"{d['cost'].loc[ix_all].sum() / (len(ix_all) / TDAYS) * 100:>11.3f}%")

    print("\n--- Sleeve allein: Korrelation zu SPY und Δ Sharpe der Overlay-Portfolios (Bootstrap, OOS) ---")
    print(f"  {'Sleeve':36}{'Korr IS':>9}{'Korr OOS':>10}{'Δ Sharpe OOS: SPY+Sleeve':>26}{'P':>7}{'SPY+0,5x':>11}{'P':>7}")
    for name in sleeves:
        key = f"{name}  [Sleeve allein, 10% Vola]"
        ce = [(port[key].loc[segs[s]] - rf.loc[segs[s]]).corr(port[bn].loc[segs[s]] - rf.loc[segs[s]]) for s in ("IS 2004-2012", "OOS 2013-2026")]
        o = segs["OOS 2013-2026"]
        a = sharpe_diff_bootstrap(port[f"SPY + {name}"].loc[o], port[bn].loc[o], rf.loc[o])
        b = sharpe_diff_bootstrap(port[f"SPY + 0,5x {name}"].loc[o], port[bn].loc[o], rf.loc[o])
        print(f"  {name:36}{ce[0]:>+9.2f}{ce[1]:>+10.2f}{a['obs']:>+17.2f} [{a['lo']:+5.2f},{a['hi']:+5.2f}]{a['p_le0']:>7.3f}"
              f"{b['obs']:>+9.2f}{b['p_le0']:>7.3f}")

    # ---- Kostenmultiplikator
    print("\n--- Kostensensitivität (Faktor auf Futures-Kosten; 1,0 = 0,5-1 bp/Seite + 4 bp/J Roll) — SPY + Sleeve, OOS ---")
    print(f"  {'Sleeve':36}" + "".join(f"{'x' + str(m):>14}" for m in (0, 1, 3, 10)))
    for name, w0 in sleeves.items():
        row = []
        for mult in (0, 1, 3, 10):
            r0, _, _ = engine(w0, rex, rf, cost_bp, mult)
            wv, _ = vol_target(r0, w0)
            r_sl, _, _ = engine(wv, rex, rf, cost_bp, mult)
            r = spy_net + (r_sl - rf)
            m = seg_metrics(r.loc[segs["OOS 2013-2026"]], port[bn].loc[segs["OOS 2013-2026"]], rf.loc[segs["OOS 2013-2026"]])
            row.append(f"{m['CAGR'] * 100:+6.2f}%/{m['Sharpe']:.2f}")
        print(f"  {name:36}" + "".join(f"{x:>14}" for x in row))

    print("\n--- Kalenderjahre: Excess vs SPY (%-Punkte), SPY + Sleeve ---")
    yb = yearly(port[bn].loc[ix_all])
    show = [f"SPY + {n}" for n in sleeves]
    ys = {k: yearly(port[k].loc[ix_all]) - yb for k in show}
    print(f"  {'Jahr':6}{'SPY':>8}" + "".join(f"{k.split()[2][:5]:>8}" for k in show))
    for y in yb.index:
        print(f"  {y:<6}{yb[y] * 100:>+7.1f}%" + "".join(f"{ys[k][y] * 100:>+8.1f}" for k in show))
    print(f"  {'>SPY':14}" + "".join(f"{int((ys[k] > 0).sum()):>8d}" for k in show) + f"   (von {len(yb)})")

    pd.DataFrame([{"Segment": s, "Strategie": k, **m} for (s, k), m in res.items()]).to_csv(OUT / "futures_metrics.csv", index=False)
    pd.DataFrame(port).to_csv(OUT / "futures_daily.csv")
    print(f"\n  gespeichert in {OUT}")


if __name__ == "__main__":
    main()
