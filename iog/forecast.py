"""Demand forecast for next week.

P1 demand = Samsung Electronics (005930.KS) close in KRW, KRX trading days only.
P2 demand = Bitcoin close in USD (investing.com, decimals dropped), every day.

Stock prices are close to a random walk, so the honest baseline is "last close", with the
buffer set by the cost asymmetry instead of trying to predict direction:
  under-forecast  -> lost sale: stockout penalty 250 + lost margin  (~350+/unit)
  over-forecast   -> FG holding 30/unit/day until the extra units sell (usually 1-3 days)
=> critical ratio ~0.9, so produce near the 90th percentile of the h-day-ahead price.

Usage (on your own PC; this needs internet):
  pip install yfinance
  python -m iog.forecast --start 2026-11-08 --days 7 --q 0.9 --holidays 2026-12-25 > forecast.csv
or with your own price CSVs (columns date,close):
  python -m iog.forecast --p1-csv samsung.csv --p2-csv btc.csv --start 2026-11-08
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import params as G


def fetch(ticker: str, years: int = 2) -> pd.Series:
    import yfinance as yf
    df = yf.download(ticker, period=f"{years}y", progress=False, auto_adjust=False)
    s = df["Close"].squeeze().dropna()
    s.index = pd.to_datetime(s.index).date
    return s


def load_csv(path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    return pd.Series(df["close"].to_numpy(float), index=df["date"].dt.date)


def horizon_quantiles(s: pd.Series, horizons, q: float, lookback: int = 250) -> dict:
    """Empirical q-quantile of the h-step log return over the last `lookback` observations."""
    lr = np.log(s.to_numpy())[-(lookback + max(horizons)):]
    out = {}
    for h in horizons:
        r = lr[h:] - lr[:-h]
        out[h] = float(np.quantile(r, q)) if len(r) > 20 else 0.0
    return out


def p1_open(d: date, holidays) -> bool:
    return d.weekday() < 5 and d not in holidays


def make_forecast(p1: pd.Series, p2: pd.Series, start: date, days: int, q: float, holidays):
    dates = [start + timedelta(days=i) for i in range(days)]
    last1, last2 = p1.index[-1], p2.index[-1]
    # horizon in *observations*: trading days for P1, calendar days for P2
    h1 = {d: sum(p1_open(last1 + timedelta(days=k), holidays) for k in range(1, (d - last1).days + 1))
          for d in dates}
    h2 = {d: (d - last2).days for d in dates}
    q1 = horizon_quantiles(p1, sorted({max(1, h) for h in h1.values()}), q)
    q2 = horizon_quantiles(p2, sorted({max(1, h) for h in h2.values()}), q)
    rows = []
    for d in dates:
        v1 = int(p1.iloc[-1] * np.exp(q1[max(1, h1[d])])) if p1_open(d, holidays) else 0
        v2 = int(p2.iloc[-1] * np.exp(q2[max(1, h2[d])]))
        rows.append(dict(date=d, P1=v1, P2=v2, P1_median=int(p1.iloc[-1]) if v1 else 0,
                         P2_median=int(p2.iloc[-1])))
    return pd.DataFrame(rows)


def backtest(s: pd.Series, h: int, qs=(0.5, 0.7, 0.8, 0.9, 0.95), under=350, over=30 * 2):
    """Cost per unit of demand for each quantile choice, using rolling 250-obs windows."""
    x = s.to_numpy()
    res = {}
    for q in qs:
        cost = []
        for t in range(260, len(x) - h):
            qq = horizon_quantiles(pd.Series(x[: t + 1]), [h], q)[h]
            f = x[t] * np.exp(qq)
            a = x[t + h]
            cost.append((under * max(a - f, 0) + over * max(f - a, 0)) / a)
        res[q] = float(np.mean(cost))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--q", type=float, default=0.9, help="quantile to produce for")
    ap.add_argument("--p1-csv")
    ap.add_argument("--p2-csv")
    ap.add_argument("--holidays", nargs="*", default=[])
    ap.add_argument("--backtest", action="store_true")
    a = ap.parse_args(argv)
    p1 = load_csv(a.p1_csv) if a.p1_csv else fetch("005930.KS")
    p2 = load_csv(a.p2_csv) if a.p2_csv else fetch("BTC-USD")
    hol = {date.fromisoformat(h) for h in a.holidays}
    if a.backtest:
        for name, s in (("P1", p1), ("P2", p2)):
            for h in (1, 3, 5, 7):
                print(name, f"h={h}", {k: round(v, 2) for k, v in backtest(s, h).items()}, file=sys.stderr)
    make_forecast(p1, p2, date.fromisoformat(a.start), a.days, a.q, hol).to_csv(sys.stdout, index=False)


if __name__ == "__main__":
    main()
