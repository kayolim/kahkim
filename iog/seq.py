"""Optimized job sequence for one production day.

  python -m iog.seq P1 2026-11-09 243 --time 30      # prints the string to paste into IOG
  python -m iog.seq P1 table --n 220-270 --time 6     # lookup table for all weekdays -> sequences/
"""
from __future__ import annotations

import argparse
from datetime import date

import pandas as pd

from . import params as G
from .curves import ROOT, time_table
from .flowshop import iterated_greedy, makespan, spt, to_ids


def best_sequence(product, day, jobs, t):
    P = time_table(product, day)[:jobs]
    seq, ms = iterated_greedy(P, time_limit=t)
    ms_spt = int(makespan(spt(P), P))
    return to_ids(seq), int(ms), ms_spt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("product", choices=["P1", "P2"])
    ap.add_argument("when", help="YYYY-MM-DD, weekday (mon..sun), or 'table'")
    ap.add_argument("jobs", nargs="?", type=int)
    ap.add_argument("--n", default="220-270", help="job range for table mode")
    ap.add_argument("--days", default="mon,tue,wed,thu,fri")
    ap.add_argument("--time", type=float, default=20)
    a = ap.parse_args(argv)
    if a.when == "table":
        lo, hi = map(int, a.n.split("-"))
        out = ROOT / "sequences"
        out.mkdir(exist_ok=True)
        for day in a.days.split(","):
            rows = []
            for n in range(lo, hi + 1):
                s, ms, ms_spt = best_sequence(a.product, day, n, a.time)
                rows.append(dict(jobs=n, makespan=ms, makespan_spt=ms_spt,
                                 cost=G.production_cost(a.product, ms, n),
                                 cost_spt=G.production_cost(a.product, ms_spt, n), sequence=s))
                print(day, n, ms, ms_spt, flush=True)
            pd.DataFrame(rows).to_csv(out / f"{a.product}_{day}.csv", index=False)
        return
    day = a.when if a.when in G.DAYS else G.DAYS[date.fromisoformat(a.when).weekday()]
    s, ms, ms_spt = best_sequence(a.product, day, a.jobs, a.time)
    print(f"{a.product} {day} {a.jobs} jobs: makespan {ms} (SPT {ms_spt}), "
          f"cost {G.production_cost(a.product, ms, a.jobs):,} (SPT {G.production_cost(a.product, ms_spt, a.jobs):,})")
    print(s)


if __name__ == "__main__":
    main()
