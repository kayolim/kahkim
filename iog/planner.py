"""Weekly planner: forecast -> production lots (DP) -> job sequences (IG) -> material orders (MRP DP).

Usage
-----
python -m iog.planner forecast.csv --today 2026-11-07 \
    --fg P1=8929 P2=23220 --mat M1=24000 M2=318000 M3=16000 \
    --buffer P1=0.03 P2=0.05 --holidays 2026-12-25 --seq-time 20

forecast.csv columns: date,P1,P2   (expected demand in units; 0 / blank on closed days)
Outputs are printed and written to plans/<today>/.
"""
from __future__ import annotations

import argparse
import math
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import params as G
from .curves import ROOT, load_curves, time_table
from .flowshop import iterated_greedy, makespan, spt, to_ids


# ----------------------------------------------------------------------------- production
def can_produce(product: str, d: date, holidays: set) -> bool:
    if product == "P1":
        return d.weekday() < 5 and d not in holidays
    return True


def plan_production(product, dates, demand, init_fg, curves, holidays, lead_ok=True):
    """Exact lot-sizing DP.

    Each lot is produced on day s and covers net demand of days t..u-1 with s <= t.
    Lots are produced on strictly increasing days (max one lot per product per day).
    Cost = setup + labor(makespan(jobs, weekday(s))) + holding * units * nights in stock.
    Returns (total_cost, [dict(lot)...], uncovered_units).
    """
    T = len(dates)
    net = np.array(demand, dtype=np.int64)
    stock = int(init_fg)
    for k in range(T):  # FIFO consume initial finished goods
        use = min(stock, net[k])
        net[k] -= use
        stock -= use
    hold = G.FG_HOLD[product]
    allowed = [can_produce(product, d, holidays) for d in dates]

    def lot_cost(s, t, u):
        qty = int(net[t:u].sum())
        if qty == 0:
            return 0, 0, 0
        jobs = math.ceil(qty / G.JOB_SIZE)
        if jobs > G.MAX_JOBS:
            return None
        ms = int(curves[(product, G.DAYS[dates[s].weekday()])][jobs])
        h = hold * int(sum(net[k] * (k - s) for k in range(t, u)))
        return G.production_cost(product, ms, jobs) + h, jobs, ms

    @lru_cache(maxsize=None)
    def F(t, p):
        # skip days with no net demand
        while t < T and net[t] == 0:
            t += 1
        if t == T:
            return 0, ()
        best = (float("inf"), ())
        for s in range(p, t + 1):
            if not allowed[s]:
                continue
            for u in range(t + 1, T + 1):
                c = lot_cost(s, t, u)
                if c is None:
                    break
                rest, lots = F(u, s + 1)
                tot = c[0] + rest
                if tot < best[0]:
                    best = (tot, ((s, t, u, c[1], c[2]),) + lots)
        return best

    cost, lots = F(0, 0)
    if math.isinf(cost):
        raise RuntimeError(f"{product}: no feasible plan (check holidays / 500-job cap)")
    out = []
    for s, t, u, jobs, ms in lots:
        out.append(dict(product=product, date=dates[s], covers=f"{dates[t]}~{dates[u - 1]}",
                        jobs=jobs, qty=jobs * G.JOB_SIZE, makespan_neh=ms))
    return cost, out


def sequence_lots(lots, seq_time):
    """Run IG on each lot; attach job sequence + exact costs."""
    for lot in lots:
        P = time_table(lot["product"], G.DAYS[lot["date"].weekday()])[: lot["jobs"]]
        seq, ms = iterated_greedy(P, time_limit=seq_time)
        lot["makespan"] = int(ms)
        lot["makespan_spt"] = int(makespan(spt(P), P))
        lot["labor"] = G.labor_cost(ms)
        lot["labor_if_spt"] = G.labor_cost(lot["makespan_spt"])
        lot["setup"] = G.SETUP_COST[lot["product"]]
        lot["sequence"] = to_ids(seq)
    return lots


# ----------------------------------------------------------------------------- materials
def plan_material(mat, req, dates, today, on_hand):
    """Wagner-Whitin with lead time. req[k] = units needed on dates[k].

    An order placed on day o arrives o+LT and is usable that day. Orders can be placed on any
    day >= today. NORMAL orders preferred; URGENT (if available) only when NORMAL can't arrive.
    """
    spec = G.MATERIALS[mat]
    T = len(dates)
    net = np.array(req, dtype=np.int64)
    stock = int(on_hand)
    for k in range(T):
        use = min(stock, net[k])
        net[k] -= use
        stock -= use
    leftover = stock

    def earliest(otype):
        return today + timedelta(days=spec["types"][otype][1])

    @lru_cache(maxsize=None)
    def F(t):
        while t < T and net[t] == 0:
            t += 1
        if t == T:
            return 0, ()
        best = (float("inf"), ())
        # arrival day a <= dates[t]; latest feasible arrival is best (less holding), so a = dates[t]
        for otype, (unit, lt) in spec["types"].items():
            if earliest(otype) > dates[t]:
                continue
            premium = (unit - spec["types"]["NORMAL"][0])
            for u in range(t + 1, T + 1):
                qty = int(net[t:u].sum())
                h = spec["hold"] * int(sum(net[k] * (k - t) for k in range(t, u)))
                c = spec["order_cost"] + h + premium * qty
                rest, orders = F(u)
                if c + rest < best[0]:
                    best = (c + rest, ((otype, dates[t] - timedelta(days=lt), dates[t], qty),) + orders)
            if otype == "NORMAL":
                break  # urgent only when normal is infeasible
        if math.isinf(best[0]):  # nothing can arrive in time
            rest, orders = F(t + 1)
            return rest, (("SHORTAGE", None, dates[t], int(net[t])),) + orders
        return best

    cost, orders = F(0)
    return cost, [dict(material=mat, order_type=o, order_date=od, arrival=a, qty=q)
                  for o, od, a, q in orders], leftover


# ----------------------------------------------------------------------------- baseline
def naive_cost(product, dates, demand, init_fg, holidays, curves):
    """Produce each day's own demand with SPT (roughly what Auto does) for comparison."""
    tot = 0
    for d, q in zip(dates, demand):
        if q <= 0 or not can_produce(product, d, holidays):
            continue
        jobs = math.ceil(q / G.JOB_SIZE)
        P = time_table(product, G.DAYS[d.weekday()])[:jobs]
        tot += G.production_cost(product, int(makespan(spt(P), P)), jobs)
    return tot


# ----------------------------------------------------------------------------- CLI
def _kv(items, cast=float):
    return {k: cast(v) for k, v in (s.split("=") for s in items or [])}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("forecast")
    ap.add_argument("--today", required=True, help="planning date (orders can be placed from today)")
    ap.add_argument("--fg", nargs="*", help="finished-goods stock, e.g. P1=8929 P2=23220")
    ap.add_argument("--mat", nargs="*", help="material stock, e.g. M1=24000 M2=318000 M3=16000")
    ap.add_argument("--buffer", nargs="*", help="safety margin on forecast, e.g. P1=0.03 P2=0.05")
    ap.add_argument("--holidays", nargs="*", default=[])
    ap.add_argument("--seq-time", type=float, default=10.0, help="IG seconds per lot")
    ap.add_argument("--commit-days", type=int, default=7, help="days whose plan you will enter")
    a = ap.parse_args(argv)

    today = date.fromisoformat(a.today)
    holidays = {date.fromisoformat(h) for h in a.holidays}
    fc = pd.read_csv(a.forecast, parse_dates=["date"]).fillna(0)
    dates = [d.date() for d in fc["date"]]
    fg, mat, buf = _kv(a.fg, int), _kv(a.mat, int), _kv(a.buffer)
    curves = load_curves()
    outdir = ROOT / "plans" / str(today)
    outdir.mkdir(parents=True, exist_ok=True)

    all_lots, summary = [], []
    for p in ("P1", "P2"):
        dem = [int(round(x * (1 + buf.get(p, 0.0)))) for x in fc[p]]
        cost, lots = plan_production(p, dates, dem, fg.get(p, 0), curves, holidays)
        lots = sequence_lots(lots, a.seq_time)
        all_lots += lots
        naive = naive_cost(p, dates, dem, fg.get(p, 0), holidays, curves)
        prod = sum(l["setup"] + l["labor"] for l in lots)
        summary.append((p, prod, cost - sum(l["setup"] + G.labor_cost(l["makespan_neh"]) for l in lots), naive))

    print("\n=== Production plan ===")
    for l in sorted(all_lots, key=lambda l: (l["date"], l["product"])):
        print(f"{l['date']} ({G.DAYS[l['date'].weekday()]}) {l['product']}: {l['jobs']:>3} jobs "
              f"covers {l['covers']}  makespan {l['makespan']} (SPT {l['makespan_spt']})  "
              f"cost {l['setup'] + l['labor']:,} (SPT would be {l['setup'] + l['labor_if_spt']:,})")
        fn = outdir / f"{l['product']}_{l['date']}.txt"
        fn.write_text(l["sequence"] + "\n")
    for p, prod, hold, naive in summary:
        print(f"{p}: production {prod:,} + FG holding {hold:,} | daily-SPT production would be {naive:,}")

    # materials
    print("\n=== Material orders ===")
    req = {m: np.zeros(len(dates), dtype=np.int64) for m in G.MATERIALS}
    idx = {d: i for i, d in enumerate(dates)}
    for l in all_lots:
        for m, k in G.BOM[l["product"]].items():
            req[m][idx[l["date"]]] += k * l["qty"]
    orders = []
    for m in G.MATERIALS:
        c, o, left = plan_material(m, tuple(req[m]), dates, today, mat.get(m, G.MATERIALS[m]["init"]))
        orders += o
        print(f"{m}: need {int(req[m].sum()):,}, ordering cost+holding {c:,}, stock left after horizon {left:,}")
    for o in sorted(orders, key=lambda o: (o["order_date"] or o["arrival"])):
        print(f"  {o['order_type']:<8} {m if False else o['material']} order {o['order_date']} -> arrive {o['arrival']}  qty {o['qty']:,}")

    pd.DataFrame(all_lots).drop(columns="sequence").to_csv(outdir / "production.csv", index=False)
    pd.DataFrame(orders).to_csv(outdir / "materials.csv", index=False)
    print(f"\nJob sequences + CSVs written to {outdir}")


if __name__ == "__main__":
    main()
