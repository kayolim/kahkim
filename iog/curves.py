"""Makespan(N) for every product x weekday x N=1..500 (NEH), cached to data/makespan_curves.csv."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .flowshop import load_times, makespan, neh
from .params import DAYS, MAX_JOBS, TIME_FILE

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "makespan_curves.csv"


def ensure_data():
    if not (DATA / "t2_500_20_mon.csv").exists():
        import zipfile
        DATA.mkdir(exist_ok=True)
        zipfile.ZipFile(ROOT / "ProcessingTimeTable_P1_P2.zip").extractall(DATA)


def time_table(product: str, day: str) -> np.ndarray:
    ensure_data()
    return load_times(DATA / TIME_FILE[product].format(day=day))


def build_curves() -> pd.DataFrame:
    rows = []
    for product in TIME_FILE:
        for day in DAYS:
            P = time_table(product, day)
            for n in range(1, MAX_JOBS + 1):
                rows.append((product, day, n, int(makespan(neh(P[:n]), P[:n]))))
    df = pd.DataFrame(rows, columns=["product", "day", "jobs", "makespan"])
    df.to_csv(CACHE, index=False)
    return df


def load_curves() -> dict:
    """{(product, day): array where a[n] = NEH makespan for n jobs (a[0]=0)}"""
    df = pd.read_csv(CACHE) if CACHE.exists() else build_curves()
    out = {}
    for (p, d), g in df.groupby(["product", "day"]):
        a = np.zeros(MAX_JOBS + 1, dtype=np.int64)
        a[g["jobs"].to_numpy()] = g["makespan"].to_numpy()
        out[(p, d)] = a
    return out


if __name__ == "__main__":
    import time
    t = time.time()
    build_curves()
    print(f"saved {CACHE} in {time.time() - t:.1f}s")
