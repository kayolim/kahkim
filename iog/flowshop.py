"""Permutation flow-shop scheduling for IOG (20 machines, up to 500 jobs/day).

- makespan(seq, P)       : makespan of a job order (same rule as scheduling_lib.build_schedule)
- spt(P, machine=0)      : the system's "Auto" rule (SPT on one machine)
- neh(P)                 : NEH heuristic with Taillard acceleration
- iterated_greedy(P,...) : Ruiz & Stuetzle IG (destroy/rebuild + insertion local search)

P is an (n_jobs x n_machines) int array; row i = JobID i+1.
Sequences returned to the user are 1-based JobIDs (what you type into IOG).
"""
from __future__ import annotations

import time

import numpy as np

try:  # numba makes IG ~100x faster; code still runs (slowly) without it
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f


@njit(cache=True)
def makespan(seq, P):
    m = P.shape[1]
    c = np.zeros(m, dtype=np.int64)
    for j in seq:
        c[0] += P[j, 0]
        for k in range(1, m):
            c[k] = max(c[k], c[k - 1]) + P[j, k]
    return c[m - 1]


@njit(cache=True)
def _best_insertion(seq, L, job, P, e, q, f):
    """Taillard acceleration: best position to insert `job` into seq[:L]."""
    m = P.shape[1]
    # heads e[i,k]: completion of seq[i] on machine k (i=0..L-1), stored at e[i+1]
    for k in range(m):
        e[0, k] = 0
    for i in range(L):
        j = seq[i]
        e[i + 1, 0] = e[i, 0] + P[j, 0]
        for k in range(1, m):
            e[i + 1, k] = max(e[i, k], e[i + 1, k - 1]) + P[j, k]
    # tails q[i,k]: time from start of seq[i] on k to the end; q[L] = 0
    for k in range(m):
        q[L, k] = 0
    for i in range(L - 1, -1, -1):
        j = seq[i]
        q[i, m - 1] = q[i + 1, m - 1] + P[j, m - 1]
        for k in range(m - 2, -1, -1):
            q[i, k] = max(q[i + 1, k], q[i, k + 1]) + P[j, k]
    best_pos, best_val = 0, np.int64(1 << 62)
    for pos in range(L + 1):
        f[0] = e[pos, 0] + P[job, 0]
        for k in range(1, m):
            f[k] = max(f[k - 1], e[pos, k]) + P[job, k]
        val = np.int64(0)
        for k in range(m):
            v = f[k] + q[pos, k]
            if v > val:
                val = v
        if val < best_val:
            best_val, best_pos = val, pos
    return best_pos, best_val


@njit(cache=True)
def _insert(seq, L, pos, job):
    for i in range(L, pos, -1):
        seq[i] = seq[i - 1]
    seq[pos] = job


@njit(cache=True)
def _neh_order(order, P):
    n, m = P.shape
    seq = np.empty(n, dtype=np.int64)
    e = np.zeros((n + 1, m), dtype=np.int64)
    q = np.zeros((n + 1, m), dtype=np.int64)
    f = np.zeros(m, dtype=np.int64)
    seq[0] = order[0]
    for L in range(1, n):
        pos, _ = _best_insertion(seq, L, order[L], P, e, q, f)
        _insert(seq, L, pos, order[L])
    return seq


@njit(cache=True)
def _local_search(seq, P, e, q, f):
    """Insertion local search: remove each job, reinsert at best spot; repeat until no gain."""
    n = seq.shape[0]
    cur = makespan(seq, P)
    improved = True
    while improved:
        improved = False
        for job in np.random.permutation(seq.copy()):
            idx = 0
            while seq[idx] != job:
                idx += 1
            for i in range(idx, n - 1):
                seq[i] = seq[i + 1]
            pos, val = _best_insertion(seq, n - 1, job, P, e, q, f)
            _insert(seq, n - 1, pos, job)
            if val < cur:
                cur = val
                improved = True
    return cur


@njit(cache=True)
def _ig(seq0, P, d, T, deadline_iters, seed):
    np.random.seed(seed)
    n, m = P.shape
    e = np.zeros((n + 1, m), dtype=np.int64)
    q = np.zeros((n + 1, m), dtype=np.int64)
    f = np.zeros(m, dtype=np.int64)
    cur = seq0.copy()
    cur_val = _local_search(cur, P, e, q, f)
    best = cur.copy()
    best_val = cur_val
    temp = T * P.sum() / (n * m * 10.0)
    for _ in range(deadline_iters):
        s = cur.copy()
        L = n
        removed = np.random.choice(s, min(d, n), replace=False)
        for job in removed:
            idx = 0
            while s[idx] != job:
                idx += 1
            for i in range(idx, L - 1):
                s[i] = s[i + 1]
            L -= 1
        for job in removed:
            pos, _ = _best_insertion(s, L, job, P, e, q, f)
            _insert(s, L, pos, job)
            L += 1
        val = _local_search(s, P, e, q, f)
        if val < cur_val:
            cur, cur_val = s, val
            if val < best_val:
                best, best_val = s.copy(), val
        elif np.random.random() < np.exp(-(val - cur_val) / temp):
            cur, cur_val = s, val
    return best, best_val


def load_times(path, n_jobs=None) -> np.ndarray:
    import pandas as pd
    df = pd.read_csv(path, index_col="JobID")
    P = df.to_numpy(dtype=np.int64)
    return P if n_jobs is None else P[:n_jobs]


def spt(P, machine=0):
    return np.argsort(P[:, machine], kind="stable")


def neh(P):
    order = np.argsort(-P.sum(axis=1), kind="stable")
    return _neh_order(order.astype(np.int64), P)


def iterated_greedy(P, time_limit=10.0, d=4, T=0.4, seed=0, start=None):
    """Run IG in chunks until time_limit seconds; returns (best_seq, best_makespan)."""
    best = neh(P) if start is None else np.asarray(start, dtype=np.int64)
    best_val = makespan(best, P)
    if len(P) <= 2:
        return best, best_val
    t0, chunk, it = time.time(), 5, 0
    while time.time() - t0 < time_limit:
        s, v = _ig(best, P, d, T, chunk, seed + it)
        it += 1
        if v < best_val:
            best, best_val = s, v
        # grow chunk size so overhead stays small but we respect the time limit
        elapsed = time.time() - t0
        if it == 1 and elapsed > 0:
            chunk = max(1, int(chunk * min(50.0, 0.5 / max(elapsed, 1e-3))))
    return best, best_val


def to_ids(seq) -> str:
    """0-based internal order -> IOG input string '3,5,4,2,1'."""
    return ",".join(str(int(j) + 1) for j in seq)
