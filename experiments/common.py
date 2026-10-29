"""Shared helpers for the tensor experiments (ERA5 and Vlasov-Poisson).

Every sweep writes rows
    method, panel, kind, truncation, param, rep, n_params, compression_ratio, rel_error
where ``kind`` is "ours" (a decomposition over a nontrivial K) or "baseline" (over R, the
algebra coordinates as an ordinary mode), and ``truncation`` is
    uniform       the smallest uniform ranks with error <= eps ||X|| (one rank per mode)
    adaptive      per-block ranks with error <= eps ||X||
    rank          SVD rank (matrices)
For a K with a single block (R) uniform and adaptive coincide and only uniform is run.
Errors are relative Frobenius errors in coefficient form.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from staralg import compression_ratio

FIELDS = ["method", "panel", "kind", "truncation", "param", "rep", "n_params", "compression_ratio", "rel_error"]
# fine tolerance grid of the paper figures (0.005 to 0.6); HOSVD also runs to 0.8, as its error stays well
# below the tolerance (about 0.73 eps), so that every curve crosses relative error 0.5
FORMAT_EPS = [0.005, 0.0075] + [round(0.01 * k, 2) for k in range(1, 51)] + [0.55, 0.6]
FORMAT_EPS_HOSVD = [0.65, 0.7, 0.75, 0.8]


class Recorder:
    def __init__(self, n_data: float, rep=0):
        self.n_data = float(n_data)
        self.rep = rep
        self.rows = []

    def add(self, method, panel, kind, trunc, param, n_params, err):
        self.rows.append([method, panel, kind, trunc, param, self.rep, float(n_params),
                          compression_ratio(self.n_data, n_params), float(err)])


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")


FLOOR = 1e-6    # errors below this are at the accuracy floor of the data (exact reconstructions)


def _interp(cr, err, target):
    """Log-log interpolation at a target compression ratio; nan outside the data, and nan
    where the curve jumps from the accuracy floor (error < FLOOR, e.g. a full-rank, exact
    reconstruction) to a measurable error (>= 100 FLOOR) inside the interval: the log of an
    error at the floor is noise, so log-log interpolation there would invent tiny errors
    (e.g. 0.005 between 1e-15 and 0.73).  Between two points both near the floor it is fine."""
    order = np.argsort(cr)
    c, e = np.asarray(cr, dtype=float)[order], np.asarray(err, dtype=float)[order]
    if target < c[0] or target > c[-1]:
        return float("nan")
    i = int(np.searchsorted(c, target))
    if i < len(c) and np.isclose(c[i], target):
        return float(e[i])
    lo, hi = i - 1, i
    if e[lo] <= 0 or e[hi] <= 0 or (e[lo] < FLOOR and e[hi] >= 100 * FLOOR):
        return float("nan")
    t = (np.log(target) - np.log(c[lo])) / (np.log(c[hi]) - np.log(c[lo]))
    return float(np.exp((1 - t) * np.log(e[lo]) + t * np.log(e[hi])))


def summarize(rows, targets=(2, 4, 8, 16, 32, 64)):
    """Relative error at given compression ratios, per (panel, method, truncation): the
    lower envelope of each curve, interpolated log-log."""
    keys = list(dict.fromkeys((r[1], r[0], r[3]) for r in rows))
    panels = list(dict.fromkeys(k[0] for k in keys))
    for panel in panels:
        print(f"\n[{panel}] relative error at compression ratio")
        print(f"{'method':52s}{'trunc':>13s}" + "".join(f"{'cr=' + str(t):>9s}" for t in targets))
        for (pn, m, tr) in keys:
            if pn != panel:
                continue
            pts = {}
            for r in rows:
                if (r[1], r[0], r[3]) == (pn, m, tr):
                    pts.setdefault(r[4], []).append((r[7], r[8]))
            cr = np.array([np.mean([a for a, _ in v]) for v in pts.values()])
            er = np.array([np.mean([b for _, b in v]) for v in pts.values()])
            # lower envelope: for increasing cr, keep the running best error from the right
            o = np.argsort(cr)
            cr, er = cr[o], er[o]
            env = np.minimum.accumulate(er[::-1])[::-1]
            print(f"{m[:52]:52s}{tr:>13s}" + "".join(f"{_interp(cr, env, t):9.4f}" for t in targets))


def load_rows(path):
    with open(path) as f:
        rd = csv.reader(f)
        next(rd)
        return [[r[0], r[1], r[2], r[3], float(r[4]), r[5], float(r[6]), float(r[7]), float(r[8])] for r in rd]


if __name__ == "__main__":
    # re-print the summary tables of a results file:
    #   python experiments/common.py results/fig6_vlasov_poisson.csv
    #   python experiments/common.py results/fig8_era5_K1.csv 8,16,32,64          (other compression ratios)
    import sys

    tg = tuple(float(t) for t in sys.argv[2].split(",")) if len(sys.argv) > 2 else (2, 4, 8, 16, 32, 64)
    summarize(load_rows(sys.argv[1]), targets=tuple(int(t) if t == int(t) else t for t in tg))
