#!/usr/bin/env python3
import argparse
import math
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class Group:
    name: str
    mean: float
    sd: float
    n: int


def load_groups(p):
    df = pd.read_csv(p)
    df.columns = [c.lower() for c in df.columns]
    g = [
        Group(str(r["name"]), float(r["mean"]), float(r["sd"]), int(r["n"]))
        for _, r in df.iterrows()
    ]
    if len(g) != 2:
        raise ValueError("need exactly 2 rows: name,mean,sd,n")
    return g[0], g[1]


def sample(g, rng):
    return rng.normal(g.mean, g.sd, g.n)


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    sp = math.sqrt(
        ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2)
    )
    return 0.0 if sp == 0 else (x.mean() - y.mean()) / sp


def power(g1, g2, trials, alpha, seed):
    rng = np.random.default_rng(seed)
    sig = corr = 0
    ds = []
    for _ in range(trials):
        x, y = sample(g1, rng), sample(g2, rng)
        _, p = stats.ttest_ind(x, y, equal_var=False)
        d = cohens_d(x, y)
        ds.append(d)
        if p < alpha:
            sig += 1
            if (g2.mean - g1.mean) * (y.mean() - x.mean()) > 0:
                corr += 1
    return sig / trials, float(np.mean(ds)), corr / trials


def plot(x, y, labels, bins, out_png):
    plt.figure(figsize=(10, 6))
    plt.hist(x, bins=bins, alpha=0.6, label=labels[0])
    plt.hist(y, bins=bins, alpha=0.6, label=labels[1])
    plt.xlabel("value")
    plt.ylabel("freq")
    plt.title("Monte Carlo (simulated)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--trials", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    g1, g2 = load_groups(a.csv)
    rng = np.random.default_rng(a.seed)
    x, y = sample(g1, rng), sample(g2, rng)
    plot(x, y, (g1.name, g2.name), a.bins, os.path.join(a.out, "hist.png"))
    pw, md, pc = power(g1, g2, a.trials, a.alpha, a.seed)
    t, p = stats.ttest_ind(x, y, equal_var=False)
    d = cohens_d(x, y)
    pd.DataFrame(
        [
            {
                "group1": g1.name,
                "group2": g2.name,
                "mean1": x.mean(),
                "sd1": x.std(ddof=1),
                "n1": g1.n,
                "mean2": y.mean(),
                "sd2": y.std(ddof=1),
                "n2": g2.n,
                "t": t,
                "p": p,
                "cohens_d": d,
                "alpha": a.alpha,
                "power": pw,
                "mean_d_trials": md,
                "prop_sig_expected_dir": pc,
            }
        ]
    ).to_csv(os.path.join(a.out, "summary.csv"), index=False)


if __name__ == "__main__":
    main()
