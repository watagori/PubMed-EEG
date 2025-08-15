# -*- coding: utf-8 -*-
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(42)
import matplotlib

matplotlib.rcParams["font.family"] = "Hiragino Sans"


def ci95_to_sd(ci_low, ci_high, n):
    z = 1.96
    return (float(ci_high) - float(ci_low)) / (2 * z) * math.sqrt(n)


def se_to_sd(se, n):
    return float(se) * math.sqrt(n)


def resolve_sd(row):
    stype = str(row["spread_type"]).strip().lower()
    if stype == "sd":
        return float(row["spread_value"])
    elif stype == "se":
        return se_to_sd(row["spread_value"], int(row["n"]))
    elif stype == "ci95":
        return ci95_to_sd(row["ci95_low"], row["ci95_high"], int(row["n"]))
    else:
        raise ValueError("spread_type は sd / se / ci95 のいずれかにしてください")


def monte_carlo(mu, sd, n_samples):
    return np.random.normal(loc=mu, scale=sd, size=n_samples)


def run(
    params_csv, outdir="results", band_filter=None, samples_per_group=None, bins=20
):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(params_csv)
    if band_filter is not None:
        df = df[df["band"] == band_filter]
    df = df.copy()
    df["sd_resolved"] = df.apply(resolve_sd, axis=1)
    if samples_per_group is None:
        df["n_samples"] = df["n"].astype(int)
    else:
        df["n_samples"] = int(samples_per_group)
    sim_data = {}
    for _, row in df.iterrows():
        label = f"{row['group']} (band={row['band']})"
        sim_data[label] = monte_carlo(
            row["mean"], row["sd_resolved"], int(row["n_samples"])
        )
    plt.figure(figsize=(10, 6))
    for label, samples in sim_data.items():
        plt.hist(samples, bins=bins, alpha=0.6, label=label)
    title = "EEGパラメータ（PubMed要約統計）からのモンテカルロ分布"
    if band_filter:
        title += f" / band={band_filter}"
    plt.title(title)
    plt.xlabel("EEGパワー（論文の単位に合わせる）")
    plt.ylabel("出現頻度")
    plt.legend()
    out_png = os.path.join(outdir, "eeg_montecarlo_from_pubmed.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    rows = []
    for label, samples in sim_data.items():
        for v in samples:
            rows.append({"group": label, "value": v})
    out_csv = os.path.join(outdir, "eeg_montecarlo_samples.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_png, out_csv


if __name__ == "__main__":
    png, csv = run("eeg_params_template.csv")
    print("図を保存しました:", png)
    print("サンプルCSVを保存しました:", csv)
