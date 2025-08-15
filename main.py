import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(42)
import matplotlib

matplotlib.rcParams["font.family"] = "Hiragino Sans"


def monte_carlo(mu, sd, n_samples):
    return np.random.normal(mu, sd, n_samples)


def run(params_csv, outdir="results", bins=20):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(params_csv)

    plt.figure(figsize=(10, 6))
    for _, row in df.iterrows():
        samples = monte_carlo(row["mean"], row["sd"], int(row["n"]))
        plt.hist(samples, bins=bins, alpha=0.6, label=row["group"])

    plt.title("EEGパラメータからのモンテカルロ分布")
    plt.xlabel("EEGパワー")
    plt.ylabel("出現頻度")
    plt.legend()

    out_png = os.path.join(outdir, "eeg_montecarlo.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run("eeg_params.csv")
