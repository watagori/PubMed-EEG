import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Mac の日本語フォント設定
matplotlib.rcParams["font.family"] = "Hiragino Sans"

# 再現性のため乱数固定
np.random.seed(42)


def monte_carlo_sim(mu, sigma, n_samples=100):
    return np.random.normal(mu, sigma, n_samples)


# データ例（論文の値に差し替え可）
low_iq_mu, low_iq_sigma = 5, 1.0
high_iq_mu, high_iq_sigma = 6, 2.5
low_iq_samples = monte_carlo_sim(low_iq_mu, low_iq_sigma)
high_iq_samples = monte_carlo_sim(high_iq_mu, high_iq_sigma)

# グラフ描画
plt.figure(figsize=(10, 6))
plt.hist(
    low_iq_samples, bins=20, alpha=0.6, label="低IQ群（予測容易）", color="skyblue"
)
plt.hist(
    high_iq_samples, bins=20, alpha=0.6, label="高IQ群（予測困難）", color="salmon"
)
plt.title("モンテカルロ法によるEEGパワー分布の模擬")
plt.xlabel("EEGパワーの値（任意単位）")
plt.ylabel("出現頻度")
plt.legend()
plt.show()
