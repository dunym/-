import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(0)
N = 10_000

samples = {
    "Cauchy": np.random.standard_cauchy(N),
    "t(3)": np.random.standard_t(df=3, size=N),
    "t(10)": np.random.standard_t(df=10, size=N),
    "Normal": np.random.randn(N)
}

results = {}

for name, sample in samples.items():
    finite_sample = sample[np.isfinite(sample)]  # исключим бесконечности
    results[name] = {
        "mean": np.mean(finite_sample),
        "median": np.median(finite_sample),
        "std": np.std(finite_sample),
        "outliers >10": np.mean(np.abs(finite_sample) > 10)
    }

df = pd.DataFrame(results).T
df_rounded = df.round(3)
print(df_rounded)

plt.figure(figsize=(12, 7))
for name, sample in samples.items():
    plt.hist(sample, bins=100, alpha=0.5, density=True, label=name, range=(-10, 10))
plt.legend()
plt.title("Гистограммы распределений с разными хвостами")
plt.xlabel("Значение")
plt.ylabel("Плотность")
plt.grid(True)
plt.show()