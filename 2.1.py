import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, cauchy

def mse(true_theta, estimates):
    return np.mean((estimates - true_theta)**2)

def simulation(dist, true_theta, n, n_sim=10000):
    mean_ests = []
    median_ests = []
    for _ in range(n_sim):
        sample = dist.rvs(size=n)
        mean_ests.append(np.mean(sample))
        median_ests.append(np.median(sample))
    return np.array(mean_ests), np.array(median_ests)

n_list = [5, 10, 30]  # sample sizes
n_sim = 10000
results = {}

distributions = {
    'Normal': norm(loc=0, scale=1),
    'Cauchy': cauchy(loc=0, scale=1)
}

for dist_name, dist in distributions.items():
    results[dist_name] = {'mean': [], 'median': []}
    for n in n_list:
        mean_ests, median_ests = simulation(dist, 0, n, n_sim)
        mse_mean = mse(0, mean_ests)
        mse_median = mse(0, median_ests)
        results[dist_name]['mean'].append(mse_mean)
        results[dist_name]['median'].append(mse_median)
        print(f"{dist_name} dist, n={n}: MSE(mean)={mse_mean:.3f}, MSE(median)={mse_median:.3f}")

# Визуализация

fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for i, dist_name in enumerate(distributions.keys()):
    axs[i].plot(n_list, results[dist_name]['mean'], marker='o', label='Mean')
    axs[i].plot(n_list, results[dist_name]['median'], marker='o', label='Median')
    axs[i].set_title(f'{dist_name} distribution')
    axs[i].set_xlabel('Sample size (n)')
    axs[i].set_ylabel('MSE')
    axs[i].set_yscale('log')
    axs[i].legend()
plt.suptitle('MSE of mean and median estimates for small samples')
plt.tight_layout()
plt.show()