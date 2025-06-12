import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
np.random.seed(42)
# "Чистые" данные
data = np.random.normal(loc=0, scale=1, size=50)
# Добавим выбросы (например, по 2 больших выброса в обе стороны)
outliers = np.array([10, 12, -11, -9])
data_with_outliers = np.concatenate([data, outliers])
# Арифметическое среднее
mean_clean = np.mean(data)
mean_outl = np.mean(data_with_outliers)
# Усечённое среднее (например, 10% с каждой стороны)
# для усечённого среднего используем scipy.stats.trim_mean
trim_fraction = 0.1  # 10%
trimmed_mean_clean = stats.trim_mean(data, trim_fraction)
trimmed_mean_outl = stats.trim_mean(data_with_outliers, trim_fraction)

# оценка разброса
var_clean = np.var(data, ddof=1)
var_outl = np.var(data_with_outliers, ddof=1)
def mad(arr):
    med = np.median(arr)
    return np.median(np.abs(arr - med))

mad_clean = mad(data)
mad_outl = mad(data_with_outliers)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(data, bins=15, alpha=0.6, label='Без выбросов')
plt.axvline(mean_clean, color='b', linestyle='--', label='Среднее')
plt.axvline(trimmed_mean_clean, color='g', linestyle='--', label='Усечённое среднее')
plt.title('Распределение без выбросов')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(data_with_outliers, bins=15, alpha=0.6, label='С выбросами')
plt.axvline(mean_outl, color='b', linestyle='--', label='Среднее')
plt.axvline(trimmed_mean_outl, color='g', linestyle='--', label='Усечённое среднее')
plt.title('Распределение с выбросами')
plt.legend()
plt.tight_layout()
plt.show()

print("Без выбросов:")
print(f"  Среднее арифметическое: {mean_clean:.2f}")
print(f"  Усечённое среднее (10%): {trimmed_mean_clean:.2f}")
print(f"  Дисперсия: {var_clean:.2f}")
print(f"  MAD: {mad_clean:.2f}")

print("\nС выбросами:")
print(f"  Среднее арифметическое: {mean_outl:.2f}")
print(f"  Усечённое среднее (10%): {trimmed_mean_outl:.2f}")
print(f"  Дисперсия: {var_outl:.2f}")
print(f"  MAD: {mad_outl:.2f}")