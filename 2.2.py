import numpy as np

# Исходные данные (например, скошенное распределение)
data = np.random.exponential(scale=2, size=30)

# Выборочная оценка параметра, например, среднего
theta_hat = np.mean(data)

# Бутстрэп: параметры
n_bootstrap = 10000
boot_samples = np.random.choice(data, size=(n_bootstrap, len(data)), replace=True)
boot_means = np.mean(boot_samples, axis=1)

# Оценка смещения (bootstrap-bias)
# bias_boot = E_bootstrap[theta_hat*] - theta_hat
bootstrap_bias = np.mean(boot_means) - theta_hat

# Смещённая (bias-corrected) оценка параметра
theta_hat_corrected = theta_hat - bootstrap_bias

print(f"Выборочная средняя (theta_hat): {theta_hat:.4f}")
print(f"Оценка bootstrap-смещения: {bootstrap_bias:.4f}")
print(f"Скорректированная оценка: {theta_hat_corrected:.4f}")