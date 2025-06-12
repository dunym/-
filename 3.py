import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

# Параметры симуляции
np.random.seed(42)
n_simulations = 1000
sample_sizes = [20, 50, 100]
effects = [0.0, 0.2, 0.5, 1.0]  # Величина эффекта

# Результаты будут храниться здесь
results = []

for n in sample_sizes:
    for effect in effects:
        # Счетчики для отвержения гипотез
        t_normal_reject = 0
        wilcoxon_normal_reject = 0
        t_nonnormal_reject = 0
        wilcoxon_nonnormal_reject = 0
        levene_reject = 0
        bartlett_reject = 0

        for _ in range(n_simulations):
            # Генерация данных (нормальное распределение)
            group1 = np.random.normal(0, 1, n)
            group2 = np.random.normal(effect, 1, n)

            # Генерация данных с нарушением нормальности (t-распределение)
            nonnormal1 = np.random.standard_t(3, n)
            nonnormal2 = np.random.standard_t(3, n) + effect

            # Проверка средних (нормальные данные)
            _, t_p = stats.ttest_ind(group1, group2)
            _, w_p = stats.mannwhitneyu(group1, group2)

            # Проверка средних (ненормальные данные)
            _, t_p_non = stats.ttest_ind(nonnormal1, nonnormal2)
            _, w_p_non = stats.mannwhitneyu(nonnormal1, nonnormal2)

            # Проверка дисперсий (на нормальных данных)
            _, l_p = stats.levene(group1, group2)
            _, b_p = stats.bartlett(group1, group2)

            # Считаем количество отвержений H0
            t_normal_reject += int(t_p < 0.05)
            wilcoxon_normal_reject += int(w_p < 0.05)
            t_nonnormal_reject += int(t_p_non < 0.05)
            wilcoxon_nonnormal_reject += int(w_p_non < 0.05)
            levene_reject += int(l_p < 0.05)
            bartlett_reject += int(b_p < 0.05)

        # Сохраняем результаты
        results.append({
            'n': n,
            'effect': effect,
            't_normal': t_normal_reject / n_simulations,
            'wilcoxon_normal': wilcoxon_normal_reject / n_simulations,
            't_nonnormal': t_nonnormal_reject / n_simulations,
            'wilcoxon_nonnormal': wilcoxon_nonnormal_reject / n_simulations,
            'levene': levene_reject / n_simulations,
            'bartlett': bartlett_reject / n_simulations
        })

# Создаем DataFrame для удобства
results_df = pd.DataFrame(results)

# Визуализация результатов
plt.figure(figsize=(15, 10))

# График для средних (нормальные данные)
plt.subplot(2, 2, 1)
for n in sample_sizes:
    subset = results_df[results_df['n'] == n]
    plt.plot(subset['effect'], subset['t_normal'], 'o-', label=f't-тест (n={n})')
    plt.plot(subset['effect'], subset['wilcoxon_normal'], 's-', label=f'Уилкоксон (n={n})')
plt.title('Сравнение средних (нормальные данные)')
plt.xlabel('Величина эффекта')
plt.ylabel('Мощность')
plt.legend()
plt.grid(True)

# График для средних (ненормальные данные)
plt.subplot(2, 2, 2)
for n in sample_sizes:
    subset = results_df[results_df['n'] == n]
    plt.plot(subset['effect'], subset['t_nonnormal'], 'o-', label=f't-тест (n={n})')
    plt.plot(subset['effect'], subset['wilcoxon_nonnormal'], 's-', label=f'Уилкоксон (n={n})')
plt.title('Сравнение средних (t-распределение, df=3)')
plt.xlabel('Величина эффекта')
plt.ylabel('Мощность')
plt.legend()
plt.grid(True)

# График для дисперсий (нормальные данные)
plt.subplot(2, 2, 3)
for n in sample_sizes:
    subset = results_df[results_df['n'] == n]
    plt.plot(subset['effect'], subset['levene'], 'o-', label=f'Левене (n={n})')
    plt.plot(subset['effect'], subset['bartlett'], 's-', label=f'Bartlett (n={n})')
plt.title('Сравнение дисперсий (нормальные данные)')
plt.xlabel('Величина эффекта')
plt.ylabel('Мощность')
plt.legend()
plt.grid(True)

# График для дисперсий (ненормальные данные)
plt.subplot(2, 2, 4)
for n in sample_sizes:
    levene_non = []
    bartlett_non = []
    for effect in effects:
        l_reject = 0
        b_reject = 0
        for _ in range(100):  # Уменьшим количество итераций для скорости
            # Генерация ненормальных данных
            group1 = np.random.standard_t(3, n)
            group2 = np.random.standard_t(3, n) * (1 + effect)  # Изменяем дисперсию

            # Проверка дисперсий
            _, l_p = stats.levene(group1, group2)
            _, b_p = stats.bartlett(group1, group2)

            l_reject += int(l_p < 0.05)
            b_reject += int(b_p < 0.05)

        levene_non.append(l_reject / 100)
        bartlett_non.append(b_reject / 100)

    plt.plot(effects, levene_non, 'o-', label=f'Левене (n={n})')
    plt.plot(effects, bartlett_non, 's-', label=f'Bartlett (n={n})')
plt.title('Сравнение дисперсий (t-распределение, df=3)')
plt.xlabel('Отношение дисперсий (1 + effect)')
plt.ylabel('Мощность')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Вывод таблицы результатов
print("\nРезультаты симуляции:")
print("n\tEffect\tt-норм\tУилк-норм\tt-ненорм\tУилк-ненорм\tЛевене\tBartlett")
for _, row in results_df.iterrows():
    print(f"{row['n']}\t{row['effect']:.1f}\t"
          f"{row['t_normal']:.3f}\t{row['wilcoxon_normal']:.3f}\t"
          f"{row['t_nonnormal']:.3f}\t{row['wilcoxon_nonnormal']:.3f}\t"
          f"{row['levene']:.3f}\t{row['bartlett']:.3f}")