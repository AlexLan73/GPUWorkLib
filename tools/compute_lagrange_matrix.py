#!/usr/bin/env python3
"""
compute_lagrange_matrix.py — Вычисление матрицы Lagrange 48×5 для DelayedFormSignalGenerator.

Алгоритм:
  Kernel использует 4-точечную кубическую Lagrange-интерполяцию.
  Узлы: x = {-1, 0, 1, 2} (sample positions: center-1, center, center+1, center+2)
  Оценка: x = μ = k/48, k = 0..47  (дробная часть задержки)

  L₀(μ) = μ(μ-1)(μ-2) / ((-1)(−2)(−3)) = -μ(μ-1)(μ-2)/6     → center-1
  L₁(μ) = (μ+1)(μ-1)(μ-2) / ((1)(−1)(−2)) = (μ+1)(μ-1)(μ-2)/2 → center
  L₂(μ) = (μ+1)μ(μ-2) / ((2)(1)(−1)) = -(μ+1)μ(μ-2)/2       → center+1
  L₃(μ) = (μ+1)μ(μ-1) / ((3)(2)(1)) = (μ+1)μ(μ-1)/6         → center+2
  L₄(μ) = 0                                                    → center+3 (unused)
"""

import sys

N_ROWS = 48
N_COLS = 5


def lagrange_4pt(mu: float) -> tuple[float, float, float, float]:
    """4-точечная Lagrange-интерполяция, узлы {-1,0,1,2}, оценка в μ."""
    L0 = -mu * (mu - 1.0) * (mu - 2.0) / 6.0
    L1 =  (mu + 1.0) * (mu - 1.0) * (mu - 2.0) / 2.0
    L2 = -(mu + 1.0) * mu * (mu - 2.0) / 2.0
    L3 =  (mu + 1.0) * mu * (mu - 1.0) / 6.0
    return L0, L1, L2, L3


def main() -> None:
    print("Вычисление матрицы Lagrange 48×5 (4-точечная кубическая)")
    print("Узлы: {-1, 0, 1, 2}, μ = k/48, k=0..47")
    print("=" * 60)

    rows: list[list[float]] = []
    max_sum_err = 0.0

    for k in range(N_ROWS):
        mu = k / N_ROWS
        L0, L1, L2, L3 = lagrange_4pt(mu)
        row = [L0, L1, L2, L3, 0.0]
        rows.append(row)

        total = sum(row)
        err = abs(total - 1.0)
        max_sum_err = max(max_sum_err, err)
        if err > 1e-9:
            print(f"  WARN row {k:2d} (μ={mu:.6f}): sum={total:.10f}, err={err:.2e}")

    print(f"\nМакс. ошибка суммы: {max_sum_err:.2e}")

    # Вывод матрицы в формате C++
    print("\n" + "=" * 60)
    print("// Скопировать в delayed_form_signal_generator.cpp:\n")
    print("static const float kBuiltinLagrangeMatrix[48 * 5] = {")

    for k, row in enumerate(rows):
        mu = k / N_ROWS
        # Форматируем с 8 значащими знаками
        vals = [f"{v:11.8f}f" for v in row]
        vals_str = ", ".join(vals)
        print(f"  {vals_str},  // Row {k:2d}  (μ = {k}/48 = {mu:.6f})")

    print("};")

    # Верификация: μ=0 должно давать [0,1,0,0,0]
    print("\n" + "=" * 60)
    print("Верификация контрольных значений:")
    checks = [
        (0, [0.0, 1.0, 0.0, 0.0, 0.0]),
        (12, None),  # μ=0.25
        (24, None),  # μ=0.5
        (48, None),  # μ=1.0 → same as μ=0
    ]
    for k, expected in checks:
        if k == 48:
            mu = 1.0
            L0, L1, L2, L3 = lagrange_4pt(mu)
            row = [L0, L1, L2, L3, 0.0]
            print(f"  μ=1.0 (expected = μ=0): {[f'{v:.6f}' for v in row]}")
            continue
        row = rows[k]
        total = sum(row)
        print(f"  Row {k:2d} (μ={k}/48={k/48:.6f}): {[f'{v:.6f}' for v in row]}, sum={total:.10f}", end="")
        if expected is not None:
            ok = all(abs(row[i] - expected[i]) < 1e-9 for i in range(N_COLS))
            print(f"  {'✓' if ok else '✗ FAIL expected ' + str(expected)}", end="")
        print()

    # Сравнение с исходными значениями в коде (first 5 rows)
    print("\n" + "=" * 60)
    print("Сравнение вычисленных vs. исходных значений (первые 5 строк):")
    original = [
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [-0.0052, 1.0417, -0.0417, 0.0052, 0.0],
        [-0.01, 1.08, -0.08, 0.01, 0.0],
        [-0.0143, 1.1143, -0.1143, 0.0143, 0.0],
        [-0.018, 1.144, -0.144, 0.018, 0.0],
    ]
    for k in range(5):
        calc = rows[k]
        orig = original[k]
        diff = [abs(calc[i] - orig[i]) for i in range(N_COLS)]
        max_diff = max(diff)
        mu = k / N_ROWS
        print(f"  Row {k} (μ={mu:.6f}): max_diff={max_diff:.6f}")
        print(f"    calc: {[f'{v:+.6f}' for v in calc[:4]]}")
        print(f"    orig: {[f'{v:+.6f}' for v in orig[:4]]}")


if __name__ == "__main__":
    main()
