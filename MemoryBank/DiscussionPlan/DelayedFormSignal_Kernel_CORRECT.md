# DelayedFormSignalGenerator — КОРРЕКТНАЯ спецификация kernel дробной задержки

> **Дата**: 2026-02-17  
> **Статус**: ИСПРАВЛЕНИЕ — предыдущая реализация содержала ошибку в формулах интерполяции.  
> **Цель**: Документ для AI-исполнителя — однозначное описание правильного алгоритма.

---

## 1. Ключевое определение: ЗАДЕРЖКА, не конструкция

**Задержка сигнала** — это сдвиг во времени. Выходной сигнал должен быть **задержанной копией** входного:

```
output[n] = input(n - τ)
```

где τ = delay_samples (задержка в отсчётах). **Форма сигнала полностью сохраняется**, только сдвигается во времени.

**Неправильно** (конструкция): создавать выход по произвольной формуле — форма искажается.  
**Правильно** (задержка): для каждого output[n] брать значение input в позиции (n − τ) через интерполяцию.

---

## 2. Математика

Для каждого выходного отсчёта с индексом `n` (sample_id):

```
read_pos = n - delay_samples     // позиция чтения во входном сигнале
```

- Если `read_pos < 0` → output[n] = 0 (ещё «до» начала сигнала).
- Иначе: output[n] = input(read_pos), где input(read_pos) — интерполированное значение в дробной позиции read_pos.

**Целая и дробная части read_pos:**
```
center = floor(read_pos)         // целая часть (индекс «центра» окна)
frac   = read_pos - center      // дробная часть ∈ [0, 1)
```

**Номер строки матрицы 48×5:**
```
row = (uint)(frac * 48) % 48    // frac ∈ [0,1) → row ∈ [0..47]
```

**Важно:** `row` определяется дробной частью **позиции чтения** (frac), а не дробной частью задержки (μ = delay_samples − floor(delay_samples)).

---

## 3. Ошибка в предыдущей спецификации

| Параметр | НЕПРАВИЛЬНО (было) | ПРАВИЛЬНО |
|----------|---------------------|-----------|
| row | `row = (uint)(μ * 48) % 48`, где μ = delay_samples − D | `row = (uint)(frac * 48) % 48`, где frac = read_pos − floor(read_pos) |
| center | `center = sample_id - D`, где D = floor(delay_samples) | `center = (int)floor(read_pos)` |

**Пример (delay = 5.23, n = 6):**
- read_pos = 6 − 5.23 = **0.77**
- center = floor(0.77) = **0**
- frac = 0.77
- row = (uint)(0.77 * 48) % 48 = **36** (не 11!)

При старых формулах получалось center=1, row=11 → интерполировали input(1.23) вместо input(0.77) → искажение формы.

---

## 4. Алгоритм kernel (пошагово)

```
Вход: input (float2), delay_us, sample_rate, lagrange_matrix (48×5), ...
Выход: output (float2)

Для каждого sample_id (индекс выходного отсчёта):

1. delay_samples = delay_us[antenna_id] * 1e-6f * sample_rate

2. read_pos = (float)sample_id - delay_samples

3. Если read_pos < 0:
      output[gid] = (0, 0)
      return

4. center = (int)floor(read_pos)
   frac   = read_pos - (float)center

5. row = ((uint)(frac * 48.0f)) % 48u

6. L[0..4] = lagrange_matrix[row * 5 + 0..4]

7. Окно интерполяции: 5 отсчётов с индексами
   center-1, center, center+1, center+2, center+3
   (за границами [0, points) подставлять 0)

8. output[gid] = L[0]*input[center-1] + L[1]*input[center] + L[2]*input[center+1] + L[3]*input[center+2] + L[4]*input[center+3]

9. Добавить шум (Philox + Box-Muller), если noise_amplitude > 0
```

---

## 5. Псевдокод (OpenCL-подобный)

```c
float read_pos = (float)sample_id - delay_samples;

if (read_pos < 0.0f) {
    output[gid] = (float2)(0.0f, 0.0f);
    return;
}

int center = (int)floor(read_pos);
float frac = read_pos - (float)center;
uint row = ((uint)(frac * 48.0f)) % 48u;

float L0 = lagrange_matrix[row * 5u + 0u];
float L1 = lagrange_matrix[row * 5u + 1u];
float L2 = lagrange_matrix[row * 5u + 2u];
float L3 = lagrange_matrix[row * 5u + 3u];
float L4 = lagrange_matrix[row * 5u + 4u];

float2 s0 = READ_SAMPLE(center - 1);  // 0 за границами
float2 s1 = READ_SAMPLE(center);
float2 s2 = READ_SAMPLE(center + 1);
float2 s3 = READ_SAMPLE(center + 2);
float2 s4 = READ_SAMPLE(center + 3);

float2 result = L0*s0 + L1*s1 + L2*s2 + L3*s3 + L4*s4;
// + noise если нужно
output[gid] = result;
```

---

## 6. Эталон (NumPy/Python)

```python
def apply_delay_numpy_correct(signal, delay_samples, lagrange_matrix):
    """Правильная задержка: output[n] = input(n - delay_samples)."""
    N = len(signal)
    output = np.zeros(N, dtype=np.complex64)
    ds = np.float32(delay_samples)

    for n in range(N):
        read_pos = float(n) - float(ds)
        if read_pos < 0:
            continue  # output[n] = 0
        center = int(np.floor(read_pos))
        frac = read_pos - center
        row = int(frac * 48) % 48
        L = lagrange_matrix[row]

        val = 0.0 + 0.0j
        for k in range(5):
            idx = center - 1 + k
            if 0 <= idx < N:
                val += L[k] * signal[idx]
        output[n] = val
    return output
```

---

## 7. Проверка корректности

1. **Целая задержка 5:** output[5] должен равняться input[0] (row=0, L=[0,1,0,0,0]).
2. **Дробная задержка 5.23, n=6:** output[6] = input(0.77) — интерполяция между input[0] и input[1] с frac=0.77, row=36.
3. **Визуально:** график output должен быть сдвинутой копией input, форма совпадает.

---

## 8. Ссылки

- [Plan_FractionalDelay_Farrow.md](Plan_FractionalDelay_Farrow.md) — общий план (раздел 7 обновлён).
- [Plan_DelayedFormSignal_Semantics_Farrow.md](Plan_DelayedFormSignal_Semantics_Farrow.md) — семантика.
- [lagrange_matrix_48x5.json](lagrange_matrix_48x5.json) — матрица коэффициентов.
- CCRMA: [Lagrange Fractional Delay](https://ccrma.stanford.edu/~jos/Interpolation/Matlab_Code_Lagrange_Fractional.html).
