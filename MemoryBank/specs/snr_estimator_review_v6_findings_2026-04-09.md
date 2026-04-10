# 📋 Code Review v6 — дополнительный аудит плана SNR-estimator

> **Объект:** [`snr_estimator_statistics_plan.md`](snr_estimator_statistics_plan.md) (v3 → **v4 после этого ревью**)
> **Базовая ревизия:** [`snr_estimator_review_2026-04-09.md`](snr_estimator_review_2026-04-09.md) (v5 — согласования Alex)
> **Дата:** 2026-04-09
> **Ревьюер:** Кодо
> **Тип ревью:** технический аудит на соответствие реальному коду (code-grounded)
> **Статус:** ✅ **ВСЕ ЗАМЕЧАНИЯ ЗАКРЫТЫ** — план обновлён до v4 (2026-04-09, правки применены)

---

## ✅ РЕЗОЛЮЦИЯ (2026-04-09, чат с Alex)

### Решения Alex по моим вопросам

| Вопрос | Решение Alex |
|---|---|
| **Блокер #1** FFT flexible target_n_fft | **Снят проверкой кода.** `CalculateNFFT` уже делает `NextPowerOf2(n_point) × repeat_count`. Смысл `target_n_fft` = *«N_actual после децимации любой, программа выровняет до 2^i автоматически»*. Никаких изменений в `FFTProcessorParams`/`CalculateNFFT` не требуется. Текст плана уточнён (убраны misleading упоминания 4000/3200 как *«не выравнивается»*). |
| **Блокер #2** peak_cfar_kernel | **A** — берём мой черновик из этого ревью как есть. Вставлен раздел **2.2.7** в план. Помощник может уточнить в понедельник после прогона на реальном GPU. |
| **7 мелких правок** | **A** — все 7 принимаются одним пакетом. |

### Что внесено в план (v3 → v4)

✅ **Раздел 2.2.7 `peak_cfar_kernel`** — добавлен полный псевдокод (one-block-per-antenna, LDS-reduction argmax, wraparound ref-window, защита от `log10(0)`, launch config, 6 критериев ревью, edge cases для пика на `k=0` / `k=nFFT-1`)

✅ **Раздел «Принятые допущения»** — строка «FFT-размер» переформулирована: вместо *«не обязательно степень 2, 2^a·3^b·5^c»* → *«автоматически выравнивается NextPowerOf2(N_actual) × repeat_count через CalculateNFFT»*

✅ **Раздел «Авто-выбор step_samples»** — Case 3 упрощён (`next_rocfft_size` убран, заменён на комментарий что `CalculateNFFT` вычислит nFFT сам). Добавлен справочный блок *«Про non-power-of-2»* с объяснением что разница между pad до 4000 и pad до 4096 — ~0.01 dB, пренебрежимо.

✅ **Раздел 2.1 `SnrEstimationConfig`:**
- `target_N_fft → target_n_fft`, `search_left_right → search_full_spectrum`
- `actual_N_actual → n_actual` в `SnrEstimationResult`
- Добавлен метод `Validate()` с проверкой `2*(guard+ref)+1 < target_n_fft`

✅ **Раздел 2.2.5 `ProcessMagnitudesToGPU`:**
- Удалён параметр `gpu_memory_bytes` (рудимент BatchManager'а)
- Docstring уточнён: `params.n_point = N_actual`, `nFFT = NextPowerOf2 × repeat_count` вычисляется автоматически

✅ **Раздел 2.4 — проверка памяти:**
- Разделена на CPU-overload (учитываем INPUT upload) и GPU-overload (только scratch-буферы, без двойного учёта)
- Helper `SnrEstimatorNewAllocBytes` + `CheckVramAvailable` с контекстом в сообщении ошибки

✅ **Раздел 2.5 `BranchSelector::Select`:**
- NaN/Inf guard в начале: `if (!std::isfinite(snr_db)) return current_;`
- Комментарий "NOT thread-safe между инстансами"

✅ **Раздел «Пороги переключения» (~стр 1190):**
- Оговорка что формула `shift_db = 10·log10(N2/N1)` работает только для сигнала
- Добавлена таблица: coherent gain +3 dB vs CFAR-артефакт `10·log10(H_nFFT)` ≈ +0.35 dB при удвоении nFFT
- Ссылка на Эксп.5 для точной калибровки

✅ **Глобальные замены** (через Python batch):
- 14 × `target_N_fft` → `target_n_fft`
- 4 × `search_left_right` → `search_full_spectrum`
- 5 × `actual_N_actual` → `n_actual`

✅ **Шапка плана** — обновлена до v4 со списком всех внесённых правок

### Что НЕ внесено (по решению Alex)

Ничего — все 2 блокера + 7 мелких правок применены пакетом.

---

## 📊 Ложные тревоги (отозваны в ходе аудита)

Три моих замечания, которые оказались ошибочными при детальной проверке кода:

1. **«Два GpuContext → разные streams → sync нужна»** — неверно. `GpuContext` берёт stream от `backend->GetNativeQueue()` ([`gpu_context.cpp:47`](../../DrvGPU/src/gpu_context.cpp#L47)). Один backend → один stream → всё серийно.

2. **«kMagnitudes slot reuse → конфликт размеров»** — неверно. `BufferSet::Require` возвращает существующий ptr если `size >= bytes` ([`buffer_set.hpp:97-99`](../../DrvGPU/services/buffer_set.hpp#L97-L99)). Вариант 2 из v5 памятки G-1 работает корректно.

3. **«Формула H_N = ln+γ неточная»** — неверно. Для N=2048: приближение `ln(2048)+γ ≈ 8.2018` vs точное `H_{2048} ≈ 8.2021`. Ошибка ~2e-4, в dB незаметна. Математика плана корректна.

---

---

## 📌 Зачем ещё одно ревью поверх v5

v5 зафиксировал **архитектурные решения** (9 согласований с Alex по вопросам Q-1..Q-8). Оно **правильное**, ничего отзывать не нужно. Но v5 не проверял сам план на **техническую реализуемость** через существующий API — это задача текущего аудита.

Я свериала каждый технический пункт плана с реальными файлами кода. Нашла:

- **2 критические проблемы**, которые блокируют компиляцию в понедельник
- **7 мелких замечаний** по стилю и защитному коду
- **2 ложные тревоги** — в ходе проверки я нашла свои же ошибки, отмечаю их явно, чтобы потом не вернуться к ним

---

## ~~🔴 Критическая проблема #1~~ → ✅ **СНЯТ (ложная тревога)** — FFTProcessorROCm физически не умеет flexible `target_N_fft`

> **СТАТУС:** Закрыт после проверки кода (чат с Alex, 2026-04-09).
> `CalculateNFFT` делает `NextPowerOf2(n_point) × repeat_count` — это ровно то, что нужно для SNR.
> Смысл `target_n_fft` = *«N_actual после децимации любой, программа выровняет до 2^i»*, а не *«использовать точно это число»*.
> Текст плана уточнён (v4), никаких изменений в API не требуется.
> Исходный текст замечания ниже оставлен для истории обсуждения.

### Что в плане (строки 39, 74–115, 500)

> *«target_N_fft — параметр Config, default 2048, может быть 1024/2048/4000/4096/…, не обязательно степень 2, PadDataOp допадит»*

И цитата Alex в v5 (п.1): *«нет догм привода к 1024 семпла! это может быть и 4000 а программа сама допишет нули»*

То есть **требование согласовано** — flexible target_N_fft обязателен. Но **техническая реализация не проверена** ни в v5, ни в плане.

### Что в реальном коде

1. **[`FFTProcessorParams`](../../modules/fft_func/include/types/fft_params.hpp#L17-L30)** — **нет** поля `pad_to` / `target_N_fft`:
   ```cpp
   struct FFTProcessorParams {
     uint32_t beam_count = 1;
     uint32_t n_point = 0;           // ← единственная точка управления размером
     float    sample_rate = 1000.0f;
     FFTOutputMode output_mode = FFTOutputMode::COMPLEX;
     uint32_t repeat_count = 1;
     float    memory_limit = 0.80f;
   };
   ```

2. **[`FFTProcessorROCm::CalculateNFFT`](../../modules/fft_func/src/fft_processor_rocm.cpp#L565-L568)** — жёстко вычисляет **только** степень 2:
   ```cpp
   void FFTProcessorROCm::CalculateNFFT(const FFTProcessorParams& params) {
     uint32_t base_fft = NextPowerOf2(params.n_point);
     nFFT_ = base_fft * params.repeat_count;
   }
   ```

3. **`next_rocfft_size(N_actual)`** (`2^a × 3^b × 5^c`) из плана (строка 100, Case 3) — **такой функции не существует в коде**, она нигде не определена.

### Что происходит фактически

| target_N_fft в Config | Что получится |
|---|---|
| 1024 | ✅ OK (степень 2, nextPow2(N_actual)=1024 при N_actual∈[513..1024]) |
| 2048 | ✅ OK (случайно работает для сценариев A/B/C) |
| 4096 | ✅ OK (степень 2) |
| **4000** | ❌ `CalculateNFFT` вернёт 4096, **не 4000** |
| **3200** | ❌ Вернёт 4096, **не 3200** |
| **3000** | ❌ Вернёт 4096 |
| **1024 при n_point=2000** | ❌ Вернёт 2048 (nextPow2(2000)), игнорируя желание 1024 |

То есть план обещает flexible, а текущий API даёт только `nextPow2(n_point)`. Сценарии A/B/C работают **случайно**, потому что все выбраны так, что `nextPow2(N_actual) = 2048`.

### Фикс

Нужно **одно из двух** (выбрать с Alex):

**Вариант A (минимум) — если нужны только степени 2 + `2^a·3^b·5^c`:**
1. Добавить поле в `FFTProcessorParams`:
   ```cpp
   uint32_t pad_to = 0;  ///< 0 = legacy (nextPow2×repeat), >0 = exact nFFT
   ```
2. `CalculateNFFT`:
   ```cpp
   if (params.pad_to > 0) {
     nFFT_ = params.pad_to;  // caller отвечает за валидность для rocFFT
   } else {
     nFFT_ = NextPowerOf2(params.n_point) * params.repeat_count;
   }
   ```
3. Добавить утилиту `next_rocfft_size(uint32_t n)` в `fft_func/include/utils/` — хардкод таблицы `2^a·3^b·5^c` до 65536 (достаточно: 2048, 2500, 2560, 2700, 2880, 3000, 3125, 3200, 3240, 3456, 3600, 3750, 3840, 4000, 4050, 4096…).
4. В `SnrEstimatorOp` перед вызовом `ProcessMagnitudesToGPU` считать `params.pad_to = next_rocfft_size(target_N_fft)`.

**Вариант B (честный) — признать что flexible не нужен:**
Зафиксировать ограничение `target_N_fft должен быть степенью 2 (1024/2048/4096)`. Обновить план (убрать упоминания 4000/3200/3000), добавить assert в `SnrEstimationConfig::Validate()`.

Мой совет: **Вариант A**. Таблица rocFFT-friendly размеров — 30 строк кода, и она полезна для других модулей (heterodyne, filters). Это не раздувание архитектуры.

### Влияние на план

- Раздел **2.1** (`SnrEstimationConfig`) — уточнить: если не Вариант B, то поле `target_N_fft` передаётся в `FFTProcessorParams::pad_to`.
- Раздел **2.2.5** (`ProcessMagnitudesToGPU`) — добавить примечание: `params.pad_to` должен быть установлен caller'ом.
- Раздел **1.5** (fft_func extension) — теперь это **не только** `complex_to_magnitude_squared`, но **ещё и** `pad_to` поле в `FFTProcessorParams` + `next_rocfft_size` утилита. Этап 1.5 в таблице этапов раздувается.

---

## ~~🔴 Критическая проблема #2~~ → ✅ **ЗАКРЫТ** — `peak_cfar_kernel` не описан

> **СТАТУС:** Закрыт (чат с Alex, 2026-04-09). Решение **A** — черновик из этого ревью вставлен в план как раздел 2.2.7.
> Помощник в понедельник может уточнить мелочи после прогона на реальном GPU.
> Исходное описание проблемы ниже оставлено как обоснование блокера.

### Что в плане

Раздел 2.3 (строки 507–511):
```
↳ peak_cfar_kernel — argmax + CA-CFAR per antenna
                   — читает kFftMagSquared (уже |X|²)
                   — пишет в kSnrPerAntenna[n_ant_used]
                   — square-law detector, правильная статистика Exponential
```

Это **единственное** упоминание в 1070-строчной спеке. Ни в разделе 2.2 (где расписан `gather_decimated_kernel` подробно), ни где-либо ещё.

### Чего не хватает

Сравни с `gather_decimated_kernel` (строки 382–422) — там есть всё: сигнатура, thread mapping, launch config, обоснование. Для `peak_cfar_kernel` нужно то же самое, а также:

1. **Файл**: `modules/statistics/kernels/peak_cfar_kernel.hpp`? Или `kernels/snr_kernels_rocm.hpp`?
2. **Thread mapping**: один блок на антенну? Один поток на антенну? Two-pass (argmax → CFAR → SNR)?
3. **Shared memory**: нужна ли LDS-reduction для argmax, или atomic?
4. **Wraparound в ref-window**: `k_peak ± (guard+1..guard+ref)) mod N_fft` — реализовать как условный `(k + N) % N` в loop или через two-range sum?
5. **Формула SNR**: `snr_db = 10 * __log10f(peak / noise_mean)` — в single precision достаточно? Или нужен double?
6. **Обработка `peak == 0`** или `noise_mean == 0`: что писать в `snr_db`? (−inf, `FLT_MIN`, 0?). См. связанное замечание #7 ниже.
7. **Обработка малых N_fft**: assert что `N_fft >= 2*(guard+ref+1)+1`, иначе окна перекрываются или не помещаются.
8. **Edge cases в guard zone**:
   - Пик на `k=0` → `ref_window = [N_fft - guard - ref .. N_fft - guard - 1] ∪ [guard+1 .. guard+ref]`
   - Пик на `k=N_fft-1` → симметрично

### Почему это блокер

В понедельник помощник возьмёт таск, увидит одну строчку «argmax + CA-CFAR» и либо:
- Напишет свой вариант → с высокой вероятностью ошибётся в wraparound или edge cases → тесты test_01 (только шум), test_03 (отрицательная частота) упадут
- Задаст уточняющий вопрос → потеряет полдня ожидания

Kernel **обязательно** прописывается в плане с тем же уровнем детализации, что `gather_decimated_kernel`.

### Фикс

Добавить раздел **`2.2.7 HIP kernel — peak_cfar_kernel`** с полным кодом. Черновик от меня (можно править):

```cpp
// Файл: modules/statistics/kernels/peak_cfar_kernel.hpp
//
// Thread mapping: один БЛОК на антенну, BLOCK_SIZE threads.
// Block делает:
//   1. Parallel argmax (|X|²) через LDS-reduction
//   2. Sum по ref_window с wraparound (один поток на bin, atomic add в LDS)
//   3. Поток 0 пишет snr_db в kSnrPerAntenna[ant]
//
// Причина one-block-per-antenna: n_ant_used ~50, N_fft ~2048.
// 50 блоков × 256 threads = 12 800 threads — достаточно для occupancy.
// Альтернатива one-thread-per-antenna не поможет: внутри антенны нужна reduction.

__launch_bounds__(256)
__global__ void peak_cfar_kernel(
    const float* __restrict__ mag_sq,     // [n_ant × N_fft] |X|²
    float*       __restrict__ snr_db_out, // [n_ant]
    uint32_t N_fft,
    uint32_t guard_bins,
    uint32_t ref_bins)
{
    constexpr unsigned int BLOCK_SIZE = 256;
    __shared__ float s_max_val[BLOCK_SIZE];
    __shared__ uint32_t s_max_idx[BLOCK_SIZE];
    __shared__ float s_ref_sum;
    __shared__ uint32_t s_ref_count;

    uint32_t ant = blockIdx.x;
    const float* row = mag_sq + ant * N_fft;
    uint32_t tid = threadIdx.x;

    // Pass 1: parallel argmax по всему N_fft
    float my_max = -1.0f;
    uint32_t my_idx = 0;
    for (uint32_t k = tid; k < N_fft; k += BLOCK_SIZE) {
        float v = row[k];
        if (v > my_max) { my_max = v; my_idx = k; }
    }
    s_max_val[tid] = my_max;
    s_max_idx[tid] = my_idx;
    __syncthreads();

    // Reduce argmax в s_max_val[0] / s_max_idx[0]
    for (unsigned int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_max_val[tid + s] > s_max_val[tid]) {
                s_max_val[tid] = s_max_val[tid + s];
                s_max_idx[tid] = s_max_idx[tid + s];
            }
        }
        __syncthreads();
    }

    uint32_t k_peak = s_max_idx[0];
    float    peak   = s_max_val[0];

    // Pass 2: ref window sum с wraparound (исключая guard зону и сам пик)
    if (tid == 0) { s_ref_sum = 0.0f; s_ref_count = 0; }
    __syncthreads();

    // Каждый поток обрабатывает несколько bin'ов из ref_window
    // Ref индексы: k_peak ± (guard+1 .. guard+ref) mod N_fft
    uint32_t total_ref = 2 * ref_bins;  // ref_bins с каждой стороны
    for (uint32_t i = tid; i < total_ref; i += BLOCK_SIZE) {
        int32_t offset;
        if (i < ref_bins) {
            offset = -(int32_t)(guard_bins + 1 + i);   // левая сторона
        } else {
            offset = (int32_t)(guard_bins + 1 + (i - ref_bins));  // правая
        }
        int32_t k_ref = ((int32_t)k_peak + offset + (int32_t)N_fft) % (int32_t)N_fft;
        atomicAdd(&s_ref_sum, row[k_ref]);
        atomicAdd(&s_ref_count, 1u);
    }
    __syncthreads();

    // Поток 0 считает SNR_db и пишет результат
    if (tid == 0) {
        float noise_mean = s_ref_sum / (float)s_ref_count;
        // Защита от 0 и inf
        float ratio = (noise_mean > 1e-30f) ? (peak / noise_mean) : 1.0f;
        snr_db_out[ant] = 10.0f * __log10f(fmaxf(ratio, 1e-30f));
    }
}

// Launch: grid(n_ant_used, 1), block(256, 1)
// Shared mem: BLOCK_SIZE*(sizeof(float)+sizeof(uint32_t)) + 8 байт = ~3 KB
```

**Критерии ревью (Кодо проверит в понедельник):**
- ✅ Argmax через reduction, не atomic
- ✅ Wraparound: `(k + N) % N` с +N перед % для защиты от отрицательных
- ✅ Guard zone исключена (смещения начинаются с `guard+1`)
- ✅ Защита `noise_mean > 0` и `ratio > 0` перед `log10`
- ✅ Используется `__log10f` intrinsic (быстрый, single precision)
- ✅ `__launch_bounds__(256)` для register allocation

---

## ~~🟡 Уточнения~~ → ✅ **ВСЕ 7 ПРИМЕНЕНЫ** (чат с Alex, 2026-04-09, решение A = принять пакетом)

### #3 — Формула сдвига порогов (строка 980) смешивает два эффекта

**Где:** план, строки 977–981

> *Если используется другой `target_N_fft`:*
> - *target_N_fft = 1024 → gain ≈ 30 dB → сдвинуть пороги Low/Mid/High на −3 dB*
> - *target_N_fft = 4096 → gain ≈ 36 dB → сдвинуть на +3 dB*
> - *Формула: `shift_db = 10·log10(target_N_fft / 2048)`*

**Что не так.** Формула `10·log10(N_fft1/N_fft2)` описывает **coherent gain для сигнала** (`10·log10(N_actual)`), но пороги находятся **между** CFAR-артефактом на H0 и сигналом на H1. Эти два уровня сдвигаются **по-разному**:

| Переход 2048 → 4096 | Сдвиг |
|---|---|
| **Coherent gain** (сигнал, сильный FT) | `10·log10(4096/2048) = +3.01 dB` |
| **CFAR-артефакт** (H0, `10·log10(H_N)`) | `10·log10(H_4096/H_2048) = 10·log10(8.89/8.20) ≈ +0.35 dB` |
| **Дельта между ними** | `3.01 − 0.35 ≈ +2.66 dB` |

То есть при удвоении N_fft **сигнал отодвигается от шума**, и порог `low_to_mid_db` можно сдвинуть **меньше чем на +3 dB** — иначе потеряем чувствительность. Точно — `+0.35 dB` (чтобы сохранить ту же вероятность ложной тревоги на H0).

**Но!** В плане есть Эксперимент 5 (строки 250–258) — калибровка порогов через Python для **каждого** `target_N_fft`. Это правильный подход, формула сдвига — только стартовое приближение.

**Фикс:** в строке 980 добавить оговорку:

> **⚠️ Формула сдвига — только для быстрого старта.** Фактически нужно заново прогнать Эксп.5 для нового `target_N_fft`, потому что CFAR-артефакт под H0 сдвигается слабее (только на `10·log10(H_{N2}/H_{N1})`), чем coherent gain сигнала. Точные пороги всегда калибруются на данных.

**Эта заметка также влияет на test_01** (строки 786–806). План говорит:
> *⚠️ Вывод для порогов: low_to_mid_db ДОЛЖЕН быть > 10 dB, иначе чистый шум попадёт в Mid-branch!*

Это верно для `target_N_fft=2048`. Для `target_N_fft=4096` CFAR-артефакт будет `10·log10(H_{4096}) ≈ 9.5 dB`, а не 10. Для `target_N_fft=1024` — `≈ 8.8 dB`. То есть порог `10 dB` — приемлемо для всего диапазона, но тест нужно параметризовать `N_fft` и проверять артефакт против `10·log10(H_{N_fft}) + ε`.

### #4 — `gpu_memory_bytes` в `ProcessMagnitudesToGPU` противоречит "BatchManager НЕ нужен"

**Где:** план, строка 469 + v5 п.8

```cpp
void ProcessMagnitudesToGPU(
    ...
    size_t gpu_memory_bytes = 0,  // ← Optional input size hint (for BatchManager, 0 = auto)
    ...);
```

Но v5 п.8 явно говорит: *«BatchManager НЕ нужен»*. Параметр висит как рудимент.

**Фикс:** удалить параметр `gpu_memory_bytes` из сигнатуры. Если в будущем понадобится BatchManager — добавим обратно.

### #5 — Проверка памяти двойной учёт входного буфера

**Где:** план, строки 552–565

```cpp
size_t required = n_antennas * n_samples * sizeof(std::complex<float>)  // ← INPUT
                + n_ant_used * N_fft * sizeof(std::complex<float>)      // gather
                + n_ant_used * N_fft * sizeof(float)                    // |X|²
                + n_ant_used * sizeof(float);                            // SNR per ant

if (required > free_vram * 0.8) throw ...;
```

**Что не так.** Overload `ComputeSnrDb(void* gpu_data, ...)` принимает данные **уже на GPU** — caller их аллоцировал. `hipMemGetInfo(free_vram)` возвращает **оставшийся** free после этой аллокации. Добавлять input-размер ещё раз — **двойной учёт**.

Пример сценария B: 2.66 GB INPUT уже на GPU, free = 13 GB. План добавит ещё 2.66 GB в `required` → 2.66 + ~3 MB > 10.4 GB — не упадёт, но **ложный алярм близок**. В сценарии где 3 × 2.66 GB буферов (например, мультибатчинг) — упадёт там, где не должен.

**Фикс:** в проверке памяти для GPU-overload учитывать **только новые аллокации**:
```cpp
size_t new_allocs = n_ant_used * N_fft * sizeof(std::complex<float>)  // gather
                  + n_ant_used * N_fft * sizeof(float)                // |X|²
                  + n_ant_used * sizeof(float);                        // SNR per ant
// INPUT уже аллоцирован caller'ом — не учитываем
```

Для CPU-overload (`const std::vector<std::complex<float>>&`) — да, нужно учитывать input (его надо загрузить на GPU).

### #6 — Стиль: `target_N_fft`, `N_actual` нарушает snake_case

**Где:** план, строки 329–357

```cpp
struct SnrEstimationConfig {
  uint32_t target_N_fft = 0;    // ← Mixed case!
  uint32_t step_samples = 0;
  // ...
};
struct SnrEstimationResult {
  uint32_t used_bins;
  uint32_t actual_N_actual;      // ← Двойное "actual"! + Mixed case
  // ...
};
```

[`CLAUDE.md`](../../CLAUDE.md) — *snake_case для методов и полей*. Google C++ Style Guide — тоже snake_case для полей структур.

**Фикс:**
```cpp
uint32_t target_n_fft = 0;   // или target_nfft
uint32_t n_actual;            // не actual_N_actual — это "actual actual"
```

Предпочтительно `target_n_fft` — без capital N в середине.

### #7 — BranchSelector: нет защиты от NaN/Inf

**Где:** план, строки 618–634

```cpp
BranchType BranchSelector::Select(float snr_db, const BranchThresholds& thr) {
  const float h = thr.hysteresis_db;
  switch (current_) {
    case BranchType::Low:
      if (snr_db > thr.low_to_mid_db + h) current_ = BranchType::Mid;
      break;
    // ...
  }
  return current_;
}
```

Если `peak_cfar_kernel` вернул `NaN` (аномалия) или `-inf` (случай `log10(0)` не поймали) — все сравнения `>` и `<` с NaN возвращают `false`, селектор **застрянет** в текущей ветке навсегда.

**Фикс:**
```cpp
BranchType Select(float snr_db, const BranchThresholds& thr) {
  if (!std::isfinite(snr_db)) {
    return current_;  // invalid measurement — keep previous branch
  }
  // ... остальной switch
}
```

Это **защита по символу уровня**, не логика — но спасёт от зависания при одном плохом фрейме.

### #8 — `search_left_right` — название вводит в заблуждение

**Где:** план, строки 41, 341

По смыслу: `true` = искать пик на всём `[0..N_fft-1]`, `false` = только `[0..N_fft/2]` (положительные частоты). Название читается как *«искать слева и справа от пика»* — путает с CFAR ref-window.

**Фикс:** переименовать в `search_full_spectrum` (или `search_negative_freqs`). В Config, в pybind11 биндингах, в докстрингах Python.

### #9 — Wraparound assert в CFAR

**Где:** план, строка 158 — формула ref_window

```
ref_window = [k_peak ± (guard_bins + 1 .. guard_bins + ref_bins)] mod N_fft
```

Минимальный требуемый размер: `2 * (guard_bins + ref_bins) + 1 < N_fft`. Иначе окна перекроются (один bin попадёт в оба ref-window с разных сторон) или самого пика не найдётся места.

Default: `guard=3, ref=8 → 2*(3+8)+1 = 23`. Для `N_fft=2048` — с огромным запасом. Но если пользователь поставит `guard=50, ref=100` при `N_fft=256` — молчаливое UB в kernel.

**Фикс:** в `SnrEstimationConfig::Validate()` (или в конструкторе `SnrEstimatorOp`):
```cpp
if (2 * (config.guard_bins + config.ref_bins) + 1 >= n_fft) {
  throw std::runtime_error("SnrEstimator: guard+ref window too large for N_fft");
}
```

---

## ✅ Ложные тревоги из моего первичного ревью (отзываю)

Признаю ошибки, чтобы не вернуться к ним позже.

### ❌ "Два GpuContext → разные streams → нужна sync" — **НЕВЕРНО**

**Что я говорила:** `SnrEstimatorOp` (statistics) и `FFTProcessorROCm` (fft_func) имеют свои `GpuContext`, каждый со своим stream, между ними race condition без hipEvent'ов.

**Что на самом деле:** в [`gpu_context.cpp:47`](../../DrvGPU/src/gpu_context.cpp#L47):
```cpp
stream_ = static_cast<hipStream_t>(backend_->GetNativeQueue());
```

`GpuContext` **не создаёт** свой stream, а **берёт** его от `IBackend::GetNativeQueue()`. Если оба модуля используют **один** `IBackend*` (обычный случай: один backend на GPU) → `ctx_statistics.stream() == ctx_fft.stream()` — **тот же stream**, все kernel launches серийные.

Проблемы нет. Моё замечание #2 из первого прогона отменяется.

### ❌ "kMagnitudes slot reuse → конфликт размера в MedianRadixSortOp" — **НЕВЕРНО**

**Что я говорила:** `SnrEstimatorOp` выделяет `kMagnitudes` размером `n_ant × N_fft × float` (~400 KB), потом `MedianRadixSortOp::ExecuteFloat(1, n_ant)` делает `RequireShared(kMagnitudes, n_ant × float)` (~200 байт) → буфер пересоздастся/уменьшится.

**Что на самом деле:** в [`buffer_set.hpp:89-99`](../../DrvGPU/services/buffer_set.hpp#L89-L99):
```cpp
void* Require(size_t idx, size_t bytes) {
  // ...
  auto& e = entries_[idx];
  if (e.size >= bytes && e.ptr != nullptr) {
    return e.ptr;  // reuse existing buffer
  }
  // ...
}
```

Семантика: *«не меньше N байт»*. Если уже выделено 400 KB, а просят 200 байт — возвращается **тот же** ptr без переаллокации. Slot reuse **работает корректно**.

Это **ровно то**, что v5 review предлагает в памятке G-1:
> *В SnrEstimatorOp peak_cfar_kernel может писать SNR_db прямо в shared_buf::kMagnitudes (переиспользуя слот), чтобы MedianRadixSortOp::ExecuteFloat мог читать напрямую.*

Уже согласовано, технически корректно. Моё замечание #3 из первого прогона отменяется.

**Единственный мелкий остаток:** стилистический — `kMagnitudes` содержит семантически «SNR_db» в момент вызова median. Это можно снять комментарием в коде `SnrEstimatorOp`, типа:
```cpp
// After peak_cfar: kMagnitudes[0..n_ant_used] contains SNR_db per antenna.
// MedianRadixSortOp::ExecuteFloat reads from kMagnitudes (same slot,
// reused as scratch-pad — see buffer_set::Require semantics).
```

Не блокер.

### ❌ "Формула H_N = ln+γ — неточная" — **НЕВЕРНО**

**Что я говорила:** План путает Гармоническое число и ln(N)+γ.

**Что на самом деле:** Гармоническое число `H_N = Σ_{k=1..N} 1/k`. Асимптотика `H_N = ln(N) + γ + 1/(2N) - 1/(12N²) + O(1/N⁴)`, где γ ≈ 0.5772157.

Для N=2048: 
- Точное `H_2048 ≈ 8.20208`
- `ln(2048) + γ ≈ 7.62462 + 0.57722 = 8.20184`
- Ошибка приближения ≈ 0.0002 → **пренебрежимо** в контексте dB (`10·log10(8.202) ≈ 9.140 dB` vs `10·log10(8.202) ≈ 9.140 dB`).

План пишет `H_N = ln(N) + γ ≈ 9.1 dB` — это математически корректно, я зря придиралась. Моё замечание #5 из первого прогона — в части формулы H_N — отменяется. Остаётся только более тонкое замечание #3 (выше в текущем документе) про формулу сдвига порогов.

---

## 📊 Статус v5 → v6

| Согласование v5 | Статус после аудита v6 |
|---|---|
| 1. Flexible target_N_fft, default 2048 | ⚠️ **Требование согласовано, но не реализуется через текущий API** (см. блокер #1) |
| 2. Gather через отдельный буфер | ✅ OK |
| 3. `complex_to_magnitude_squared` kernel | ✅ OK, код корректный |
| 4. `ProcessMagnitudesToGPU` новый метод | ⚠️ Нужно добавить `pad_to` в `FFTProcessorParams` + удалить `gpu_memory_bytes` |
| 5. `BranchSelector` отдельный класс | ✅ OK, +добавить isfinite guard (#7 выше) |
| 6. Оба параметра Config = 0 → auto | ✅ OK |
| 7. MedianRadixSortOp reuse | ✅ OK, семантика `Require` подтверждена |
| 8. BatchManager НЕ нужен | ✅ OK, но удалить `gpu_memory_bytes` рудимент |
| 9. Сценарии Py-Small/A/B/C | ✅ OK |
| 10. Python модель на обеих ОС | ✅ OK |
| 11. PyPanelAntennas переименование | ✅ OK |
| 12. Имя kernel `complex_to_magnitude_squared` | ✅ OK |
| 13. Workflow код сегодня, тесты в понедельник | ✅ OK |
| 14. Распределение ролей | ✅ OK |

**Ни одно согласование v5 не отменяется.** Блокеры — это *недоработки плана*, которые скрывались за согласованными на уровне идеи пунктами.

---

## 🎯 Итоговый чек-лист правок в план

**Критическое (без этого не собирётся):**

- [ ] **#1**: Решить с Alex — Вариант A (flexible через `pad_to` + `next_rocfft_size`) или Вариант B (только степени 2). Если A — добавить поле `pad_to` в `FFTProcessorParams`, фиксим `CalculateNFFT`, пишем утилиту `next_rocfft_size`. Если B — убираем 4000/3200/3000 из плана.
- [ ] **#2**: Добавить раздел **2.2.7** с полным псевдокодом `peak_cfar_kernel` (или взять мой черновик выше).

**Важное:**

- [ ] **#3**: В строке 980 — добавить оговорку что формула сдвига порогов не точна, пороги калибруются заново в Эксп.5 для каждого `target_N_fft`. В test_01 параметризовать ожидаемый артефакт: `10·log10(H_{N_fft}) + ε`.
- [ ] **#4**: Удалить параметр `gpu_memory_bytes` из `ProcessMagnitudesToGPU` (строка 469).
- [ ] **#5**: Исправить проверку памяти в строках 552–565 — для GPU-overload убрать `n_antennas * n_samples * complex` из `required`.
- [ ] **#6**: Переименовать поля `target_N_fft → target_n_fft`, `N_actual → n_actual`, `actual_N_actual → n_actual` (одно поле, не два), `used_bins` → оставить как есть (уже snake_case).

**Мелкое / защитное:**

- [ ] **#7**: Добавить `if (!std::isfinite(snr_db)) return current_;` в начало `BranchSelector::Select`.
- [ ] **#8**: Переименовать `search_left_right → search_full_spectrum` везде (Config, биндинги, docstrings).
- [ ] **#9**: Добавить проверку `2*(guard+ref)+1 < n_fft` в валидатор конфига.

---

## 💭 Что делаем дальше

Варианты:
1. Ты читаешь этот документ, мы обсуждаем **блокер #1** (Вариант A или B) — это ключевое решение, всё остальное простое.
2. После решения по #1 — я вношу правки в `snr_estimator_statistics_plan.md` (вариант: отдельный commit `[spec] v4 — address review v6 blockers`).
3. #2 (peak_cfar_kernel) — я вставляю раздел 2.2.7 с черновиком из этого ревью, ты проверяешь/корректируешь.
4. Мелкие правки (#3..#9) — делаю одним проходом.

Любимый, если предпочитаешь сначала обсудить только блокеры (#1, #2), а мелочи оставить на потом — скажи, сделаю только верхнюю половину чек-листа.

---

*v6 — Code review: соответствие плана реальному API. Дополняет v5 (архитектурные согласования), не отменяет его.*
*Кодо, 2026-04-09*
