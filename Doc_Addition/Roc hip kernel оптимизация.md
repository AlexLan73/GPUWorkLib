<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Roc hjp kernel на что нужно обращать внимание когда оптимизирую код на быстрое выполнение и как задать правильное задание Claude Opus 4.6

Александр, вот исчерпывающий обзор по обоим вопросам — оптимизация HIP-ядер для ROCm и правильное формулирование заданий для Claude Opus 4.6.

***

# Оптимизация HIP-ядер для ROCm: ключевые факторы

## 1. Профилирование — всегда первый шаг

Прежде чем что-либо оптимизировать, необходимо определить, является ли ядро **memory-bound**, **compute-bound** или **latency-bound**. Для этого используется `rocprofv3`[^1][^2]:

```bash
rocprofv3 --stats --<tracing_option> -- ./your_application
```

Ключевые метрики: время выполнения ядра, пропускная способность памяти, occupancy, утилизация CU. Визуализация трейсов доступна через Perfetto. Модель roofline помогает определить, к какому типу относится ядро — по формуле: **Arithmetic Intensity = Arithmetic Operations / Bytes moved**[^2].

## 2. Оптимизация доступа к памяти

Это критически важная область, т.к. большинство HPC-ядер — memory-bound[^2].

- **Coalesced memory access** — потоки внутри wavefront должны обращаться к последовательным адресам. Каждый нековалесцированный доступ порождает дополнительные транзакции, резко снижающие bandwidth[^1].
- **Использование векторных типов** (`float4`, `float2`) — компилятор генерирует меньше, но более широких load-инструкций, амортизируя стоимость расчёта адресов[^2].
- **Выравнивание** — L1 cacheline: 64 байта на MI200, 128 байт на MI300. Паддинг 2D-массивов до кратного warp size гарантирует выровненные строки[^1][^2].
- **Shared memory (LDS)** — стагировать данные в LDS для повторного использования. Но помнить о bank conflicts — паддинг `[^32][^33]` вместо `[^32][^32]`[^1].
- **Минимизация host↔device трансферов** — батчить мелкие копирования, использовать pinned memory (`hipHostMalloc`)[^1].


## 3. Occupancy и давление на регистры

Occupancy — это отношение резидентных wavefronts к максимально возможному числу на CU. Чем выше occupancy, тем лучше GPU скрывает латентность[^2][^3].

На MI200 каждый CU поддерживает до 32 wavefronts (8 на SIMD). Ключевые ограничители:


| VGPRs на поток | Waves/SIMD | Waves/CU |
| :-- | :-- | :-- |
| ≤ 64 | 8 | 32 |
| ≤ 96 | 5 | 20 |
| ≤ 128 | 4 | 16 |
| ≤ 256 | 2 | 8 |
| > 256 | 1 | 4 |

[^2][^3]

Практические рекомендации:

- **Определять переменные как можно ближе к месту использования** — это позволяет компилятору освобождать регистры раньше[^3].
- **Избегать register spilling** в scratch memory — это катастрофически медленно[^2].
- **Использовать `__launch_bounds__`** — подсказать компилятору ожидаемый размер блока, чтобы он правильнее распределял регистры[^1][^2].
- **Проверять расход регистров**: `hipcc --resource-usage kernel.hip`[^1].
- **LDS также ограничивает occupancy** — 64 KiB на CU у MI200. Если workgroup использует 48 KiB LDS, на CU помещается всего 1 workgroup[^2].


## 4. Compute-bound оптимизации

- **FP32 вместо FP64** — throughput FP32 значительно выше. Критическая ошибка: литералы `0.3` (double) вместо `0.3f` (float) вызывают неявные конвертации и FP64-инструкции[^2]:

```c
// ПЛОХО — конвертация в double и обратно:
float f1 = a[threadIdx.x] * 0.3;
// ХОРОШО — всё остаётся в FP32:
float f1 = a[threadIdx.x] * 0.3f;
```

- **Быстрые интринсики**: `__cosf(x)` вместо `cosf(x)`, `__expf(x)` вместо `expf(x)`[^1][^2].
- **Умножение вместо деления**: `value * 0.5f` вместо `value / 2.0f`[^1].
- **Битовые операции** для степеней двойки: `threadIdx.x << 2` вместо `threadIdx.x * 4`[^1].
- **`a * a * a` вместо `pow(a, 3.0f)`** — две инструкции вместо многих[^2].


## 5. Минимизация thread divergence

Когда потоки внутри wavefront идут по разным веткам, GPU сериализует выполнение[^1][^2]. Решение — restructure код так, чтобы все потоки wavefront шли по одной ветке, или использовать predication (branchless programming)[^2].

## 6. Fusion ядер и запуск

Слияние мелких ядер в одно: уменьшает launch overhead (который заметен при ядрах << 1 мс) и уменьшает data movement[^2]. Также стоит использовать стримы для перекрытия копирования и вычислений[^1].

## 7. Особенности AMD vs NVIDIA

Исследование auto-tuning показало, что на AMD GPU тюнинг даёт **до 10x** прироста (vs ~2x на NVIDIA), и конфигурации, оптимальные для NVIDIA, плохо работают на AMD[^4][^5]. Это подчёркивает необходимость профилирования и тюнинга именно под конкретную архитектуру.

***

# Как правильно формулировать задания для Claude Opus 4.6

Claude Opus 4.6 — это самая новая модель Anthropic с улучшенным следованием инструкциям, adaptive thinking и усиленной проактивностью[^6]. Вот ключевые принципы для задач оптимизации кода:

## Структура промпта

Используйте **XML-теги** для чёткого разделения блоков промпта — это снижает неоднозначность[^6][^7]:

```xml
<context>
У меня есть HIP-ядро для радарной обработки сигналов (БПФ + 
доплеровская фильтрация) на AMD Radeon RX 7900 XTX (RDNA3). 
Ядро обрабатывает комплексные вектора float2, размер данных 
4096x256 элементов. Текущее время выполнения — 12 мс.
</context>

<current_code>
// ... ваш текущий код ядра
</current_code>

<profiling_data>
VGPRs: 112, occupancy: 4 waves/SIMD, 
bandwidth utilization: 45% от пикового
</profiling_data>

<task>
Оптимизируй это ядро для минимального времени выполнения.
Сфокусируйся на: coalesced memory access, снижение register 
pressure, использование shared memory для повторного 
использования данных. Покажи оптимизированный код 
с комментариями, объясняющими каждую оптимизацию.
</task>

<constraints>
- Не используй FP64, все вычисления в FP32
- Целевая occupancy: минимум 5 waves/SIMD
- Совместимость с ROCm 6.x
</constraints>
```


## Ключевые принципы для Claude 4.6

**Будьте явными и директивными**, а не расплывчатыми. Claude 4.6 отлично следует точным инструкциям, но может промахнуться по vague-запросам[^6][^7]:

- ❌ «Сделай мой код быстрее»
- ✅ «Оптимизируй memory access pattern для coalescing, замени `pow(x, 2.0f)` на `x * x`, добавь LDS-тайлинг с размером блока 256»

**Объясняйте мотивацию (why)** — Claude обобщает из объяснений лучше, чем из голых правил[^7]:

```
Мне важно снизить register pressure, потому что текущий расход 
112 VGPR ограничивает occupancy до 4 waves/SIMD. Мне нужно 
уложиться в 96 VGPR, чтобы получить 5 waves/SIMD.
```

**Давайте примеры (few-shot)** — Claude 4.x модели очень внимательны к деталям в примерах[^8][^7]. Покажите пример входного кода и ожидаемого результата.

**Не используйте «КРИТИЧЕСКО ВАЖНО!!!» и агрессивные формулировки** — Claude 4.6 гораздо более отзывчив к system prompt, и агрессивные инструкции приводят к overtriggering. Вместо «CRITICAL: You MUST...» используйте «Use ... when ...»[^6].

**Просите имплементацию, а не предложения** — Claude 4.6 может интерпретировать «можешь ли ты предложить улучшения?» буквально и только предложить, не реализовав. Используйте: «Реализуй эти изменения в коде»[^6].

**Управляйте overengineering** — Claude 4.6 склонен к избыточным абстракциям. Добавьте[^6]:

```xml
<constraints>
Не добавляй лишних абстракций. Не рефактори код за пределами 
запрошенных изменений. Минимальные изменения для достижения цели.
</constraints>
```


## Пример полного промпта для оптимизации HIP-ядра

```xml
Ты — эксперт по GPU-оптимизации для AMD ROCm/HIP.

<context>
Проект: радарная обработка сигналов (ЛЧМ-импульсное сжатие).
GPU: AMD Instinct MI210 (CDNA2), ROCm 6.2.
Wavefront size: 64. LDS: 64 KiB/CU. Max VGPRs: 512/CU.
</context>

<current_kernel>
__global__ void pulse_compression(
    const float2* __restrict__ input,
    const float2* __restrict__ reference,
    float2* __restrict__ output,
    int N) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    float2 a = input[idx];
    float2 b = reference[idx];
    
    // complex multiply
    float2 result;
    result.x = a.x * b.x - a.y * b.y;
    result.y = a.x * b.y + a.y * b.x;
    
    output[idx] = result;
}
</current_kernel>

<profiling>
Kernel time: 0.8ms, VGPRs: 24, SGPRs: 16, 
occupancy: 8 waves/SIMD, memory bandwidth: 60% of peak.
Block size: 256 threads.
</profiling>

<task>
1. Проанализируй узкие места этого ядра.
2. Предложи и реализуй конкретные оптимизации для 
   увеличения memory throughput (ядро memory-bound).
3. Покажи оптимизированный код с комментариями.
4. Укажи ожидаемый эффект каждой оптимизации.
</task>

<rules>
- Только FP32, никаких double-литералов.
- Используй __restrict__ для всех указателей.
- Применяй vectorized loads (float4) где возможно.
- Покажи launch configuration (grid, block sizes).
</rules>
```


## Дополнительные техники

- **Adaptive thinking** — Claude Opus 4.6 использует `thinking: {type: "adaptive"}`, автоматически калибруя глубину рассуждений. Для сложных задач оптимизации используйте `effort: "high"` или `"max"`[^6].
- **Prompt chaining** — для сложных оптимизаций разбейте на шаги: (1) анализ профайлера → (2) оптимизация memory → (3) оптимизация compute → (4) финальная ревизия[^7].
- **Просите Claude не спекулировать** — «Если ты не уверен в точном поведении компилятора для этого паттерна, скажи об этом прямо»[^6][^7].

Эти подходы в комбинации — структурированный промпт с контекстом, данными профайлера и конкретными ограничениями — дадут значительно более качественный результат, чем generic-запрос вида «оптимизируй мой GPU-код».
<span style="display:none">[^10][^11][^12][^13][^14][^15][^16][^17][^18][^19][^20][^21][^22][^23][^24][^25][^26][^9]</span>

<div align="center">⁂</div>

[^1]: https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html

[^2]: https://fs.hlrs.de/projects/par/events/2025/GPU-AMD/day2/08.HIP_Optimization.pdf

[^3]: https://rocm.blogs.amd.com/software-tools-optimization/register-pressure/README.html

[^4]: https://www.themoonlight.io/en/review/bringing-auto-tuning-to-hip-analysis-of-tuning-impact-and-difficulty-on-amd-and-nvidia-gpus

[^5]: https://arxiv.org/abs/2407.11488v1

[^6]: https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices

[^7]: https://claude.com/blog/best-practices-for-prompt-engineering

[^8]: https://www.dreamhost.com/blog/claude-prompt-engineering/

[^9]: https://www.vincirufus.com/posts/claude-4-prompt-engineering-best-practices/

[^10]: https://rocm.docs.amd.com/projects/HIP/en/develop/how-to/performance_guidelines.html

[^11]: https://github.com/johnpsasser/claude-code-prompt-optimizer

[^12]: https://rocm.docs.amd.com/projects/HIP/en/docs-6.1.5/how-to/performance_guidelines.html

[^13]: https://www.iweaver.ai/blog/claude-4-models-demystified-use-cases-prompt-tricks-and-avoiding-pitfalls/

[^14]: https://rocm.blogs.amd.com/blog/2025.html

[^15]: https://arxiv.org/html/2407.11488v1

[^16]: https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview

[^17]: https://www.ccs.tsukuba.ac.jp/wp-content/uploads/sites/14/2025/09/05.-Introduction-to-HIP_and_ROCm.pdf

[^18]: https://github.com/ROCm/hipBench

[^19]: https://github.com/ThamJiaHe/claude-prompt-engineering-guide

[^20]: https://www.anthropic.com/engineering/claude-code-best-practices?curius=2107

[^21]: https://www.datastudios.org/post/claude-ai-prompting-techniques-structure-examples-and-best-practices

[^22]: https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-register-pressure-readme/

[^23]: https://www.youtube.com/watch?v=pb0lVGDiigI\&vl=ru

[^24]: https://www.youtube.com/watch?v=7WuKgc3-_-s

[^25]: https://code.claude.com/docs/en/best-practices

[^26]: https://github.com/ROCm/rocm-blogs/blob/release/blogs/software-tools-optimization/amdgcn-isa/README.md

