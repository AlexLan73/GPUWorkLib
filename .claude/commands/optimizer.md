---
description: Анализ и оптимизация HIP/ROCm/OpenCL ядер модуля GPUWorkLib
---

Используй агент gpu-optimizer для оптимизации: $ARGUMENTS

Если модуль/файл не указан — спроси что оптимизировать.

Алгоритм:
1. Прочитай Doc_Addition/Info_ROCm_HIP_Optimization_Guide.md — теория и паттерны
2. Прочитай modules/vector_algebra/ как эталон реализации
3. Найди ядра модуля:
   - modules/$ARGUMENTS/kernels/**/*.cl  (OpenCL)
   - modules/$ARGUMENTS/kernels/**/*.hip (ROCm/HIP)
4. Проверь по чеклисту:
   - __launch_bounds__ на всех ядрах?
   - LDS bank conflicts (нужен +1 padding)?
   - __fsqrt_rn, __atan2f, native_sqrt вместо медленных?
   - Warp shuffle вместо LDS для финальной редукции?
   - 2D grid чтобы убрать div/mod?
   - KernelCacheService + --offload-arch + -O3?
   - __restrict на pointer параметрах?
5. Выдай конкретные правки с указанием файл:строка
6. Оцени ожидаемый прирост производительности

Приоритет: 🔴 критично → 🟠 важно → 🟡 желательно
