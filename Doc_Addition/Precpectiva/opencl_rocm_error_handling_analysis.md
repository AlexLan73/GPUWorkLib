<!-- ДАЛЬШЕ ПЕРЕНОСИТЬ: Doc_Addition/Info_OpenCL_ROCm_Error_Handling.md -->

# Обработка ошибок OpenCL & ROCm — анализ

> Источник: MemoryBank/specs/Precpectiva/opencl_rocm_error_handling_analysis.md  
> Дата: 2026-02-11

---

## Текущее состояние в GPUWorkLib

| Паттерн | Где | Пример |
|---------|-----|--------|
| `if (err != CL_SUCCESS) throw std::runtime_error(...)` | Большинство модулей | spectrum_maxima_finder.cpp |
| `CheckCLError(err, "operation")` | opencl_core, external_cl_buffer_adapter | opencl_core.hpp |
| Прямой throw | Валидация | Модули, DrvGPU |

**Проблема**: код ошибки передаётся как число без расшифровки (CL_INVALID_VALUE и т.д.).

**Где нет единообразия**: spectrum_maxima_finder, spectrum_processor_opencl — вручную; clFFT — отдельный enum; ROCm/HIP — hipError_t.

---

## Паттерны из Khronos / OpenCL SDK

- CL_CHECK (return error), Error-класс (exception с кодом), print_error + ранний выход.
- Рекомендация: единый CheckCLError с расшифровкой кода (clGetErrorName/clGetErrorString или таблица).

**Полный анализ и предложения** — в исходном файле MemoryBank/specs/Precpectiva/opencl_rocm_error_handling_analysis.md (части 2–5).
