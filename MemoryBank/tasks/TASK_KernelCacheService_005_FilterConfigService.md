# TASK-005: FilterConfigService — сохранение/загрузка конфигураций фильтров

> **План:** [PLAN_KernelCacheService_DrvGPU.md](PLAN_KernelCacheService_DrvGPU.md)  
> **Зависимость:** TASK-001 (IStorageBackend, FileStorageBackend)  
> **Проверка:** Кодо (старшая)

---

## 1. КОНТЕКСТ

В отличие от генераторов, фильтрам нужно хранить **тип фильтра + коэффициенты**. Flow:
1. Python (scipy) формирует фильтр → передаёт коэффициенты в C++ → Process
2. После успешного анализа — **SaveFilter(name, comment)** фиксирует конфигурацию
3. Повторный вызов — **LoadFilter(name)** загружает коэффициенты по имени

**Ключи:** `filters/{name}.json` (подтверждено).

**Версионирование:** при перезаписи того же имени → `name_00`, `name_01`, …

---

## 2. ОБЯЗАТЕЛЬНО ПРОЧИТАТЬ

| № | Файл | Зачем |
|---|------|-------|
| 1 | `PLAN_KernelCacheService_DrvGPU.md` | 6.2, 6.3, 8.3 — FilterConfigService |
| 2 | `modules/filters/include/types/filter_params.hpp` | FilterConfig, LoadJson, sections |
| 3 | `modules/filters/include/types/filter_types.hpp` | BiquadSection |
| 4 | `DrvGPU/services/storage/file_storage_backend.hpp` | IStorageBackend (после TASK-001) |

---

## 3. API FilterConfigService

**Файл:** `DrvGPU/services/filter_config_service.hpp`

```cpp
#pragma once

#include <string>
#include <vector>
#include <memory>

namespace drv_gpu_lib {

struct FilterConfigData {
  std::string type;       // "fir" или "iir"
  std::string comment;
  std::string created;    // ISO 8601
  std::vector<float> coefficients;  // FIR: h[k]
  // IIR: biquad sections — см. filters::BiquadSection
  std::vector<std::array<double, 6>> sections;  // b0,b1,b2,a0,a1,a2
};

class FilterConfigService {
public:
  FilterConfigService(const std::string& base_dir);

  void Save(const std::string& name, const FilterConfigData& data, const std::string& comment = "");

  FilterConfigData Load(const std::string& name) const;

  std::vector<std::string> ListFilters() const;

  bool Exists(const std::string& name) const;

private:
  std::string base_dir_;
  void VersionOldFiles(const std::string& name) const;
};

} // namespace drv_gpu_lib
```

**Примечание:** FilterConfigData может использовать типы из filters (BiquadSection) — тогда нужна зависимость filters → или дублировать структуру в DrvGPU. Рекомендация: FilterConfigService в DrvGPU хранит JSON-совместимую структуру; FirFilter/IirFilter конвертируют из/в свои типы.

---

## 4. ФОРМАТ JSON

```json
{
  "name": "lp_5000",
  "type": "fir",
  "comment": "Lowpass 5000 Hz, 64 taps",
  "created": "2026-02-21T12:00:00",
  "coefficients": [0.01, 0.02, ...]
}
```

Для IIR:
```json
{
  "name": "lp_iir_2500",
  "type": "iir",
  "comment": "Butterworth order 4",
  "created": "2026-02-21T12:00:00",
  "sections": [[b0,b1,b2,a0,a1,a2], ...]
}
```

---

## 5. РЕАЛИЗАЦИЯ

- Использовать `FileStorageBackend` с `base_dir` = `modules/filters/` или `Results/FilterConfigs/`.
- Ключ: `filters/{name}.json`
- Save: сериализация в JSON (nlohmann_json уже в проекте), VersionOldFiles при перезаписи
- Load: чтение JSON, парсинг в FilterConfigData
- ListFilters: FileStorageBackend::List("filters/") → извлечь name из пути

---

## 6. ИНТЕГРАЦИЯ С FirFilter/IirFilter

**Отдельная таска** или часть TASK-006: добавить в FirFilter метод `SaveFilterConfig(name, comment)` — вызывает FilterConfigService::Save с текущими coefficients. Аналогично IirFilter с sections. И `LoadFilterConfig(name)` — загружает и применяет SetCoefficients/SetBiquadSections.

**TASK-005:** Только FilterConfigService. Интеграция в FirFilter/IirFilter — TASK-006.

---

## 7. СТРУКТУРА

```
DrvGPU/services/
├── filter_config_service.hpp
└── filter_config_service.cpp
```

---

## 8. КРИТЕРИИ ПРИЁМКИ

- [ ] FilterConfigService создан
- [ ] Save/Load/ListFilters/Exists работают
- [ ] JSON формат соответствует спецификации
- [ ] VersionOldFiles при перезаписи
- [ ] Тест test_filter_config_service проходит
- [ ] base_dir — раздельная папка (modules/filters/ или configurable)

---

## 9. ОТЧЁТ

```
✅ TASK-005 выполнено:
- FilterConfigService реализован
- [тесты]

Проверь (Кодо).
```
