# 📝 DrvGPU — Спецификация

> **Модуль**: `DrvGPU`
> **Статус**: 🟢 Active
> **Платформы**: OpenCL, ROCm, HIP
> **Автор**: Alex
> **Создано**: 2026-01-01
> **Обновлено**: 2026-02-09

---

## 🎯 Назначение

Базовый драйвер GPU — унифицированный слой абстракции для работы с разными GPU-платформами.

**Ключевые компоненты:**
- Backend abstraction (OpenCL, ROCm, HIP)
- Memory management (ZeroCopy, Device, Pinned)
- Logging system (plog-based, per-GPU)
- Profiling system (GPUProfiler)
- Configuration (configGPU.json)

---

## 📋 Требования

### Функциональные
- [x] REQ-001: Абстракция backend (IBackend interface)
- [x] REQ-002: Multi-GPU support (до 32 устройств)
- [x] REQ-003: Per-GPU логирование (DefaultLogger)
- [x] REQ-004: GPU Profiler с async сбором данных
- [x] REQ-005: JSON-конфигурация (configGPU.json)
- [x] REQ-006: Console output service

### Нефункциональные
- [x] NFR-001: Потокобезопасность всех сервисов
- [x] NFR-002: Минимальный overhead логирования (<1μs)
- [x] NFR-003: Кроссплатформенность (Windows/Linux)

---

## 🔧 Структура

```
DrvGPU/
├── config/
│   ├── config_types.hpp      # GPUDeviceConfig struct
│   ├── configGPU.json        # Runtime configuration
│   └── gpu_config.hpp/cpp    # GPUConfig singleton
│
├── interface/
│   ├── i_backend.hpp         # IBackend interface
│   └── i_logger.hpp          # ILogger interface
│
├── logger/
│   ├── config_logger.hpp/cpp # ConfigLogger (paths, enabled)
│   └── default_logger.hpp/cpp # DefaultLogger (plog-based)
│
├── services/
│   ├── console_output.hpp    # ConsoleOutput service
│   ├── gpu_profiler.hpp      # GPUProfiler service
│   └── service_manager.hpp   # ServiceManager
│
├── backends/
│   └── opencl_backend.hpp/cpp # OpenCL implementation
│
└── src/
    └── drv_gpu.cpp           # Main DrvGPU class
```

---

## ⚙️ Конфигурация (configGPU.json)

```json
{
  "version": "1.0",
  "gpus": [
    {
      "id": 0,
      "name": "GPU 0",
      "is_prof": true,      // GPUProfiler enabled
      "is_logger": true,    // File logging enabled
      "is_console": true,   // Console output enabled
      "is_active": true,    // GPU active for compute
      "is_db": false        // Database logging (future)
    }
  ]
}
```

---

## 📝 История изменений

| Дата | Автор | Изменение |
|------|-------|-----------|
| 2026-02-09 | Кодо | Logger path fix (filesystem::path) |
| 2026-02-09 | Кодо | Logger linked to is_logger from config |
| 2026-02-09 | Кодо | Custom plog formatter (removed ThreadID) |
| 2026-02-09 | Кодо | GPUProfiler: delta times instead of absolute |
