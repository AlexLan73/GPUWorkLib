# DrvGPU/interface/

Директория для абстрактных интерфейсов (чистые виртуальные классы).

## Содержимое

| Файл | Описание | Источник |
|------|----------|----------|
| i_backend.hpp | IBackend - абстракция GPU бэкенда | common/ (redirect) |
| i_compute_module.hpp | IComputeModule - интерфейс модуля | common/ (redirect) |
| i_logger.hpp | ILogger - интерфейс логгера | common/logger_interface.hpp (redirect) |
| i_memory_buffer.hpp | IMemoryBuffer - интерфейс буфера | memory/ (redirect) |
| i_data_sink.hpp | IDataSink - интерфейс приёмника данных | **НОВЫЙ** |

## Использование

```cpp
// Новый код (предпочтительно)
#include "interface/i_backend.hpp"
#include "interface/i_compute_module.hpp"
#include "interface/i_logger.hpp"

// Старый код (обратная совместимость)
#include "common/i_backend.hpp"
```

## План миграции

1. [x] Создать redirect-заголовки
2. [ ] Физически переместить файлы из common/
3. [ ] Обновить все #include пути
4. [ ] Удалить redirect'ы из common/
