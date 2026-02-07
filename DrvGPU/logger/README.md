# DrvGPU/logger/

Директория для системы логирования.

## Статус миграции

| Файл | Текущее расположение | Статус |
|------|---------------------|--------|
| logger.hpp | common/ | Redirect создан |
| logger.cpp | common/ | Ожидает миграции |
| logger_interface.hpp | common/ | Redirect создан |
| default_logger.hpp | common/ | Redirect создан |
| default_logger.cpp | common/ | Ожидает миграции |
| config_logger.hpp | common/ | Redirect создан |
| config_logger.cpp | common/ | Ожидает миграции |

## Использование

Новый код:
```cpp
#include "logger/logger.hpp"
```

Старый код (обратная совместимость):
```cpp
#include "common/logger.hpp"
```

## План миграции

1. [x] Создать redirect-заголовки в logger/
2. [ ] Физически переместить файлы из common/ в logger/
3. [ ] Обновить redirect'ы в common/ → указывать на logger/
4. [ ] Обновить все #include пути
