# Ref05: Logger — неправильный путь к файлам логов

**Дата:** 2026-02-09
**Статус:** В работе
**Приоритет:** Высокий

---

## Проблема

Logger создаёт файлы логов с **неправильными именами** вместо правильной структуры директорий:

**Текущее (неправильное) поведение:**
```
GPUWorkLib92Logs92DRVGPU_00922026-02-099221-17-52.log
GPUWorkLib92Logs92DRVGPU_00922026-02-099221-10-28.log
```

**Ожидаемое (правильное) поведение:**
```
GPUWorkLib/
  Logs/
    DRVGPU_00/
      2026-02-09/
        21-17-52.log
        21-10-28.log
```

Файлы создаются **в корне** вместо вложенной структуры.
Символ `\` (код 92) выводится как число `92`, а не как разделитель пути.

---

## Анализ причины

### Локализация проблемы

**Файл:** `DrvGPU/logger/config_logger.cpp`
**Функция:** `GetLogFilePathForGPU(int gpu_id)` (строки 202-247)

### Проблемный код (строки 236-244):

```cpp
std::ostringstream path_ss;
path_ss << base_path;
if (!base_path.empty() && base_path.back() != '/' && base_path.back() != '\\') {
    path_ss << std::filesystem::path::preferred_separator;  // ❌ ПРОБЛЕМА!
}
path_ss << kLogsDir << std::filesystem::path::preferred_separator;      // ❌
path_ss << gpu_subdir << std::filesystem::path::preferred_separator;    // ❌
path_ss << date_str << std::filesystem::path::preferred_separator;      // ❌
path_ss << time_str << ".log";
```

### Причина ошибки

На Windows `std::filesystem::path::preferred_separator` имеет тип `wchar_t` со значением `L'\\'` (код 92).

Когда `wchar_t` передаётся в `operator<<` для `std::ostringstream`, он **интерпретируется как число**, а не как символ!

**Результат:**
```
path_ss << "Logs" << std::filesystem::path::preferred_separator;
// Ожидание: "Logs\"
// Реальность: "Logs92"
```

Полный путь `GPUWorkLib\Logs\DRVGPU_00\2026-02-09\21-17-52.log` превращается в строку `GPUWorkLib92Logs92DRVGPU_00922026-02-099221-17-52.log`, которая используется как **имя файла** в текущей директории.

---

## Решение

### Вариант A: Использовать std::filesystem::path (рекомендуется)

```cpp
std::string ConfigLogger::GetLogFilePathForGPU(int gpu_id) const {
    // ... получение date_str, time_str, gpu_subdir ...

    // Базовый путь
    std::filesystem::path base_path_fs = log_path_;
    if (log_path_.empty()) {
        base_path_fs = std::filesystem::current_path();
    }

    // Конкатенация через operator/= (автоматически добавляет правильный разделитель)
    std::filesystem::path full_path = base_path_fs;
    full_path /= kLogsDir;           // Logs
    full_path /= gpu_subdir;         // DRVGPU_00
    full_path /= date_str;           // 2026-02-09
    full_path /= time_str + ".log";  // 21-17-52.log

    return full_path.string();
}
```

**Преимущества:**
- Кроссплатформенность (Windows/Linux/macOS)
- Автоматическая нормализация путей
- Безопасность от edge cases (двойные слэши и т.д.)

### Вариант B: Использовать прямой слэш `/`

```cpp
path_ss << kLogsDir << "/";
path_ss << gpu_subdir << "/";
path_ss << date_str << "/";
path_ss << time_str << ".log";
```

**Примечание:** Windows понимает `/` как разделитель пути, но это менее "чисто".

---

## План изменений

### Файлы для изменения

| Файл | Изменение |
|------|-----------|
| `DrvGPU/logger/config_logger.cpp` | Переписать `GetLogFilePathForGPU()` с использованием `std::filesystem::path` |

### Шаги

1. **Переписать `GetLogFilePathForGPU()`** — использовать `std::filesystem::path` для конкатенации
2. **Проверить `CreateLogDirectoryForGPU()`** — она уже использует `std::filesystem`, должна работать
3. **Сборка и тестирование** — убедиться что логи создаются в правильной структуре
4. **Удалить старые некорректные файлы** — `GPUWorkLib92Logs92*.log` в корне

---

## Проверка

### До исправления:
```
e:\C++\GPUWorkLib\GPUWorkLib92Logs92DRVGPU_00922026-02-099221-17-52.log
```

### После исправления:
```
e:\C++\GPUWorkLib\Logs\DRVGPU_00\2026-02-09\21-17-52.log
```

### Тест:
1. Собрать проект
2. Запустить тест (любой, использующий Logger)
3. Проверить что:
   - Создана директория `Logs/DRVGPU_00/YYYY-MM-DD/`
   - Файл лога внутри директории, не в корне
   - Содержимое лога корректно

---

## Связанные файлы

- `DrvGPU/logger/config_logger.hpp` — объявление функций
- `DrvGPU/logger/default_logger.cpp` — использует `GetLogFilePathForGPU()` (строка 74)
- `DrvGPU/services/service_manager.hpp` — вызывает `CreateLogDirectoryForGPU()` (строка 130)
- `Doc/PLAN/DETAILED_EXECUTION_PLAN.md` — секция 3.1 Logger

---

## Ссылки на план

Из `DETAILED_EXECUTION_PLAN.md`, секция 3.1:

```
БЫЛО:  ${path}/Logs/DRVGPU/YYYY-MM-DD/HH-MM-SS.log
СТАЛО: ${path}/Logs/DRVGPU_00/YYYY-MM-DD/HH-MM-SS.log
       ${path}/Logs/DRVGPU_01/YYYY-MM-DD/HH-MM-SS.log
       ${path}/Logs/DRVGPU_13/YYYY-MM-DD/HH-MM-SS.log
```

---

*Автор: Кодо (AI Assistant)*
*Проект: LCH-Farrow1 / GPUWorkLib*
*Для: Алекс*
