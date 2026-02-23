# ROCm Setup — Инструкции для добавления (2026-02-24)

> **Цель**: Добавить поддержку ROCm + OpenCL на Linux (Debian 13, Ubuntu).
> **Оборудование**: Radeon 9070 (тесты), AMD Instinct MI100 (работа — ROCm).
> **Пути**: Адаптировать под локальные пути на рабочей машине.

---

## 1. ОБЯЗАТЕЛЬНО ДОБАВИТЬ

### 1.1 Файл `CMakePresets.json` (в корне проекта)

Если файла нет — создать. Если есть — добавить/обновить configurePresets:

```json
{
  "version": 3,
  "cmakeMinimumRequired": { "major": 3, "minor": 20, "patch": 0 },
  "configurePresets": [
    {
      "name": "base",
      "hidden": true,
      "generator": "Ninja",
      "binaryDir": "${sourceDir}/build",
      "cacheVariables": {
        "CMAKE_CXX_STANDARD": "17"
      }
    },
    {
      "name": "Ubuntu",
      "displayName": "Ubuntu",
      "inherits": "base",
      "condition": { "type": "equals", "lhs": "${hostSystemName}", "rhs": "Linux" },
      "cacheVariables": {
        "ENABLE_OPENCL": "ON",
        "ENABLE_ROCM": "ON",
        "ROCM_VERSION": "7.5"
      }
    },
    {
      "name": "Debian",
      "displayName": "Debian",
      "inherits": "base",
      "condition": { "type": "equals", "lhs": "${hostSystemName}", "rhs": "Linux" },
      "cacheVariables": {
        "ENABLE_OPENCL": "ON",
        "ENABLE_ROCM": "ON",
        "ROCM_VERSION": "5.7"
      }
    },
    {
      "name": "Debian-Radeon9070",
      "displayName": "Debian + Radeon 9070 (ROCm 7.2)",
      "inherits": "base",
      "condition": { "type": "equals", "lhs": "${hostSystemName}", "rhs": "Linux" },
      "cacheVariables": {
        "ENABLE_OPENCL": "ON",
        "ENABLE_ROCM": "ON",
        "ROCM_VERSION": "7.2"
      }
    }
  ]
}
```

**Использование**:
```bash
cmake --preset Ubuntu            # Ubuntu 22.04/24.04 + ROCm 7.5
cmake --preset Debian            # Debian 13 + ROCm 5.7 (MI100)
cmake --preset Debian-Radeon9070 # Debian 13 + ROCm 7.2 (Radeon 9070)
```

---

### 1.2 Изменения в `cmake/gpu-config.cmake`

**Добавить** (после блока ENABLE_OPENCL):

```cmake
# ============================================================================
# ROCm SUPPORT (Linux only)
# ============================================================================
if(NOT DEFINED ENABLE_ROCM)
  if(IS_LINUX)
    set(ENABLE_ROCM ON CACHE BOOL "Enable ROCm (Linux)")
  else()
    set(ENABLE_ROCM OFF CACHE BOOL "ROCm not available on Windows")
  endif()
endif()

if(NOT DEFINED ROCM_VERSION)
  set(ROCM_VERSION "7.5" CACHE STRING "ROCm version (5.7 Debian, 7.5 Ubuntu)")
endif()

if(ENABLE_ROCM AND NOT IS_LINUX)
  message(WARNING "ROCm requested on non-Linux - disabling")
  set(ENABLE_ROCM OFF CACHE BOOL "ROCm disabled" FORCE)
endif()

message(STATUS "  ENABLE_ROCM: ${ENABLE_ROCM}")
if(ENABLE_ROCM)
  message(STATUS "  ROCM_VERSION: ${ROCM_VERSION}")
endif()
```

---

### 1.3 Изменения в `cmake/dependencies.cmake`

**Добавить** (после блока OpenCL, перед clFFT):

```cmake
# ============================================================================
# ROCm/HIP (Linux only, when ENABLE_ROCM)
# ============================================================================
set(ROCM_ENABLED FALSE)

if(ENABLE_ROCM AND IS_LINUX)
  message(STATUS "🔍 Searching for ROCm/HIP...")
  find_package(hip QUIET)
  if(hip_FOUND)
    set(ROCM_ENABLED TRUE)
    message(STATUS "✅ ROCm/HIP found!")
  else()
    message(WARNING "❌ ROCm not found - ENABLE_ROCM=ON but hip not found")
    message(STATUS "   Debian: apt install rocm-hip-sdk")
    message(STATUS "   Ubuntu: apt install hip-dev")
  endif()
endif()
```

---

### 1.4 Признаки для Linux + Radeon 9070

Если нужны compile-time флаги (добавить в target_compile_definitions drvgpu):

```cmake
if(ENABLE_ROCM AND IS_LINUX)
  target_compile_definitions(drvgpu PRIVATE DRVGPU_ROCM_ENABLED=1)
  if(ROCM_VERSION VERSION_GREATER_EQUAL "7.0")
    target_compile_definitions(drvgpu PRIVATE DRVGPU_ROCM_GFX1201=1)  # Radeon 9070
  endif()
endif()
```

---

### 1.5 Каталог `cmake/` и Git

**Вариант B**: Оставить `.gitignore` как есть. На рабочей машине ИИ добавляет `cmake/` по этой инструкции (вручную или через `git add -f cmake/`).

**Содержимое каталога `cmake/`** (должно быть в репозитории):

| Файл | Описание |
|------|----------|
| `platform-detection.cmake` | Определение IS_WINDOWS, IS_LINUX, PLATFORM_NAME |
| `gpu-config.cmake` | ENABLE_OPENCL, ENABLE_ROCM, ROCM_VERSION (с изменениями из п. 1.2) |
| `dependencies.cmake` | Поиск OpenCL, ROCm/HIP, clFFT, nlohmann_json (с изменениями из п. 1.3) |

**Действия на рабочей машине**:
1. Убедиться, что каталог `cmake/` существует
2. Содержит три файла выше с нужными изменениями
3. При необходимости: `git add -f cmake/platform-detection.cmake cmake/gpu-config.cmake cmake/dependencies.cmake`

---

## 2. Установка ROCm на Debian 13

### 2.1 Radeon 9070 — ROCm 7.x (обязательно)

Radeon 9070 (gfx1201, RDNA 4) поддерживается только в **ROCm 7.0.2+**. ROCm 5.7 не подходит.

**Способ: офлайн-установщик AMD**

1. Скачать ROCm 7.2 для Debian 13:
   ```
   https://repo.radeon.com/rocm/installer/rocm-linux-install-offline/rocm-rel-7.2/debian/13/
   ```

2. Установить по инструкции AMD (см. `README` в архиве или [ROCm Install Guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)).

3. Использовать preset **Debian-Radeon9070**:
   ```bash
   cmake --preset Debian-Radeon9070
   cmake --build build
   ```

### 2.2 AMD Instinct MI100 — ROCm 5.7 или 7.x

MI100 (gfx908, CDNA 1) поддерживается в ROCm 5.7 и ROCm 7.x.

- **ROCm 5.7** — из стандартных репозиториев Debian (если доступно)
- **ROCm 7.x** — офлайн-установщик или `apt.rocm.debian.net`

Preset **Debian** (ROCM_VERSION=5.7) — для MI100 с ROCm 5.7.

### 2.3 Альтернатива: Debian ROCm Team (apt.rocm.debian.net)

Неофициальный репозиторий для Debian (bookworm/trixie):

```bash
sudo wget -O /usr/share/keyrings/rocm-archive-keyring.gpg \
  https://apt.rocm.debian.net/debian/rocm-archive-keyring.gpg

echo "deb [signed-by=/usr/share/keyrings/rocm-archive-keyring.gpg] https://apt.rocm.debian.net/debian bookworm main" | \
  sudo tee /etc/apt/sources.list.d/rocm.list

sudo apt update
sudo apt install rocm
```

Для Debian 13 (trixie) заменить `bookworm` на `trixie` (если пакеты доступны).

---

## 3. Порядок проверки

```bash
cmake --preset Ubuntu            # или Debian, Debian-Radeon9070
cmake --build build
```

---

## 4. Оборудование

| GPU | Архитектура | ROCm 5.7 | ROCm 7.x |
|-----|-------------|----------|----------|
| Radeon 9070 | RDNA 4 (gfx1201) | ❌ | ✅ (Ubuntu, Debian офлайн) |
| AMD Instinct MI100 | CDNA 1 (gfx908) | ✅ | ✅ |

---

*Создано: 2026-02-24*
