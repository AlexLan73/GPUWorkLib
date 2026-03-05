# Debian + AMD Radeon 9070 — настройка окружения

> **Цель**: Сборка и пошаговая отладка GPUWorkLib на Debian с AMD Radeon 9070 (gfx1201).
> **Дата**: 2026-02-24

---

## 1. Системные библиотеки (apt)

### 1.1 Базовые инструменты сборки

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build gdb
```

### 1.2 OpenCL (для OpenCL backend)

```bash
sudo apt install -y ocl-icd-opencl-dev opencl-headers
```

### 1.3 ROCm 7.2 (обязательно для Radeon 9070)

Radeon 9070 (gfx1201, RDNA 4) поддерживается только в **ROCm 7.0.2+**. ROCm 5.7 не подходит.

**Способ: офлайн-установщик AMD**

1. Скачать ROCm 7.2 для Debian 13:
   - https://repo.radeon.com/rocm/installer/rocm-linux-install-offline/rocm-rel-7.2/debian/13/

2. Установить по инструкции AMD (см. `README` в архиве или [ROCm Install Guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)).

3. После установки ROCm обычно ставится в `/opt/rocm`.

**Альтернатива: Debian ROCm Team (apt.rocm.debian.net)**

```bash
sudo wget -O /usr/share/keyrings/rocm-archive-keyring.gpg \
  https://apt.rocm.debian.net/debian/rocm-archive-keyring.gpg

# Для Debian 13 (trixie):
echo "deb [signed-by=/usr/share/keyrings/rocm-archive-keyring.gpg] https://apt.rocm.debian.net/debian trixie main" | \
  sudo tee /etc/apt/sources.list.d/rocm.list

# Для Debian 12 (bookworm) заменить trixie на bookworm
sudo apt update
sudo apt install -y rocm rocm-hip-sdk rocm-opencl-runtime
```

### 1.4 clFFT (OpenCL FFT)

clFFT может быть в составе ROCm или собираться отдельно. Проверка:

```bash
# Проверить наличие
ls /opt/rocm/lib/libclFFT* 2>/dev/null || ls /usr/lib/*/libclFFT* 2>/dev/null
```

Если не найден — см. [ROCm_Setup_Instructions.md](../ROCm_Setup_Instructions.md) или сборка из исходников.

### 1.5 Дополнительно (опционально)

```bash
# spdlog (если не через FetchContent)
sudo apt install -y libspdlog-dev

# nlohmann-json (проект использует third_party/nlohmann/json.hpp — не обязательно)
# sudo apt install -y nlohmann-json3-dev
```

---

## 2. Проверка окружения

```bash
# ROCm
rocminfo
hipinfo

# OpenCL
clinfo
```

---

## 3. Сборка проекта

```bash
cd /home/alex/C++/GPUWorkLib

# Конфигурация (Release)
cmake --preset Debian-Radeon9070

# Сборка Release
cmake --build build/debian-radeon9070 -j8

# Конфигурация (Debug)
cmake --preset Debian-Radeon9070-Debug

# Сборка Debug
cmake --build build/debian-radeon9070-debug -j4

# Запуск
./build/debian-radeon9070-debug/GPUWorkLib
```

---

## 4. VSCode — расширения для отладки

Установить в VSCode/Cursor:

| Расширение | ID | Назначение |
|------------|-----|------------|
| **C/C++** | `ms-vscode.cpptools` | IntelliSense, отладка (GDB) |
| **CMake Tools** | `ms-vscode.cmake-tools` | CMake presets, конфигурация |
| **CMake** | `ms-vscode.cmake-tools` | Синтаксис CMake |

### Установка через командную строку

```bash
code --install-extension ms-vscode.cpptools
code --install-extension ms-vscode.cmake-tools
```

Или в Cursor: `Ctrl+Shift+X` → поиск "C/C++", "CMake Tools" → Install.

---

## 5. Отладка по шагам в VSCode

### 5.1 Первый запуск

1. Открыть проект в VSCode/Cursor.
2. Выбрать preset: **CMake: Select Configure Preset** → `Debian-Radeon9070-Debug`.
3. Собрать: **Terminal → Run Build Task** (Ctrl+Shift+B) или задача `CMake: Configure + Build Debug`.
4. Поставить breakpoint в `src/main.cpp` или другом файле (клик слева от номера строки).
5. Нажать **F5** или **Run → Start Debugging**.

### 5.2 Конфигурация launch.json

Уже настроено в `.vscode/launch.json`:

- **GPUWorkLib (Debian Radeon 9070 Debug)** — запуск с отладчиком.
- `preLaunchTask` — автоматическая сборка перед запуском.
- `cwd` — каталог с `configGPU.json` (копируется при сборке).

### 5.3 Горячие клавиши отладки

| Клавиша | Действие |
|---------|----------|
| F5 | Запуск / продолжить |
| F10 | Step Over |
| F11 | Step Into |
| Shift+F11 | Step Out |
| F9 | Toggle Breakpoint |

---

## 6. Переменные окружения

При запуске из терминала (если не через VSCode):

```bash
export ROCM_HOME=/opt/rocm
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
export PATH=/opt/rocm/bin:$PATH
./build/debian-radeon9070-debug/GPUWorkLib
```

---

## 7. Чек-лист

- [ ] `build-essential`, `cmake`, `ninja-build`, `gdb`
- [ ] `ocl-icd-opencl-dev`, `opencl-headers`
- [ ] ROCm 7.2 (офлайн или apt.rocm.debian.net)
- [ ] `rocminfo`, `hipinfo` — выводят информацию о GPU
- [ ] `clinfo` — видит AMD GPU
- [ ] VSCode: C/C++, CMake Tools
- [ ] CMake preset: `Debian-Radeon9070-Debug`
- [ ] F5 — отладка работает

---

*Создано: 2026-02-24 | Кодо*
