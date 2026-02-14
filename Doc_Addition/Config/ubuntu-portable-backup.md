# 📦 PORTABLE BACKUP НА UBUNTU - ПОЛНОЕ РУКОВОДСТВО

**Цель:** Восстановить все настройки DrvGPU на Ubuntu машине за 2 минуты! 🚀

---

## 🎯 ТРЕБОВАНИЯ

```bash
# На Ubuntu должно быть установлено:
- git (обычно уже есть)
- unzip (для распаковки архива)
- cmake (для сборки)
- code (VSCode)

# Проверить:
which unzip
which code
which cmake
```

---

## 📥 ПОДГОТОВКА АРХИВА НА WINDOWS

### На машине 1 (Windows):

```powershell
# Запустить скрипт backup
cd E:\C++\GPUWorkLib
powershell -ExecutionPolicy Bypass -File backup_settings.ps1

# Результат: DrvGPU-Settings-2026-02-01-2000.zip ✅

# Скопировать архив на флешку или в облако
# (Google Drive, OneDrive, Dropbox, и т.д.)
```

---

## 🚀 ВОССТАНОВЛЕНИЕ НА UBUNTU

### Способ 1: С ПОМОЩЬЮ СКРИПТА (РЕКОМЕНДУЕТСЯ)

#### Шаг 1: Скопировать архив на Ubuntu машину

```bash
# Через флешку, облако или git
# Поместить архив в папку проекта:
~/GPUWorkLib/DrvGPU-Settings-2026-02-01-2000.zip
```

#### Шаг 2: Скопировать скрипт восстановления

```bash
# Либо скачать скрипт, либо создать вручную:
# restore_settings.sh (см. ниже)

# Сделать исполняемым
chmod +x restore_settings.sh
```

#### Шаг 3: Запустить восстановление

```bash
# Перейти в папку проекта
cd ~/GPUWorkLib

# Запустить скрипт
./restore_settings.sh

# Вывод:
# 🔄 Восстановление DrvGPU настроек на Ubuntu...
# 📦 Найден архив: DrvGPU-Settings-2026-02-01-2000.zip
# ⏳ Распаковываю...
# ✅ Успешно восстановлено!
# 🚀 Готово! Открой проект в VSCode: code .
```

---

### Способ 2: ВРУЧНУЮ (БЫСТРО)

```bash
# 1. Перейти в папку проекта
cd ~/GPUWorkLib

# 2. Распаковать архив
unzip DrvGPU-Settings-*.zip

# 3. Проверить
ls -la
# .vscode ✓
# cmake ✓
# CMakeLists.txt ✓

# 4. Готово!
code .
```

---

## 📋 СКРИПТ ВОССТАНОВЛЕНИЯ ДЛЯ UBUNTU

### Создай файл: `restore_settings.sh`

```bash
#!/bin/bash
# Восстановление настроек на Ubuntu машине
# Запуск: bash restore_settings.sh

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "\n${YELLOW}🔄 Восстановление DrvGPU настроек на Ubuntu...${NC}\n"

# Найти архив
BACKUP_FILE=$(ls DrvGPU-Settings-*.zip 2>/dev/null | sort -r | head -1)

if [ -z "$BACKUP_FILE" ]; then
    echo -e "${RED}❌ Backup файл DrvGPU-Settings-*.zip не найден!${NC}"
    echo -e "${YELLOW}Поместите архив в текущую папку проекта и повторите.${NC}"
    exit 1
fi

echo -e "${GREEN}📦 Найден архив: $BACKUP_FILE${NC}"
echo -e "${YELLOW}⏳ Распаковываю...${NC}\n"

# Проверить наличие unzip
if ! command -v unzip &> /dev/null; then
    echo -e "${RED}❌ unzip не установлен!${NC}"
    echo -e "${YELLOW}Установи: sudo apt install unzip${NC}"
    exit 1
fi

# Распаковать
if unzip -o "$BACKUP_FILE" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Успешно восстановлено!${NC}"
    echo -e "${GREEN}   Файлы:${NC}"
    echo -e "${GREEN}   ✓ .vscode/ (настройки VSCode)${NC}"
    echo -e "${GREEN}   ✓ cmake/ (модули CMake)${NC}"
    echo -e "${GREEN}   ✓ CMakeLists.txt (все уровни)${NC}"
    echo -e "\n${GREEN}🚀 Готово! Открой проект в VSCode:${NC}"
    echo -e "${YELLOW}   code .${NC}\n"
else
    echo -e "${RED}❌ Ошибка при распаковке архива${NC}"
    exit 1
fi
```

---

## 🔧 УСТАНОВКА ЗАВИСИМОСТЕЙ НА UBUNTU

### Если что-то не установлено:

```bash
# Обновить пакеты
sudo apt update

# Установить unzip (если нужно)
sudo apt install unzip

# Установить cmake (если нужно)
sudo apt install cmake

# Установить VSCode (если нужно)
# Через официальный репозиторий:
sudo snap install --classic code

# ИЛИ через пакет:
wget https://code.visualstudio.com/sha/download?build=stable&os=linux-deb-x64
sudo dpkg -i code_*.deb

# Проверить установку
cmake --version
code --version
unzip -v
```

---

## 📁 СТРУКТУРА ПРОЕКТА НА UBUNTU

### После восстановления:

```
~/GPUWorkLib/
├── .vscode/
│   ├── settings.json          # Все настройки VSCode
│   ├── launch.json            # Отладка конфиг
│   ├── tasks.json             # Build tasks
│   └── extensions.json        # Список расширений
├── cmake/
│   ├── modules/
│   ├── FindSPDLOG.cmake
│   └── config.cmake
├── CMakeLists.txt             # Корневой
├── include/DrvGPU/
│   ├── CMakeLists.txt
│   ├── backends/
│   │   ├── opencl/CMakeLists.txt
│   │   └── rocm/CMakeLists.txt
│   ├── memory/CMakeLists.txt
│   └── common/CMakeLists.txt
├── tests/CMakeLists.txt
├── build/
└── restore_settings.sh        # Скрипт для следующего раза
```

---

## 🛠️ СБОРКА ПРОЕКТА НА UBUNTU

### После восстановления настроек:

```bash
# 1. Перейти в папку проекта
cd ~/GPUWorkLib

# 2. Создать build папку
mkdir -p build
cd build

# 3. Запустить cmake
cmake ..

# 4. Собрать проект
make -j$(nproc)

# ИЛИ через cmake:
cmake --build . --config Release

# 5. Проверить
./bin/drvgpu_example  # Если примеры есть

# 6. Готово! ✅
```

---

## 🐛 ОТЛАДКА UBUNTU СПЕЦИФИЧНЫХ ПРОБЛЕМ

### Проблема 1: unzip не установлен

```bash
# Решение
sudo apt install unzip

# Проверить
unzip -v
```

---

### Проблема 2: Нет прав на выполнение скрипта

```bash
# Решение
chmod +x restore_settings.sh
./restore_settings.sh
```

---

### Проблема 3: CMake не находит OpenCL

```bash
# На Ubuntu OpenCL нужно установить:
sudo apt install ocl-icd-opencl-dev opencl-headers

# Для NVIDIA GPU:
sudo apt install nvidia-opencl-icd

# Для AMD GPU:
sudo apt install rocm-opencl
```

---

### Проблема 4: Кодировка путей (Unicode)

```bash
# На Ubuntu кодировка обычно не проблема, но если есть:
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

# Затем распаковать архив
./restore_settings.sh
```

---

## 🔄 СИНХРОНИЗАЦИЯ МЕЖДУ МАШИНАМИ

### Если часто переключаешься:

```bash
# На Windows (машина 1)
powershell -ExecutionPolicy Bypass -File backup_settings.ps1

# На Ubuntu (машина 2)
./restore_settings.sh

# Всё синхронизировано! ✅
```

### Через Google Drive / Dropbox:

```bash
# На обеих машинах установить Google Drive / Dropbox
# Например, в ~/GoogleDrive/

# На Windows:
powershell -ExecutionPolicy Bypass -File backup_settings.ps1
# Скопировать архив в ~/GoogleDrive/DrvGPU-Settings/

# На Ubuntu:
cp ~/GoogleDrive/DrvGPU-Settings/*.zip ./
./restore_settings.sh

# Синхронизация в облаке! ☁️
```

---

## 🎯 БЫСТРЫЕ КОМАНДЫ ДЛЯ UBUNTU

```bash
# Всё в одной строке:
cd ~/GPUWorkLib && \
unzip -o DrvGPU-Settings-*.zip && \
mkdir -p build && \
cd build && \
cmake .. && \
make -j$(nproc) && \
echo "✅ Готово!"
```

---

## 📊 ПРОЦЕСС НА UBUNTU

| Шаг | Команда | Время |
|-----|---------|-------|
| 1 | `cd ~/GPUWorkLib` | 1 сек |
| 2 | `./restore_settings.sh` | 30 сек |
| 3 | `mkdir -p build && cd build` | 1 сек |
| 4 | `cmake ..` | 5 сек |
| 5 | `make -j$(nproc)` | 30 сек |
| **ИТОГО** | От архива до работы | **67 сек** ⚡ |

---

## 🚀 ПОЛНЫЙ WORKFLOW

### Машина 1 (Windows):

```powershell
# Один раз в неделю
cd E:\C++\GPUWorkLib
powershell -ExecutionPolicy Bypass -File backup_settings.ps1

# Скопировать DrvGPU-Settings-*.zip в облако
```

---

### Машина 2 (Ubuntu):

```bash
# При каждом переходе
cd ~/GoogleDrive/DrvGPU-Settings
cp DrvGPU-Settings-*.zip ~/GPUWorkLib/
cd ~/GPUWorkLib
./restore_settings.sh
code .

# Готово! Все настройки как на Windows! 🎉
```

---

## ✅ ПРОВЕРКА

```bash
# На Ubuntu проверить что восстановилось:

# 1. VSCode настройки
ls -la .vscode/
# settings.json, launch.json, tasks.json ✓

# 2. CMake модули
ls -la cmake/
# modules/, *.cmake ✓

# 3. CMakeLists.txt
find . -name "CMakeLists.txt" | head -5
# CMakeLists.txt ✓
# include/DrvGPU/CMakeLists.txt ✓

# Всё на месте! ✅
```

---

## 💾 ХРАНЕНИЕ АРХИВОВ

### На Ubuntu:

```bash
# Создать папку для архивов
mkdir -p ~/DrvGPU-Backups

# Сохранять туда архивы
cp ~/GoogleDrive/DrvGPU-Settings/*.zip ~/DrvGPU-Backups/

# Список версий
ls -lh ~/DrvGPU-Backups/
# DrvGPU-Settings-2026-02-01.zip
# DrvGPU-Settings-2026-02-02.zip
# DrvGPU-Settings-latest.zip
```

---

## 🎁 SHELL ALIASES (ОПЦИОНАЛЬНО)

### Добавить в ~/.bashrc:

```bash
# Backup (если есть Windows машина рядом)
alias backup-drvgpu='cd ~/GPUWorkLib && echo "Запусти на Windows: backup_settings.ps1"'

# Restore
alias restore-drvgpu='cd ~/GPUWorkLib && ./restore_settings.sh'

# Build
alias build-drvgpu='cd ~/GPUWorkLib && mkdir -p build && cd build && cmake .. && make -j$(nproc)'

# Quick setup
alias setup-drvgpu='restore-drvgpu && build-drvgpu'
```

**Затем использовать:**

```bash
source ~/.bashrc
setup-drvgpu  # Всё восстановится и соберётся автоматически! 🚀
```

---

## 📝 ИТОГОВАЯ ШПАРГАЛКА ДЛЯ UBUNTU

```bash
# 1. Скопировать архив (с Windows машины или облака)
cd ~/GPUWorkLib
cp ~/GoogleDrive/DrvGPU-Settings/*.zip ./

# 2. Восстановить (один клик!)
./restore_settings.sh

# 3. Собрать (если нужно)
mkdir -p build && cd build && cmake .. && make -j$(nproc)

# 4. Готово! 🎉
code .
```

---

## ✨ ФИНАЛЬНАЯ РЕКОМЕНДАЦИЯ

**Windows машина:**
```powershell
# Каждую неделю
backup_settings.ps1
```

**Ubuntu машина:**
```bash
# Каждый раз когда работаешь
./restore_settings.sh
```

**Результат:** Идентичные настройки на обеих машинах! 💪

---

**Файлы для Ubuntu:**
- `restore_settings.sh` - Скрипт восстановления
- Этот гайд - полное руководство

**Готово к использованию на Linux!** 🐧🚀
