# 📦 PORTABLE BACKUP - ПЕРЕНОС НАСТРОЕК НА ДРУГУЮ МАШИНУ

**Цель:** Один раз - настроил на одной машине, скопировал, и всё работает на другой! 🚀

---

## 🎯 ЧТО АРХИВИРОВАТЬ?

```
├── .vscode/                    # VSCode settings + extensions
├── cmake/                      # CMake modules
├── CMakeLists.txt              # Корневой CMakeLists
├── include/DrvGPU/
│   ├── CMakeLists.txt          # Основной CMakeLists
│   ├── backends/
│   │   ├── opencl/CMakeLists.txt
│   │   └── rocm/CMakeLists.txt
│   ├── memory/CMakeLists.txt
│   └── common/CMakeLists.txt
└── tests/CMakeLists.txt        # Тесты CMakeLists
```

---

## 🚀 СПОСОБ 1: АВТОМАТИЧЕСКИЙ СКРИПТ (ЛУЧШИЙ)

### Создай файл: `backup_settings.ps1`

```powershell
# ════════════════════════════════════════════════════════════════════
# backup_settings.ps1 - Создание portable backup всех настроек
# ════════════════════════════════════════════════════════════════════

# Цвета для вывода
$Green = @{ ForegroundColor = 'Green' }
$Red = @{ ForegroundColor = 'Red' }
$Yellow = @{ ForegroundColor = 'Yellow' }

Write-Host "🔄 Создание portable backup..." @Yellow

# Определить корневую папку проекта
$ProjectRoot = Get-Location
$BackupName = "DrvGPU-Settings-$(Get-Date -Format 'yyyy-MM-dd-HHmm').zip"
$BackupPath = Join-Path $ProjectRoot $BackupName

# Папки для архивирования
$FoldersToBackup = @(
    '.vscode',
    'cmake'
)

# Файлы для архивирования (CMakeLists.txt)
$FilesToBackup = @(
    'CMakeLists.txt',
    'include/DrvGPU/CMakeLists.txt',
    'include/DrvGPU/backends/opencl/CMakeLists.txt',
    'include/DrvGPU/backends/rocm/CMakeLists.txt',
    'include/DrvGPU/memory/CMakeLists.txt',
    'include/DrvGPU/common/CMakeLists.txt',
    'tests/CMakeLists.txt'
)

# ════════════════════════════════════════════════════════════════════
# Проверка существования файлов/папок
# ════════════════════════════════════════════════════════════════════

Write-Host "`n✓ Проверка файлов:" @Green

$ValidItems = @()

foreach ($Folder in $FoldersToBackup) {
    $FolderPath = Join-Path $ProjectRoot $Folder
    if (Test-Path $FolderPath) {
        Write-Host "  ✓ $Folder" @Green
        $ValidItems += $FolderPath
    } else {
        Write-Host "  ✗ $Folder (не найдена)" @Red
    }
}

foreach ($File in $FilesToBackup) {
    $FilePath = Join-Path $ProjectRoot $File
    if (Test-Path $FilePath) {
        Write-Host "  ✓ $File" @Green
        $ValidItems += $FilePath
    } else {
        Write-Host "  ✗ $File (не найден)" @Red
    }
}

# ════════════════════════════════════════════════════════════════════
# Создание архива
# ════════════════════════════════════════════════════════════════════

if ($ValidItems.Count -eq 0) {
    Write-Host "`n❌ Нечего архивировать!" @Red
    exit 1
}

Write-Host "`n📦 Создание архива: $BackupName" @Yellow

try {
    Compress-Archive -Path $ValidItems -DestinationPath $BackupPath -Force
    Write-Host "✓ Архив создан успешно!" @Green
    Write-Host "  Размер: $(([System.IO.FileInfo]$BackupPath).Length / 1MB | [math]::Round(2)) MB"
    Write-Host "  Путь: $BackupPath" @Green
} catch {
    Write-Host "❌ Ошибка при создании архива: $_" @Red
    exit 1
}

# ════════════════════════════════════════════════════════════════════
# Информация для восстановления
# ════════════════════════════════════════════════════════════════════

Write-Host "`n📋 Информация для восстановления:" @Yellow
Write-Host @'
1. Скопируй файл на другую машину:
   - Папка проекта (e.g. E:\C++\GPUWorkLib\)

2. Распакуй архив (в командной строке):
   PowerShell -Command "Expand-Archive -Path <имя-архива> -DestinationPath ."

3. После распаковки:
   - .vscode/ → переписаны настройки VSCode
   - cmake/ → переписаны CMake модули
   - CMakeLists.txt → переписаны все CMakeLists

4. Готово! 🚀
   - Открой проект в VSCode
   - Все настройки уже там

'@

Write-Host "✓ Скрипт завершён!" @Green
```

---

### Запуск скрипта:

```bash
# В PowerShell в папке проекта
cd E:\C++\GPUWorkLib
powershell -ExecutionPolicy Bypass -File backup_settings.ps1

# Результат:
# DrvGPU-Settings-2026-02-01-2000.zip ✅
```

---

## 🎮 СПОСОБ 2: КОМАНДНАЯ СТРОКА

### Один-два клика!

```bash
# Открой PowerShell в папке проекта
cd E:\C++\GPUWorkLib

# Создать архив
$Items = @('.vscode', 'cmake', 'CMakeLists.txt', 'include/DrvGPU/CMakeLists.txt')
Compress-Archive -Path $Items -DestinationPath "DrvGPU-Settings-backup.zip" -Force

# Результат: DrvGPU-Settings-backup.zip ✅
```

---

## 📤 СПОСОБ 3: СТРУКТУРИРОВАННЫЙ АРХИВ

### С правильной структурой папок

**Создай файл: `create_backup.bat`**

```batch
@echo off
REM ════════════════════════════════════════════════════════════════════
REM Создание backup с структурой
REM ════════════════════════════════════════════════════════════════════

setlocal enabledelayedexpansion
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c%%a%%b)
for /f "tokens=1-2 delims=/:" %%a in ('time /t') do (set mytime=%%a%%b)

set BACKUP_NAME=DrvGPU-Settings-%mydate%-%mytime%.zip
set BACKUP_DIR=Backups
set TEMP_DIR=temp_backup

echo Creating backup structure...

REM Создать временную папку
if exist %TEMP_DIR% rmdir /s /q %TEMP_DIR%
mkdir %TEMP_DIR%

REM Копировать файлы
echo Copying .vscode...
xcopy /s /e /i /y .vscode %TEMP_DIR%\.vscode > nul

echo Copying cmake...
xcopy /s /e /i /y cmake %TEMP_DIR%\cmake > nul

echo Copying CMakeLists.txt files...
copy CMakeLists.txt %TEMP_DIR%\ > nul
mkdir %TEMP_DIR%\include\DrvGPU\backends\opencl %TEMP_DIR%\include\DrvGPU\backends\rocm %TEMP_DIR%\include\DrvGPU\memory %TEMP_DIR%\include\DrvGPU\common %TEMP_DIR%\tests > nul

copy include\DrvGPU\CMakeLists.txt %TEMP_DIR%\include\DrvGPU\ > nul
copy include\DrvGPU\backends\opencl\CMakeLists.txt %TEMP_DIR%\include\DrvGPU\backends\opencl\ > nul
copy include\DrvGPU\backends\rocm\CMakeLists.txt %TEMP_DIR%\include\DrvGPU\backends\rocm\ > nul
copy include\DrvGPU\memory\CMakeLists.txt %TEMP_DIR%\include\DrvGPU\memory\ > nul
copy include\DrvGPU\common\CMakeLists.txt %TEMP_DIR%\include\DrvGPU\common\ > nul
copy tests\CMakeLists.txt %TEMP_DIR%\tests\ > nul

REM Создать backup папку если нужна
if not exist %BACKUP_DIR% mkdir %BACKUP_DIR%

REM Создать zip
cd %TEMP_DIR%
tar -a -c -f ..\%BACKUP_DIR%\%BACKUP_NAME% *
cd ..

REM Очистить временную папку
rmdir /s /q %TEMP_DIR%

echo.
echo ✓ Backup создан: %BACKUP_DIR%\%BACKUP_NAME%
echo.
pause
```

**Запуск:**
```bash
create_backup.bat
```

---

## 📥 ВОССТАНОВЛЕНИЕ НА ДРУГОЙ МАШИНЕ

### На новой машине:

```bash
# 1. Скопировал архив в папку проекта
cd E:\C++\GPUWorkLib

# 2. Распаковать архив (PowerShell)
Expand-Archive -Path "DrvGPU-Settings-backup.zip" -DestinationPath "." -Force

# 3. Проверить
ls -la
# .vscode/  ✓
# cmake/    ✓
# CMakeLists.txt ✓
# include/DrvGPU/CMakeLists.txt ✓

# 4. Готово! Открыть в VSCode
code .
```

---

## 📋 ЧТО ВКЛЮЧАЕТ АРХИВ

### После распаковки всё на месте:

```
E:\C++\GPUWorkLib\
├── .vscode/
│   ├── settings.json          # Все VSCode настройки
│   ├── launch.json            # Отладка конфиг
│   ├── tasks.json             # Build tasks
│   └── extensions.json        # Список расширений
│
├── cmake/
│   ├── modules/
│   ├── FindSPDLOG.cmake       # Все модули
│   └── config.cmake
│
├── CMakeLists.txt             # Корневой
├── include/DrvGPU/
│   ├── CMakeLists.txt
│   ├── backends/
│   │   ├── opencl/CMakeLists.txt
│   │   └── rocm/CMakeLists.txt
│   ├── memory/CMakeLists.txt
│   └── common/CMakeLists.txt
└── tests/CMakeLists.txt

# Готово к build! 🚀
```

---

## 🔄 АВТОМАТИЗАЦИЯ: Периодический Backup

### Добавить в Task Scheduler (Windows)

**Создай файл: `auto_backup.ps1`**

```powershell
# Запускается каждый день в 6 PM
$ScriptPath = "C:\Scripts\backup_settings.ps1"
$ProjectPath = "E:\C++\GPUWorkLib"

cd $ProjectPath
& $ScriptPath
```

**Добавить в Task Scheduler:**
```powershell
$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File C:\Scripts\auto_backup.ps1"
$Trigger = New-ScheduledTaskTrigger -Daily -At 6PM
$Task = New-ScheduledTask -Action $Action -Trigger $Trigger -Description "DrvGPU Settings Backup"
Register-ScheduledTask -TaskName "DrvGPU-Backup" -InputObject $Task
```

---

## 🎯 РЕКОМЕНДУЕМАЯ СТРУКТУРА BACKUP

### Для удобства:

```
Backups/
├── DrvGPU-Settings-2026-02-01.zip     # Backup от 1 февраля
├── DrvGPU-Settings-2026-02-02.zip     # Backup от 2 февраля
├── DrvGPU-Settings-latest.zip         # Последний backup
└── README.md
    │
    └─ Инструкции по восстановлению
```

---

## ✅ ИТОГОВЫЙ ПРОЦЕСС

### На машине 1 (Дома):

```bash
# 1. Один раз настроил всё
# 2. Запустил скрипт
powershell -ExecutionPolicy Bypass -File backup_settings.ps1

# 3. Результат: DrvGPU-Settings-2026-02-01-2000.zip
```

---

### На машине 2 (На работе):

```bash
# 1. Скопировал архив в папку проекта
# 2. Один клик
Expand-Archive -Path "DrvGPU-Settings-*.zip" -DestinationPath "." -Force

# 3. Открыл VSCode
code .

# 4. Готово! Всё как дома! 🎉
```

---

## 💡 БОНУС: Git-friendly backup

### Добавить в .gitignore:

```gitignore
# Исключить большие backup файлы
Backups/
*.zip
```

**Но добавить в Arc (для распространения):**

```bash
# Копировать latest backup в Arc
cp DrvGPU-Settings-latest.zip Arc/

# Это было 4-й архив! 🎯
git add Arc/
git commit -m "Update settings backup archive"
```

---

## 🚀 БЫСТРАЯ УСТАНОВКА НА ДРУГОЙ МАШИНЕ

### Один скрипт для восстановления:

**restore_settings.ps1:**

```powershell
# Распаковать, скопировать в нужные места, всё готово!

$BackupFile = Get-ChildItem "DrvGPU-Settings-*.zip" | Sort-Object LastWriteTime -Descending | Select-Object -First 1

if (-not $BackupFile) {
    Write-Host "❌ Backup файл не найден!" -ForegroundColor Red
    exit 1
}

Write-Host "📦 Восстанавливаю из: $($BackupFile.Name)" -ForegroundColor Yellow

Expand-Archive -Path $BackupFile.FullName -DestinationPath "." -Force

Write-Host "✓ Готово! Все настройки восстановлены! 🎉" -ForegroundColor Green
```

---

## 📊 ИТОГ

| Машина | Действие | Время |
|--------|----------|-------|
| **Дома** | `backup_settings.ps1` | 30 сек |
| **Дома** | Скопировать архив на флешку | 1 мин |
| **На работе** | `restore_settings.ps1` | 30 сек |
| **На работе** | Открыть VSCode | Instant |
| **ИТОГО** | От настройки до работы | 2 мин вместо 2 часов! 🚀 |

---

## ✨ ФИНАЛЬНАЯ РЕКОМЕНДАЦИЯ

**Лучший способ:**

```bash
# 1. Один раз на машине 1
powershell -ExecutionPolicy Bypass -File backup_settings.ps1

# 2. Скопировать архив
# DrvGPU-Settings-2026-02-01-2000.zip

# 3. На машине 2
Expand-Archive -Path "DrvGPU-Settings-*.zip" -DestinationPath "." -Force

# 4. Готово! 🎉

# Так можно делать каждый раз когда обновляешь настройки!
```

---

**Файлы для скачивания:**
- `backup_settings.ps1` - Автоматический backup
- `create_backup.bat` - Batch версия
- `restore_settings.ps1` - Восстановление

**Все готовы к использованию!** 🚀
