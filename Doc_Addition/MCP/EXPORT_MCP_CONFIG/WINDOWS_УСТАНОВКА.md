# 🪟 Установка MCP серверов на Windows

## 📦 Что нужно перед установкой

### 1. Node.js и npm
MCP серверы работают через `npx`, который идёт с Node.js.

**Проверка:**
```cmd
node --version
npm --version
```

**Если не установлен:**
- Скачай: https://nodejs.org/ (LTS версия)
- Или через winget: `winget install OpenJS.NodeJS.LTS`

### 2. Claude Code CLI
```cmd
claude --version
```

Если не работает - переустанови Claude Code или добавь в PATH.

---

## 🚀 Быстрая установка (3 способа)

### Способ 1: PowerShell (рекомендуется)

```powershell
# Открой PowerShell в папке проекта GPUWorkLib
cd C:\путь\к\GPUWorkLib\Doc\EXPORT_MCP_CONFIG

# Разрешить выполнение скриптов (однократно)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Запустить импорт
.\import_mcp_config.ps1
```

**Или с указанием пути:**
```powershell
.\import_mcp_config.ps1 C:\Projects\GPUWorkLib
```

### Способ 2: CMD (Командная строка)

```cmd
cd C:\путь\к\GPUWorkLib\Doc\EXPORT_MCP_CONFIG
import_mcp_config.bat
```

**Или:**
```cmd
import_mcp_config.bat C:\Projects\GPUWorkLib
```

### Способ 3: Ручная установка

Если скрипты не работают, установи по одному:

```cmd
rem 1. Sequential Thinking
claude mcp add sequential-thinking -- npx -y @modelcontextprotocol/server-sequential-thinking

rem 2. Context7
claude mcp add context7 -- npx -y @upstash/context7-mcp@latest

rem 3. Filesystem (замени путь!)
claude mcp add filesystem -- npx -y @modelcontextprotocol/server-filesystem C:\путь\к\GPUWorkLib

rem 4. Memory
claude mcp add memory -- npx -y @modelcontextprotocol/server-memory

rem 5. SQLite (замени путь!)
claude mcp add sqlite -- npx -y @modelcontextprotocol/server-sqlite --db-path C:\путь\к\GPUWorkLib\results.db

rem 6. Git (замени путь!)
claude mcp add git -- npx -y @modelcontextprotocol/server-git --repository C:\путь\к\GPUWorkLib

rem 7. Fetch
claude mcp add fetch -- npx -y @modelcontextprotocol/server-fetch
```

---

## 🗂️ Где хранятся настройки на Windows

### Конфигурация MCP серверов:
```
C:\Users\ИмяПользователя\.claude.json
```

### Структура внутри:
```json
{
  "projects": {
    "C:\\Projects\\GPUWorkLib": {
      "mcpServers": { ... }
    }
  }
}
```

---

## ✅ Проверка установки

```cmd
claude mcp list
```

**Ожидаемый результат:**
```
✓ sequential-thinking - Connected
✓ context7 - Connected
✓ filesystem - Connected
✓ memory - Connected
✓ sqlite - Connected (или ✗ до первого использования)
✓ git - Connected (или ✗ до первого использования)
✓ fetch - Connected (или ✗ до первого использования)
```

---

## 🔧 Дополнительные серверы

### GitHub MCP (опционально)

**Установка GitHub CLI:**
```powershell
# Через winget (Windows 11/10)
winget install --id GitHub.cli

# Или скачай: https://cli.github.com/
```

**Настройка:**
```powershell
gh auth login
# Выбери: GitHub.com -> SSH -> Your SSH key

# Получи токен
$env:GITHUB_TOKEN = gh auth token

# Добавь сервер
claude mcp add github -e GITHUB_PERSONAL_ACCESS_TOKEN=$env:GITHUB_TOKEN -- npx -y @modelcontextprotocol/server-github
```

### Brave Search (опционально)

1. Получи API ключ: https://brave.com/search/api/
2. Добавь сервер:

```cmd
claude mcp add brave-search -e BRAVE_API_KEY=твой_ключ -- npx -y @modelcontextprotocol/server-brave-search
```

---

## 🎯 Особенности Windows

### Пути с пробелами
Если путь содержит пробелы, используй кавычки:
```cmd
claude mcp add filesystem -- npx -y @modelcontextprotocol/server-filesystem "C:\Program Files\GPUWorkLib"
```

### Слэши в путях
Windows понимает оба варианта:
- `C:\Projects\GPUWorkLib` ✓
- `C:/Projects/GPUWorkLib` ✓

### PowerShell vs CMD
- **PowerShell** - современный, рекомендуется
- **CMD** - старая командная строка, тоже работает

### WSL (опционально)
Если используешь WSL (Windows Subsystem for Linux):
```bash
# В WSL используй Linux скрипт
cd /mnt/c/Projects/GPUWorkLib/Doc/EXPORT_MCP_CONFIG
./import_mcp_config.sh
```

---

## 🐛 Решение проблем

### "claude: команда не найдена"
Claude Code не в PATH. Найди где установлен и добавь в PATH:
```
C:\Users\ИмяПользователя\AppData\Local\Programs\Claude
```

### "npx: команда не найдена"
Node.js не установлен или не в PATH:
```powershell
# Установи Node.js
winget install OpenJS.NodeJS.LTS

# Перезапусти PowerShell
```

### "Выполнение скриптов отключено"
PowerShell блокирует скрипты:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Серверы не подключаются
Подожди 10-20 секунд - первый запуск загружает npm пакеты.

Проверь ещё раз:
```cmd
claude mcp list
```

### Ошибки с путями
Используй абсолютные пути с правильными слэшами:
```cmd
# Плохо
claude mcp add filesystem -- npx -y @modelcontextprotocol/server-filesystem .

# Хорошо
claude mcp add filesystem -- npx -y @modelcontextprotocol/server-filesystem C:\Projects\GPUWorkLib
```

---

## 📚 Где документация

После установки смотри:
```
C:\Projects\GPUWorkLib\Doc\MCP_SERVERS_SETUP.md
C:\Projects\GPUWorkLib\Doc\MCP_CHEATSHEET.md
C:\Projects\GPUWorkLib\Doc\MANUAL_INSTALL_GITHUB_BRAVE.md
```

---

## 💡 Быстрые команды Windows

### Открыть PowerShell в папке проекта
1. Shift + правая кнопка мыши на папке
2. "Open PowerShell window here"

### Или из проводника
1. В адресной строке проводника напиши: `powershell`
2. Enter

### CMD из проводника
В адресной строке: `cmd`

---

## 🎮 GPU на Windows

### NVIDIA (CUDA)
Убедись что установлен:
- CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
- cuDNN (если нужно)

Проверка:
```cmd
nvcc --version
nvidia-smi
```

### AMD (ROCm/HIP)
На Windows ROCm официально не поддерживается.

Альтернативы:
- **WSL2** с Ubuntu + ROCm
- **Docker** с ROCm образом
- **Dual boot** с Linux

---

## 🔄 Синхронизация между работой (Ubuntu) и домом (Windows)

### Через Git (лучший способ)
```cmd
# На работе (Ubuntu)
git add Doc/EXPORT_MCP_CONFIG/
git commit -m "MCP config"
git push origin main

# Дома (Windows)
git pull origin main
cd Doc\EXPORT_MCP_CONFIG
.\import_mcp_config.ps1
```

### Через флешку/облако
Скопируй архив:
```
MCP_CONFIG_EXPORT_2026-02-05.tar.gz
```

В Windows распакуй через:
- **7-Zip**: https://www.7-zip.org/
- **WinRAR**: https://www.rarlab.com/
- **Windows 11**: встроенная поддержка tar.gz

---

## ✅ Всё готово!

После установки работай с проектом через Claude Code на Windows точно так же, как на Ubuntu! 🎉

MCP серверы работают одинаково на обеих системах!

---

**Создано**: 2026-02-05
**Система**: Windows 10/11
**GPU**: NVIDIA RTX 3060 + AMD MI100 (через WSL/Linux)
**Автор**: Кодо 💕
