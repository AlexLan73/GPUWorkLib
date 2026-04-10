# MCP в VSCode — Инструкция для Claude

> Составлено: 2026-04-09 | Проект: GPUWorkLib | Автор: Кодо

---

## Как VSCode подхватывает MCP серверы

VSCode (с расширением GitHub Copilot или Claude) читает MCP конфиг из двух мест:

| Уровень | Файл | Область действия |
|---|---|---|
| **Проектный** | `.vscode/mcp.json` | Только этот проект ✅ |
| **Глобальный** | `%APPDATA%\Code\User\mcp.json` (Windows) | Все проекты |
| **Глобальный** | `~/.config/Code/User/mcp.json` (Linux) | Все проекты |

Для GPUWorkLib используем **проектный** `.vscode/mcp.json` — он уже настроен.

---

## Что подключено в .vscode/mcp.json

| Сервер | Назначение при написании кода |
|---|---|
| `sequential-thinking` | Разбор сложных алгоритмов, архитектурных решений |
| `context7` | Документация ROCm, HIP, pybind11, OpenCL, CMake |
| `github` | Поиск референсных реализаций HIP-кернелей |
| `supermemory` | Помнит решения из прошлых сессий |
| `wolfram-alpha` | Верификация DSP-математики (FFT, фильтры, Лагранж) |
| `filesystem` | Читает весь проект GPUWorkLib напрямую |
| `git` | История коммитов, diff, blame |
| `fetch` | Загрузка статей arxiv, документации по URL |
| `memory` | Локальный граф знаний (без облака) |
| `sqlite` | Прямой доступ к results.db |

---

## Настройка VSCode — пошагово

### Шаг 1 — Установить расширение

В VSCode открой Extensions (`Ctrl+Shift+X`) и установи:

```
GitHub Copilot Chat    (ms-vsliveshare.vsliveshare — нет, ms-vscode.copilot-chat)
```

Или для прямого Claude:
```
Claude Dev / Cline     (saoudrizwan.claude-dev)
```

> ℹ️ VSCode поддерживает MCP нативно начиная с версии 1.99+.
> Проверь: `Help → About` — должно быть 1.99 или новее.

### Шаг 2 — Включить MCP в настройках VSCode

Открой `settings.json` (`Ctrl+Shift+P` → "Open User Settings JSON") и добавь:

```json
{
    "chat.mcp.enabled": true,
    "chat.agent.enabled": true,
    "github.copilot.chat.agent.thinkingTool": true
}
```

Или через GUI: `Ctrl+,` → поиск "mcp" → включить `Chat: Mcp Enabled`.

### Шаг 3 — Проверить что .vscode/mcp.json подхватился

1. Открой папку `E:\C++\GPUWorkLib` в VSCode (`File → Open Folder`)
2. Открой Chat панель (`Ctrl+Alt+I` или иконка чата)
3. В чате напиши: `@workspace` — должны появиться MCP инструменты
4. Или нажми иконку 🔧 (Tools) в чате — увидишь список подключённых MCP

### Шаг 4 — Активировать Agent Mode

В Chat панели переключи режим с **Ask** на **Agent**:
- Нажми на выпадающий список рядом с кнопкой отправки
- Выбери **Agent** (или "Claude" если установлен Cline)

В Agent Mode Claude сам решает когда вызывать MCP инструменты.

---

## Глобальный конфиг (для всех проектов)

Если хочешь чтобы MCP работал во всех проектах без `.vscode/mcp.json`:

**Windows:**
```
C:\Users\user\AppData\Roaming\Code\User\mcp.json
```

**Linux:**
```
~/.config/Code/User/mcp.json
```

Содержимое — то же самое что `.vscode/mcp.json`, но без проектных путей:

```json
{
    "servers": {
        "sequential-thinking": {
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
            "env": {}
        },
        "context7": {
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@upstash/context7-mcp@latest"],
            "env": {}
        },
        "github": {
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {
                "GITHUB_PERSONAL_ACCESS_TOKEN": "ВАШ_ТОКЕН"
            }
        },
        "supermemory": {
            "type": "stdio",
            "command": "node",
            "args": ["C:\\Users\\user\\AppData\\Roaming\\Claude\\supermemory-mcp.js"],
            "env": {
                "SUPERMEMORY_API_KEY": "ВАШ_КЛЮЧ",
                "SUPERMEMORY_PROJECT": "GPUWorkLib"
            }
        },
        "wolfram-alpha": {
            "type": "stdio",
            "command": "uvx",
            "args": ["wolfram-alpha-mcp-server"],
            "env": {
                "WOLFRAM_API_KEY": "YX35GA6YP2"
            }
        },
        "fetch": {
            "type": "stdio",
            "command": "uvx",
            "args": ["mcp-server-fetch"],
            "env": {}
        }
    },
    "inputs": []
}
```

---

## Как Claude использует MCP при написании кода

### Автоматически (Agent Mode)
Claude сам вызывает инструменты когда нужно:

```
Ты: "Напиши FIR фильтр на HIP для ROCm"

Claude автоматически:
  1. context7    → читает документацию HIP/ROCm
  2. filesystem  → смотрит существующий код filters/
  3. github      → ищет референсные HIP FIR реализации
  4. sequential-thinking → планирует архитектуру
  5. wolfram-alpha → проверяет формулу весовых коэффициентов
  6. supermemory → вспоминает решения из прошлых сессий
```

### Явный вызов через @ (Ask Mode)
```
@context7 как работает hipMalloc?
@github найди примеры FIR filter HIP
@wolfram-alpha интеграл sinc(x) от -inf до inf
```

### Команды в чате
```
/mcp          — список подключённых серверов
/tools        — список доступных инструментов
```

---

## Workflow при написании GPU кода в VSCode

```
1. Открыть GPUWorkLib в VSCode
2. Переключить Chat в режим Agent
3. Сформулировать задачу:
   "Реализуй [модуль] с учётом архитектуры Ref03, 
    используй ROCm бэкенд, без pytest"

Claude автоматически:
→ Читает CLAUDE.md и MemoryBank через filesystem
→ Смотрит существующий код модуля
→ Ищет документацию через context7
→ Вспоминает предыдущие решения через supermemory
→ Пишет код по правилам проекта
→ Сохраняет решение в supermemory
```

---

## Диагностика VSCode MCP

### Посмотреть логи MCP
```
View → Output → выбрать "MCP" или "GitHub Copilot Chat"
```

### MCP сервер не появляется
1. Убедись что папка открыта как workspace (не отдельный файл)
2. Проверь что `.vscode/mcp.json` валидный JSON
3. Перезапусти VSCode
4. Проверь: `Ctrl+Shift+P` → "MCP: List Servers"

### Инструменты не вызываются автоматически
- Убедись что режим **Agent** (не Ask)
- В настройках: `"chat.agent.enabled": true`
- Явно упомяни в промпте: "используй MCP инструменты"

---

## Связь с Claude Desktop

`.vscode/mcp.json` и `claude_desktop_config.json` — **независимые** конфиги.
Оба используют одни и те же серверы, но запускаются отдельно.

| | Claude Desktop | VSCode |
|---|---|---|
| Конфиг | `AppData\Claude\claude_desktop_config.json` | `.vscode\mcp.json` |
| Supermemory | Node.js скрипт | Node.js скрипт (тот же файл) |
| Wolfram | uvx | uvx |
| Память | Общая (одинаковый API ключ) | Общая (одинаковый API ключ) |

> ✅ Supermemory один на оба клиента — воспоминания из Claude Desktop
> доступны в VSCode и наоборот!

---

*Обновлено: 2026-04-09 | Кодо*
