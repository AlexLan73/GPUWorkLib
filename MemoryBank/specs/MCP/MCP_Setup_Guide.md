# MCP Servers — Инструкция по подключению

> Составлено: 2026-04-09 | Платформа: Windows + Debian Linux  
> Проект: GPUWorkLib | Автор: Кодо

---

## Обзор — что подключено

| Сервер | Назначение | Статус Windows | Статус Debian |
|---|---|---|---|
| `sequential-thinking` | Разбор сложных задач по шагам | ✅ | ✅ |
| `github` | Поиск кода, репозиториев | ✅ | ✅ |
| `supermemory` | Долгосрочная память между сессиями | ✅ (Node.js скрипт) | ✅ |
| `wolfram-alpha` | Точная математика, символьные вычисления | ✅ | ✅ |
| `filesystem` | Прямой доступ к файлам проекта | ✅ | ✅ |
| `context7` | Документация по библиотекам (в .mcp.json) | ✅ | ✅ |

---

## Часть 1 — Windows (Claude Desktop)

### Расположение конфига
```
C:\Users\user\AppData\Roaming\Claude\claude_desktop_config.json
```

### Итоговый рабочий конфиг

```json
{
  "mcpServers": {
    "sequential-thinking": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"]
    },

    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ВАШ_GITHUB_PAT_ТОКЕН"
      }
    },

    "supermemory": {
      "command": "node",
      "args": ["C:\\Users\\user\\AppData\\Roaming\\Claude\\supermemory-mcp.js"],
      "env": {
        "SUPERMEMORY_API_KEY": "ВАШ_SUPERMEMORY_API_KEY",
        "SUPERMEMORY_PROJECT": "GPUWorkLib"
      }
    },

    "wolfram-alpha": {
      "command": "uvx",
      "args": ["wolfram-alpha-mcp-server"],
      "env": {
        "WOLFRAM_API_KEY": "ВАШ_WOLFRAM_APP_ID"
      }
    },

    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "E:\\C++\\GPUWorkLib",
        "E:\\C++\\Refactoring",
        "C:\\Users\\user\\AppData\\Roaming\\Claude"
      ]
    }
  }
}
```

### Важные заметки Windows

**Supermemory — особый случай:**
- Claude Desktop v1.1.4498 НЕ поддерживает формат `url`+`headers` для MCP
- `mcp-remote` падает на Windows мгновенно (известная проблема)
- `@supermemory/mcp` пакет — НЕ существует на npm (404)
- **Решение:** кастомный Node.js скрипт `supermemory-mcp.js` в AppData

Файл скрипта: `C:\Users\user\AppData\Roaming\Claude\supermemory-mcp.js`  
(уже создан, трогать не нужно)

**Wolfram — важно:**
- Переменная называется `WOLFRAM_API_KEY` (НЕ `WOLFRAM_APP_ID`!)
- App ID берётся с https://developer.wolframalpha.com
- Python 3.13+ нужен для uvx — у нас Python 3.14 на `F:\Program Files (x86)\Python314`

---

## Часть 2 — Debian Linux (Claude Code / .mcp.json)

### Расположение конфига
```
/home/user/.config/claude/claude_desktop_config.json   # Claude Desktop (если установлен)
E:/C++/GPUWorkLib/.mcp.json                            # Claude Code (проектный)
```

### Требования

```bash
# Node.js 18+
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Python 3.13+ и uv
sudo apt install python3.13 python3.13-venv
pip install uv --break-system-packages
# или через curl:
curl -LsSf https://astral.sh/uv/install.sh | sh

# npx уже есть с Node.js
```

### Конфиг для Claude Desktop на Debian

Файл: `~/.config/claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "sequential-thinking": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"]
    },

    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ВАШ_GITHUB_PAT_ТОКЕН"
      }
    },

    "supermemory": {
      "url": "https://mcp.supermemory.ai/mcp",
      "headers": {
        "Authorization": "Bearer ВАШ_SUPERMEMORY_API_KEY",
        "x-sm-project": "GPUWorkLib"
      }
    },

    "wolfram-alpha": {
      "command": "uvx",
      "args": ["wolfram-alpha-mcp-server"],
      "env": {
        "WOLFRAM_API_KEY": "ВАШ_WOLFRAM_APP_ID"
      }
    },

    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/home/user/projects/GPUWorkLib"
      ]
    }
  }
}
```

> ⚠️ **На Debian supermemory использует `url`+`headers` формат** — это работает
> в новых версиях Claude Desktop для Linux. Node.js скрипт там НЕ нужен.

### Конфиг для Claude Code (.mcp.json в репозитории)

Файл уже существует: `E:/C++/GPUWorkLib/.mcp.json`  
На Linux путь: `/path/to/GPUWorkLib/.mcp.json`

```json
{
  "mcpServers": {
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
    "filesystem": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/GPUWorkLib"],
      "env": {}
    },
    "memory": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"],
      "env": {}
    },
    "git": {
      "type": "stdio",
      "command": "uvx",
      "args": ["mcp-server-git", "--repository", "/path/to/GPUWorkLib"],
      "env": {}
    },
    "fetch": {
      "type": "stdio",
      "command": "uvx",
      "args": ["mcp-server-fetch"],
      "env": {}
    },
    "repomix": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "repomix-mcp"],
      "env": {}
    },
    "github": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ВАШ_GITHUB_PAT_ТОКЕН"
      }
    }
  }
}
```

### Установка на Debian — пошаговая инструкция

```bash
# 1. Клонировать репозиторий
git clone https://github.com/AlexLan73/GPUWorkLib.git
cd GPUWorkLib

# 2. Установить Node.js 20+
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
node --version   # должно быть v20+

# 3. Установить uv (для Python MCP серверов)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uvx --version

# 4. Проверить что npx работает
npx --version

# 5. Первый запуск sequential-thinking (скачается автоматически)
npx -y @modelcontextprotocol/server-sequential-thinking --help

# 6. Проверить wolfram-alpha
WOLFRAM_API_KEY=YX35GA6YP2 uvx wolfram-alpha-mcp-server

# 7. Установить Claude Code (если нужен)
npm install -g @anthropic-ai/claude-code
claude --version

# 8. Запустить Claude Code в проекте
cd /path/to/GPUWorkLib
claude
# Claude Code автоматически подхватит .mcp.json из корня проекта
```

### Supermemory на Debian

**Вариант A — через URL (предпочтительный, если Claude Desktop новый):**
```json
"supermemory": {
  "url": "https://mcp.supermemory.ai/mcp",
  "headers": {
    "Authorization": "Bearer ВАШ_SUPERMEMORY_API_KEY"
  }
}
```

**Вариант B — через Node.js скрипт (если URL формат не работает):**
```bash
# Скопировать скрипт из Windows или создать заново
cp /mnt/windows/Users/user/AppData/Roaming/Claude/supermemory-mcp.js ~/
```
```json
"supermemory": {
  "command": "node",
  "args": ["/home/user/supermemory-mcp.js"],
  "env": {
    "SUPERMEMORY_API_KEY": "ВАШ_SUPERMEMORY_API_KEY",
    "SUPERMEMORY_PROJECT": "GPUWorkLib"
  }
}
```

---

## Часть 3 — Ключи и токены

| Сервис | Где получить | Переменная |
|---|---|---|
| GitHub PAT | https://github.com/settings/tokens → Classic → repo, read:org | `GITHUB_PERSONAL_ACCESS_TOKEN` |
| Supermemory | https://app.supermemory.ai → Settings → API Keys | `SUPERMEMORY_API_KEY` |
| Wolfram Alpha | https://developer.wolframalpha.com → My Apps | `WOLFRAM_API_KEY` |

> ⚠️ Wolfram: переменная называется `WOLFRAM_API_KEY`, НЕ `WOLFRAM_APP_ID`!

---

## Часть 4 — Диагностика

### Windows — логи Claude Desktop
```
C:\Users\user\AppData\Roaming\Claude\logs\mcp-server-<name>.log
```

### Debian — логи Claude Desktop
```bash
~/.config/claude/logs/mcp-server-<name>.log
tail -f ~/.config/claude/logs/mcp-server-supermemory.log
```

### Частые ошибки и решения

| Ошибка | Причина | Решение |
|---|---|---|
| `not valid MCP server configurations` | Claude Desktop старый, не поддерживает `url` формат | Использовать `command`/`args` формат |
| `WOLFRAM_API_KEY environment variable not set` | Неправильное имя переменной | Переименовать `WOLFRAM_APP_ID` → `WOLFRAM_API_KEY` |
| `@supermemory/mcp: 404 Not Found` | Пакет не существует на npm | Использовать Node.js скрипт или `url` формат |
| `mcp-remote` падает мгновенно | Известная проблема Windows | Использовать Node.js скрипт напрямую |
| `'C:\Program' is not recognized` | Пробел в пути к Node.js | Использовать короткий путь или кавычки |
| `Server disconnected` сразу | Процесс падает до инициализации | Смотреть лог, искать traceback |

### Проверка что всё работает
После запуска Claude Desktop в логе должно быть:
```
[info] Server started and connected successfully
[info] Message from server: {"result":{"serverInfo":{"name":"..."}}}
[info] Message from server: {"result":{"tools":[...]}}
```

---

## Часть 5 — Workflow использования

```
Новая задача (DSP, GPU, алгоритм):
  1. Формулируем задачу с Alex
  2. sequential-thinking → разбиваем на шаги
  3. Context7 → документация по библиотекам (ROCm, HIP, pybind11...)
  4. web_fetch → статьи arxiv/IEEE по алгоритму
  5. github → ищем референсные реализации
  6. filesystem → читаем текущий код GPUWorkLib
  7. wolfram-alpha → верифицируем математику (FFT, фильтры, Лагранж)
  8. Синтез решения
  9. supermemory → сохраняем принятое решение для следующих сессий
```

---

*Обновлено: 2026-04-09 | Кодо*
