# Подключение GitHub MCP Server к Claude Code

## Что это даёт
После настройки Кодо сможет искать репозитории на GitHub, читать код,
смотреть issues и PR прямо во время работы с проектом.

---

## Требования
- Установлен **Node.js** (проверить: `node --version`)
- Установлен **Claude Code** (`npm install -g @anthropic-ai/claude-code`)

---

## Шаг 1 — Найти файл `.mcp.json`

Файл лежит в корне проекта GPUWorkLib:
```
/path/to/GPUWorkLib/.mcp.json
```

Открыть любым редактором.

---

## Шаг 2 — Добавить секцию `github`

Найти закрывающую скобку `}` в конце файла и добавить перед ней:

```json
,
"github": {
  "type": "stdio",
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-github"],
  "env": {
    "GITHUB_PERSONAL_ACCESS_TOKEN": "ТОКЕН_ИЗ_ФАЙЛА_github_token.txt"
  }
}
```

### Пример готового `.mcp.json` (только нужная часть):
```json
{
  "mcpServers": {
    "sequential-thinking": { ... },
    "context7": { ... },
    "github": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ТОКЕН_ИЗ_ФАЙЛА_github_token.txt"
      }
    }
  }
}
```

---

## Шаг 3 — Перезапустить Claude Code

Закрыть и открыть снова. При следующем запуске Claude Code сам спросит:
> *"Approve MCP server 'github'?"* → нажать **Yes**

---

## Шаг 4 — Проверить

В Claude Code написать:
```
/mcp
```
В списке должен появиться `github` со статусом ✅ connected.

Или попросить меня:
> "Найди на GitHub примеры HIP kernels для ROCm"

---

## Если что-то не работает

### Проверить Node.js и npx:
```bash
node --version   # должно быть v18+
npx --version
```

### Установить Node.js (если нет):
```bash
# Ubuntu/Debian
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo bash -
sudo apt install -y nodejs

# Windows — скачать с nodejs.org
```

### Проверить токен вручную:
```bash
curl -s -H "Authorization: token ТОКЕН_ИЗ_ФАЙЛА_github_token.txt" \
     https://api.github.com/user | grep login
# должно вывести: "login": "AlexLan73"
```

---

## Примечание по безопасности
Токен даёт **read-only** доступ к публичным репозиториям.
Не публиковать токен в открытых репо!
Файл `.mcp.json` уже добавлен в `.gitignore` если там есть токен.

---

*Настроено: 2026-03-10 | Проект: GPUWorkLib*
