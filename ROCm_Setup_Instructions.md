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

## 5. Claude Code + MCP Серверы (Linux — воспроизведение окружения)

> Этот раздел описывает полную настройку AI-окружения (Claude Code + MCP серверы + хуки)
> для воспроизведения рабочего окружения проекта GPUWorkLib на Linux-машине.

---

### 5.1 Установка Node.js и Claude Code

```bash
# Node.js 20+ (обязательно)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# Проверка версии (нужна 20+)
node --version

# Claude Code CLI
npm install -g @anthropic-ai/claude-code

# Войти в аккаунт Anthropic
claude auth login
```

---

### 5.2 Используемые MCP серверы

Полный список серверов проекта (Windows: `.mcp.json`, Linux: создать аналогично):

| Сервер | NPM-пакет | Назначение в проекте |
|--------|-----------|----------------------|
| `sequential-thinking` | `@modelcontextprotocol/server-sequential-thinking` | Анализ сложных задач: ROCm-миграция, GPU алгоритмы, рефакторинг |
| `context7` | `@upstash/context7-mcp@latest` | Документация: clFFT, HIP, OpenCL, pybind11, ROCm API |
| `filesystem` | `@modelcontextprotocol/server-filesystem` | Расширенные операции с файлами проекта |
| `memory` | `@modelcontextprotocol/server-memory` | Постоянная память между сессиями (дополнение к MemoryBank/) |
| `sqlite` | `@modelcontextprotocol/server-sqlite` | Доступ к `results.db` — анализ результатов тестов |
| `git` | `@modelcontextprotocol/server-git` | Git-операции через MCP |
| `fetch` | `@modelcontextprotocol/server-fetch` | Загрузка документации по URL (ROCm docs, clFFT, AMD) |
| `testsprite` | `testsprite-mcp` | Тестирование Python-биндингов |
| `dap-debug` | `@uhd_kr/mcp-debug-tools` | Отладка C++ через DAP (breakpoints, stack, variables) |

**Опциональные (нужны API ключи):**

| Сервер | NPM-пакет | Назначение |
|--------|-----------|-----------|
| `github` | `@modelcontextprotocol/server-github` | Поиск референсного кода ROCm/HIP/OpenCL на GitHub |
| `brave-search` | `@modelcontextprotocol/server-brave-search` | Поиск документации GPU без браузера |

---

### 5.3 Создать `.mcp.json` в корне проекта

Скопировать и заменить `PROJECT_PATH` на реальный путь:

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
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "PROJECT_PATH"],
      "env": {}
    },
    "memory": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"],
      "env": {}
    },
    "sqlite": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sqlite", "--db-path", "PROJECT_PATH/results.db"],
      "env": {}
    },
    "git": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-git", "--repository", "PROJECT_PATH"],
      "env": {}
    },
    "fetch": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-fetch"],
      "env": {}
    },
    "testsprite": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "testsprite-mcp"],
      "env": {}
    },
    "dap-debug": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@uhd_kr/mcp-debug-tools@latest"],
      "env": {}
    }
  }
}
```

**Опциональные — добавить при наличии ключей:**
```json
"github": {
  "type": "stdio",
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-github"],
  "env": { "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_ВАШ_ТОКЕН" }
},
"brave-search": {
  "type": "stdio",
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-brave-search"],
  "env": { "BRAVE_API_KEY": "BSA_ВАШ_КЛЮЧ" }
}
```

---

### 5.4 Хуки Claude Code (Linux)

Хуки запускаются автоматически при определённых событиях в Claude Code.

#### Создать директорию и скрипты:

```bash
mkdir -p PROJECT_PATH/.claude/hooks
```

**`.claude/hooks/pre_bash.sh`** — блокировка опасных команд:
```bash
#!/bin/bash
# Hook: PreToolUse(Bash) — защита от опасных команд
# exit 2 = заблокировать команду

DATA=$(cat)
CMD=$(echo "$DATA" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('tool_input', {}).get('command', ''))
except:
    pass
" 2>/dev/null)

DANGEROUS=(
    "git reset --hard"
    "git clean -f"
    "git push --force"
    "git push -f "
    "git branch -D"
    "rm -rf /"
    "rm -rf ~"
)

for PATTERN in "${DANGEROUS[@]}"; do
    if echo "$CMD" | grep -qF "$PATTERN"; then
        echo ""
        echo "⛔ [HOOK] ЗАБЛОКИРОВАНО: опасная операция!"
        echo "   Команда содержит: '$PATTERN'"
        echo "   Подтверди явно в чате, если уверен."
        exit 2
    fi
done

exit 0
```

**`.claude/hooks/post_write.sh`** — напоминание при изменении .cl файлов:
```bash
#!/bin/bash
# Hook: PostToolUse(Write) — проверка kernel файлов

DATA=$(cat)
FILE=$(echo "$DATA" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('tool_input', {}).get('file_path', ''))
except:
    pass
" 2>/dev/null)

if echo "$FILE" | grep -qE '\.cl$'; then
    echo ""
    echo "🔔 [HOOK] Изменён OpenCL kernel: $FILE"
    echo "   Проверь manifest.json в папке kernels/ этого модуля!"
fi

if echo "$FILE" | grep -qE 'CLAUDE\.md$'; then
    echo ""
    echo "📝 [HOOK] CLAUDE.md изменён — не забудь обновить MEMORY.md"
fi

exit 0
```

**`.claude/hooks/on_stop.sh`** — напоминание обновить MemoryBank:
```bash
#!/bin/bash
# Hook: Stop — напоминание в конце сессии

TODAY=$(date +%Y-%m-%d)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 [HOOK] Сессия завершена — $TODAY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   Если что-то важное сделали:"
echo "   1. Создай/обнови: MemoryBank/sessions/$TODAY.md"
echo "   2. Обнови: MemoryBank/MASTER_INDEX.md"
echo "   3. Перенеси задачи в: MemoryBank/tasks/COMPLETED.md"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
exit 0
```

```bash
# Сделать исполняемыми
chmod +x PROJECT_PATH/.claude/hooks/*.sh
```

---

### 5.5 `.claude/settings.local.json` для Linux

Создать файл `PROJECT_PATH/.claude/settings.local.json`:

```json
{
  "permissions": {
    "allow": [
      "Bash(cmake --build*)",
      "Bash(cmake --preset*)",
      "Bash(cmake -B*)",
      "Bash(python3*)",
      "Bash(make*)",
      "Bash(ninja*)",
      "Bash(bash PROJECT_PATH/.claude/hooks/*)",
      "mcp__sequential-thinking__*",
      "mcp__context7__*",
      "mcp__filesystem__*",
      "mcp__memory__*",
      "mcp__sqlite__*",
      "mcp__git__*",
      "mcp__fetch__*",
      "mcp__dap-debug__get-call-stack",
      "mcp__dap-debug__evaluate-expression",
      "mcp__dap-debug__step-over",
      "mcp__dap-debug__step-into",
      "mcp__dap-debug__step-out",
      "mcp__dap-debug__continue",
      "mcp__dap-debug__get-debug-state",
      "mcp__dap-debug__get-variables-scope",
      "mcp__dap-debug__get-debug-console"
    ]
  },
  "enableAllProjectMcpServers": true,
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "bash PROJECT_PATH/.claude/hooks/on_stop.sh"
          }
        ]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "bash PROJECT_PATH/.claude/hooks/pre_bash.sh"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Write",
        "hooks": [
          {
            "type": "command",
            "command": "bash PROJECT_PATH/.claude/hooks/post_write.sh"
          }
        ]
      }
    ]
  }
}
```

> ⚠️ Заменить `PROJECT_PATH` на реальный путь, например `/home/alex/GPUWorkLib`

---

### 5.6 Скрипт автоматической настройки (Linux)

Сохранить как `setup_linux_claude.sh` и запустить из корня проекта:

```bash
#!/bin/bash
# Скрипт настройки Claude Code + MCP для GPUWorkLib на Linux
# Использование: bash setup_linux_claude.sh /path/to/GPUWorkLib

PROJECT_PATH="${1:-$(pwd)}"
echo "🚀 Настройка Claude Code для: $PROJECT_PATH"

# 1. Создать директорию хуков
mkdir -p "$PROJECT_PATH/.claude/hooks"

# 2. Создать хук pre_bash.sh
cat > "$PROJECT_PATH/.claude/hooks/pre_bash.sh" << 'HOOK'
#!/bin/bash
DATA=$(cat)
CMD=$(echo "$DATA" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('tool_input', {}).get('command', ''))
except:
    pass
" 2>/dev/null)
DANGEROUS=("git reset --hard" "git clean -f" "git push --force" "git push -f " "git branch -D" "rm -rf /" "rm -rf ~")
for PATTERN in "${DANGEROUS[@]}"; do
    if echo "$CMD" | grep -qF "$PATTERN"; then
        echo ""; echo "⛔ [HOOK] ЗАБЛОКИРОВАНО: '$PATTERN'"; exit 2
    fi
done
exit 0
HOOK

# 3. Создать хук post_write.sh
cat > "$PROJECT_PATH/.claude/hooks/post_write.sh" << 'HOOK'
#!/bin/bash
DATA=$(cat)
FILE=$(echo "$DATA" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('tool_input', {}).get('file_path', ''))
except:
    pass
" 2>/dev/null)
if echo "$FILE" | grep -qE '\.cl$'; then
    echo "🔔 [HOOK] .cl изменён: $FILE — проверь manifest.json!"
fi
exit 0
HOOK

# 4. Создать хук on_stop.sh
cat > "$PROJECT_PATH/.claude/hooks/on_stop.sh" << 'HOOK'
#!/bin/bash
TODAY=$(date +%Y-%m-%d)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 [HOOK] Сессия завершена — $TODAY"
echo "   Обнови: MemoryBank/sessions/$TODAY.md"
echo "   Обнови: MemoryBank/MASTER_INDEX.md"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
exit 0
HOOK

chmod +x "$PROJECT_PATH/.claude/hooks/"*.sh

# 5. Создать .mcp.json (с подстановкой пути)
cat > "$PROJECT_PATH/.mcp.json" << EOF
{
  "mcpServers": {
    "sequential-thinking": {
      "type": "stdio", "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"], "env": {}
    },
    "context7": {
      "type": "stdio", "command": "npx",
      "args": ["-y", "@upstash/context7-mcp@latest"], "env": {}
    },
    "filesystem": {
      "type": "stdio", "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "$PROJECT_PATH"], "env": {}
    },
    "memory": {
      "type": "stdio", "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"], "env": {}
    },
    "sqlite": {
      "type": "stdio", "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sqlite", "--db-path", "$PROJECT_PATH/results.db"], "env": {}
    },
    "git": {
      "type": "stdio", "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-git", "--repository", "$PROJECT_PATH"], "env": {}
    },
    "fetch": {
      "type": "stdio", "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-fetch"], "env": {}
    },
    "dap-debug": {
      "type": "stdio", "command": "npx",
      "args": ["-y", "@uhd_kr/mcp-debug-tools@latest"], "env": {}
    }
  }
}
EOF

# 6. Создать settings.local.json
cat > "$PROJECT_PATH/.claude/settings.local.json" << EOF
{
  "permissions": {
    "allow": [
      "Bash(cmake --build*)", "Bash(cmake --preset*)", "Bash(cmake -B*)",
      "Bash(python3*)", "Bash(make*)", "Bash(ninja*)",
      "Bash(bash $PROJECT_PATH/.claude/hooks/*)",
      "mcp__sequential-thinking__*", "mcp__context7__*",
      "mcp__filesystem__*", "mcp__memory__*", "mcp__sqlite__*",
      "mcp__git__*", "mcp__fetch__*",
      "mcp__dap-debug__get-call-stack", "mcp__dap-debug__evaluate-expression",
      "mcp__dap-debug__step-over", "mcp__dap-debug__continue",
      "mcp__dap-debug__get-debug-state", "mcp__dap-debug__get-variables-scope"
    ]
  },
  "enableAllProjectMcpServers": true,
  "hooks": {
    "Stop": [{"hooks": [{"type": "command", "command": "bash $PROJECT_PATH/.claude/hooks/on_stop.sh"}]}],
    "PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "bash $PROJECT_PATH/.claude/hooks/pre_bash.sh"}]}],
    "PostToolUse": [{"matcher": "Write", "hooks": [{"type": "command", "command": "bash $PROJECT_PATH/.claude/hooks/post_write.sh"}]}]
  }
}
EOF

echo ""
echo "✅ Готово! Файлы созданы:"
echo "   $PROJECT_PATH/.mcp.json"
echo "   $PROJECT_PATH/.claude/settings.local.json"
echo "   $PROJECT_PATH/.claude/hooks/pre_bash.sh"
echo "   $PROJECT_PATH/.claude/hooks/post_write.sh"
echo "   $PROJECT_PATH/.claude/hooks/on_stop.sh"
echo ""
echo "🚀 Запуск: cd $PROJECT_PATH && claude"
```

```bash
# Использование:
chmod +x setup_linux_claude.sh
bash setup_linux_claude.sh /home/alex/GPUWorkLib
```

---

*Создано: 2026-02-24*
*MCP раздел добавлен: 2026-02-23*
