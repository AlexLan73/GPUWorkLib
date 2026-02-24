#!/bin/bash
# =============================================================================
# Скрипт настройки Claude Code + MCP для GPUWorkLib на Linux
# =============================================================================
# Использование:
#   bash setup_linux_claude.sh /path/to/GPUWorkLib
#   bash setup_linux_claude.sh          # текущая директория
# =============================================================================

PROJECT_PATH="${1:-$(pwd)}"
echo ""
echo "🚀 Настройка Claude Code для: $PROJECT_PATH"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. Создать директорию хуков
mkdir -p "$PROJECT_PATH/.claude/hooks"
echo "📁 Создана директория: .claude/hooks/"

# 2. Хук pre_bash.sh — блокировка опасных команд
cat > "$PROJECT_PATH/.claude/hooks/pre_bash.sh" << 'HOOK_EOF'
#!/bin/bash
# Hook: PreToolUse(Bash) — защита от опасных команд
# exit 2 = заблокировать команду и показать сообщение

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
HOOK_EOF
echo "✅ Создан: .claude/hooks/pre_bash.sh"

# 3. Хук post_write.sh — напоминание при изменении .cl файлов
cat > "$PROJECT_PATH/.claude/hooks/post_write.sh" << 'HOOK_EOF'
#!/bin/bash
# Hook: PostToolUse(Write) — проверка kernel файлов и CLAUDE.md

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
HOOK_EOF
echo "✅ Создан: .claude/hooks/post_write.sh"

# 4. Хук on_stop.sh — напоминание обновить MemoryBank
cat > "$PROJECT_PATH/.claude/hooks/on_stop.sh" << 'HOOK_EOF'
#!/bin/bash
# Hook: Stop — напоминание в конце сессии

TODAY=$(date +%Y-%m-%d)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 [HOOK] Сессия завершена — $TODAY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   Если что-то важное сделали:"
echo "   1. Создай/обнови: MemoryBank/sessions/$TODAY.md"
echo "   2. Обнови:        MemoryBank/MASTER_INDEX.md"
echo "   3. Перенеси задачи в: MemoryBank/tasks/COMPLETED.md"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
exit 0
HOOK_EOF
echo "✅ Создан: .claude/hooks/on_stop.sh"

# 5. Сделать хуки исполняемыми
chmod +x "$PROJECT_PATH/.claude/hooks/"*.sh
echo "🔒 chmod +x применён к хукам"

# 6. Создать .mcp.json
cat > "$PROJECT_PATH/.mcp.json" << EOF
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
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "$PROJECT_PATH"],
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
      "args": ["-y", "@modelcontextprotocol/server-sqlite", "--db-path", "$PROJECT_PATH/results.db"],
      "env": {}
    },
    "git": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-git", "--repository", "$PROJECT_PATH"],
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
EOF
echo "✅ Создан: .mcp.json"

# 7. Создать .claude/settings.local.json
cat > "$PROJECT_PATH/.claude/settings.local.json" << EOF
{
  "permissions": {
    "allow": [
      "Bash(cmake --build*)",
      "Bash(cmake --preset*)",
      "Bash(cmake -B*)",
      "Bash(python3*)",
      "Bash(make*)",
      "Bash(ninja*)",
      "Bash(bash $PROJECT_PATH/.claude/hooks/*)",
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
            "command": "bash $PROJECT_PATH/.claude/hooks/on_stop.sh"
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
            "command": "bash $PROJECT_PATH/.claude/hooks/pre_bash.sh"
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
            "command": "bash $PROJECT_PATH/.claude/hooks/post_write.sh"
          }
        ]
      }
    ]
  }
}
EOF
echo "✅ Создан: .claude/settings.local.json"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 Настройка завершена!"
echo ""
echo "   Опционально — добавить API ключи в .mcp.json:"
echo "   • GitHub:      GITHUB_PERSONAL_ACCESS_TOKEN"
echo "   • BraveSearch: BRAVE_API_KEY"
echo ""
echo "   Запуск Claude Code:"
echo "   cd $PROJECT_PATH && claude"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
