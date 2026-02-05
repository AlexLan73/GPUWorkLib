#!/bin/bash
# 🚀 Скрипт импорта MCP конфигурации на новый компьютер
# Использование: ./import_mcp_config.sh [путь_к_проекту]

set -e

# Цвета
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       🚀 Импорт MCP конфигурации GPUWorkLib 🚀           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Определяем путь к проекту
if [ -z "$1" ]; then
    # Если не передан аргумент, используем текущую директорию
    PROJECT_PATH=$(pwd)
else
    PROJECT_PATH="$1"
fi

# Убираем trailing slash
PROJECT_PATH="${PROJECT_PATH%/}"

echo -e "${YELLOW}📂 Путь к проекту: ${PROJECT_PATH}${NC}"
echo ""

# Проверка, что мы в правильной директории
if [ ! -d "$PROJECT_PATH/modules" ]; then
    echo -e "${RED}❌ Ошибка: Не похоже на GPUWorkLib проект!${NC}"
    echo -e "${RED}   Не найдена папка 'modules'${NC}"
    echo ""
    echo "Использование: ./import_mcp_config.sh /путь/к/GPUWorkLib"
    exit 1
fi

echo -e "${GREEN}✓ Проект найден${NC}"
echo ""

# Создаем базу данных results.db если не существует
if [ ! -f "$PROJECT_PATH/results.db" ]; then
    touch "$PROJECT_PATH/results.db"
    echo -e "${GREEN}✓ Создана база данных: $PROJECT_PATH/results.db${NC}"
else
    echo -e "${GREEN}✓ База данных уже существует${NC}"
fi

echo ""
echo -e "${YELLOW}🔧 Установка MCP серверов...${NC}"
echo ""

# Функция для добавления сервера
add_server() {
    local name=$1
    shift
    echo -e "${BLUE}→ Добавляю $name...${NC}"
    if claude mcp add "$name" "$@" 2>/dev/null; then
        echo -e "${GREEN}  ✓ $name добавлен${NC}"
    else
        echo -e "${YELLOW}  ⚠ $name уже существует или ошибка${NC}"
    fi
}

# 1. Sequential Thinking
add_server "sequential-thinking" -- npx -y @modelcontextprotocol/server-sequential-thinking

# 2. Context7
add_server "context7" -- npx -y @upstash/context7-mcp@latest

# 3. Filesystem
add_server "filesystem" -- npx -y @modelcontextprotocol/server-filesystem "$PROJECT_PATH"

# 4. Memory
add_server "memory" -- npx -y @modelcontextprotocol/server-memory

# 5. SQLite
add_server "sqlite" -- npx -y @modelcontextprotocol/server-sqlite --db-path "$PROJECT_PATH/results.db"

# 6. Git
add_server "git" -- npx -y @modelcontextprotocol/server-git --repository "$PROJECT_PATH"

# 7. Fetch
add_server "fetch" -- npx -y @modelcontextprotocol/server-fetch

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${GREEN}✅ Базовая конфигурация установлена!${NC}"
echo ""

# Проверка серверов
echo -e "${YELLOW}📊 Проверка установленных серверов...${NC}"
echo ""
claude mcp list
echo ""

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}📋 Дополнительные серверы (установить вручную):${NC}"
echo ""
echo -e "${BLUE}1️⃣  GitHub MCP:${NC}"
echo "   sudo apt install -y gh  # или: sudo snap install gh"
echo "   gh auth login"
echo "   export GITHUB_TOKEN=\$(gh auth token)"
echo "   claude mcp add github -e GITHUB_PERSONAL_ACCESS_TOKEN=\$GITHUB_TOKEN -- npx -y @modelcontextprotocol/server-github"
echo ""
echo -e "${BLUE}2️⃣  Brave Search:${NC}"
echo "   → Получить ключ: https://brave.com/search/api/"
echo "   claude mcp add brave-search -e BRAVE_API_KEY=ваш_ключ -- npx -y @modelcontextprotocol/server-brave-search"
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${GREEN}🎉 Готово! MCP серверы настроены для:${NC}"
echo -e "${GREEN}   $PROJECT_PATH${NC}"
echo ""
echo -e "${YELLOW}📚 Документация:${NC}"
echo "   $PROJECT_PATH/Doc/MCP_SERVERS_SETUP.md"
echo "   $PROJECT_PATH/Doc/MCP_CHEATSHEET.md"
echo "   $PROJECT_PATH/Doc/MANUAL_INSTALL_GITHUB_BRAVE.md"
echo ""
