#!/bin/bash
# 🚀 Скрипт установки MCP серверов для GPUWorkLib
# Использование: ./install_mcp_servers.sh

set -e

echo "🔧 Установка MCP серверов для GPUWorkLib"
echo "=========================================="
echo ""

# Цвета для вывода
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Проверка базовых серверов
echo "📋 Проверка установленных серверов..."
claude mcp list

echo ""
echo "=========================================="
echo "🛠️  Установка дополнительных серверов"
echo "=========================================="
echo ""

# 1. Установка GitHub CLI
echo "1️⃣  GitHub CLI и MCP сервер"
if ! command -v gh &> /dev/null; then
    echo -e "${YELLOW}GitHub CLI не найден. Устанавливаю...${NC}"
    sudo apt update
    sudo apt install -y gh
    echo -e "${GREEN}✓ GitHub CLI установлен${NC}"
else
    echo -e "${GREEN}✓ GitHub CLI уже установлен${NC}"
fi

# Проверка авторизации GitHub
if gh auth status &> /dev/null; then
    echo -e "${GREEN}✓ GitHub авторизован${NC}"

    # Добавление GitHub MCP сервера
    export GITHUB_TOKEN=$(gh auth token)
    if [ ! -z "$GITHUB_TOKEN" ]; then
        echo "Добавляю GitHub MCP сервер..."
        claude mcp add github -e GITHUB_PERSONAL_ACCESS_TOKEN=$GITHUB_TOKEN -- npx -y @modelcontextprotocol/server-github || true
        echo -e "${GREEN}✓ GitHub MCP добавлен${NC}"
    fi
else
    echo -e "${YELLOW}⚠ GitHub не авторизован. Запустите: gh auth login${NC}"
    echo "  Выберите: GitHub.com -> SSH -> Your SSH public key"
fi

echo ""

# 2. Brave Search
echo "2️⃣  Brave Search MCP (для поиска документации)"
read -p "У вас есть Brave API ключ? (y/n): " has_brave_key

if [ "$has_brave_key" = "y" ] || [ "$has_brave_key" = "Y" ]; then
    read -p "Введите Brave API ключ: " brave_key
    claude mcp add brave-search -e BRAVE_API_KEY=$brave_key -- npx -y @modelcontextprotocol/server-brave-search
    echo -e "${GREEN}✓ Brave Search MCP добавлен${NC}"
else
    echo -e "${YELLOW}ℹ Получите бесплатный ключ на: https://brave.com/search/api/${NC}"
    echo -e "${YELLOW}  Бесплатный план: 2000 запросов/месяц${NC}"
fi

echo ""

# 3. SQLite для результатов
echo "3️⃣  SQLite MCP (для хранения результатов тестов)"
DB_PATH="/home/alex/C++/GPUWorkLib/results.db"

if [ ! -f "$DB_PATH" ]; then
    touch "$DB_PATH"
    echo -e "${GREEN}✓ База данных создана: $DB_PATH${NC}"
fi

claude mcp add sqlite -- npx -y @modelcontextprotocol/server-sqlite --db-path "$DB_PATH" || {
    echo -e "${YELLOW}⚠ Не удалось добавить SQLite MCP (возможно, уже установлен)${NC}"
}

echo ""

# 4. Git MCP
echo "4️⃣  Git MCP (расширенная работа с репозиторием)"
claude mcp add git -- npx -y @modelcontextprotocol/server-git || {
    echo -e "${YELLOW}⚠ Не удалось добавить Git MCP (возможно, уже установлен)${NC}"
}

echo ""

# 5. Fetch MCP
echo "5️⃣  Fetch MCP (загрузка документации и примеров)"
claude mcp add fetch -- npx -y @modelcontextprotocol/server-fetch || {
    echo -e "${YELLOW}⚠ Не удалось добавить Fetch MCP (возможно, уже установлен)${NC}"
}

echo ""
echo "=========================================="
echo "🎉 Установка завершена!"
echo "=========================================="
echo ""

# Итоговая проверка
echo "📊 Финальный список серверов:"
claude mcp list

echo ""
echo "📚 Документация: Doc/MCP_SERVERS_SETUP.md"
echo ""
echo "🚀 Рекомендации:"
echo "  1. Если GitHub не авторизован: gh auth login"
echo "  2. Для Brave Search получите ключ: https://brave.com/search/api/"
echo "  3. Проверьте работу: claude mcp list"
echo ""
echo -e "${GREEN}✓ Все готово для работы!${NC}"
