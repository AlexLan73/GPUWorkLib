# 🚀 PowerShell скрипт импорта MCP конфигурации для Windows
# Использование: .\import_mcp_config.ps1 [путь_к_проекту]

param(
    [string]$ProjectPath = $PWD.Path
)

Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Blue
Write-Host "║       🚀 Импорт MCP конфигурации GPUWorkLib 🚀           ║" -ForegroundColor Blue
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Blue
Write-Host ""

# Убираем trailing slash
$ProjectPath = $ProjectPath.TrimEnd('\', '/')

Write-Host "📂 Путь к проекту: $ProjectPath" -ForegroundColor Yellow
Write-Host ""

# Проверка, что мы в правильной директории
if (-not (Test-Path "$ProjectPath\modules")) {
    Write-Host "❌ Ошибка: Не похоже на GPUWorkLib проект!" -ForegroundColor Red
    Write-Host "   Не найдена папка 'modules'" -ForegroundColor Red
    Write-Host ""
    Write-Host "Использование: .\import_mcp_config.ps1 C:\путь\к\GPUWorkLib" -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ Проект найден" -ForegroundColor Green
Write-Host ""

# Создаем базу данных results.db если не существует
$DbPath = Join-Path $ProjectPath "results.db"
if (-not (Test-Path $DbPath)) {
    New-Item -Path $DbPath -ItemType File -Force | Out-Null
    Write-Host "✓ Создана база данных: $DbPath" -ForegroundColor Green
} else {
    Write-Host "✓ База данных уже существует" -ForegroundColor Green
}

Write-Host ""
Write-Host "🔧 Установка MCP серверов..." -ForegroundColor Yellow
Write-Host ""

# Функция для добавления сервера
function Add-MCPServer {
    param(
        [string]$Name,
        [string[]]$Args
    )

    Write-Host "→ Добавляю $Name..." -ForegroundColor Blue

    $result = & claude mcp add $Name -- $Args 2>&1

    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ $Name добавлен" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ $Name уже существует или ошибка" -ForegroundColor Yellow
    }
}

# 1. Sequential Thinking
Add-MCPServer "sequential-thinking" @("npx", "-y", "@modelcontextprotocol/server-sequential-thinking")

# 2. Context7
Add-MCPServer "context7" @("npx", "-y", "@upstash/context7-mcp@latest")

# 3. Filesystem
Add-MCPServer "filesystem" @("npx", "-y", "@modelcontextprotocol/server-filesystem", $ProjectPath)

# 4. Memory
Add-MCPServer "memory" @("npx", "-y", "@modelcontextprotocol/server-memory")

# 5. SQLite
Add-MCPServer "sqlite" @("npx", "-y", "@modelcontextprotocol/server-sqlite", "--db-path", $DbPath)

# 6. Git
Add-MCPServer "git" @("npx", "-y", "@modelcontextprotocol/server-git", "--repository", $ProjectPath)

# 7. Fetch
Add-MCPServer "fetch" @("npx", "-y", "@modelcontextprotocol/server-fetch")

Write-Host ""
Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Blue
Write-Host ""
Write-Host "✅ Базовая конфигурация установлена!" -ForegroundColor Green
Write-Host ""

# Проверка серверов
Write-Host "📊 Проверка установленных серверов..." -ForegroundColor Yellow
Write-Host ""
& claude mcp list
Write-Host ""

Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Blue
Write-Host ""
Write-Host "📋 Дополнительные серверы (установить вручную):" -ForegroundColor Yellow
Write-Host ""
Write-Host "1️⃣  GitHub MCP:" -ForegroundColor Blue
Write-Host "   # Установка GitHub CLI для Windows:"
Write-Host "   winget install --id GitHub.cli"
Write-Host "   # Или скачай: https://cli.github.com/"
Write-Host ""
Write-Host "   gh auth login"
Write-Host "   `$env:GITHUB_TOKEN = gh auth token"
Write-Host "   claude mcp add github -e GITHUB_PERSONAL_ACCESS_TOKEN=`$env:GITHUB_TOKEN -- npx -y @modelcontextprotocol/server-github"
Write-Host ""
Write-Host "2️⃣  Brave Search:" -ForegroundColor Blue
Write-Host "   → Получить ключ: https://brave.com/search/api/"
Write-Host "   claude mcp add brave-search -e BRAVE_API_KEY=ваш_ключ -- npx -y @modelcontextprotocol/server-brave-search"
Write-Host ""
Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Blue
Write-Host ""
Write-Host "🎉 Готово! MCP серверы настроены для:" -ForegroundColor Green
Write-Host "   $ProjectPath" -ForegroundColor Green
Write-Host ""
Write-Host "📚 Документация:" -ForegroundColor Yellow
Write-Host "   $ProjectPath\Doc\MCP_SERVERS_SETUP.md"
Write-Host "   $ProjectPath\Doc\MCP_CHEATSHEET.md"
Write-Host "   $ProjectPath\Doc\MANUAL_INSTALL_GITHUB_BRAVE.md"
Write-Host ""
