@echo off
REM ============================================================================
REM   🚀 Импорт MCP конфигурации GPUWorkLib для Windows (CMD/BAT версия)
REM   Использование: import_mcp_config.bat [путь_к_проекту]
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║       🚀 Импорт MCP конфигурации GPUWorkLib 🚀           ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

REM Определяем путь к проекту
if "%~1"=="" (
    set "PROJECT_PATH=%CD%"
) else (
    set "PROJECT_PATH=%~1"
)

echo 📂 Путь к проекту: %PROJECT_PATH%
echo.

REM Проверка наличия папки modules
if not exist "%PROJECT_PATH%\modules" (
    echo ❌ Ошибка: Не похоже на GPUWorkLib проект!
    echo    Не найдена папка 'modules'
    echo.
    echo Использование: import_mcp_config.bat C:\путь\к\GPUWorkLib
    exit /b 1
)

echo ✓ Проект найден
echo.

REM Создаем базу данных results.db
set "DB_PATH=%PROJECT_PATH%\results.db"
if not exist "%DB_PATH%" (
    type nul > "%DB_PATH%"
    echo ✓ Создана база данных: %DB_PATH%
) else (
    echo ✓ База данных уже существует
)

echo.
echo 🔧 Установка MCP серверов...
echo.

REM 1. Sequential Thinking
echo → Добавляю sequential-thinking...
claude mcp add sequential-thinking -- npx -y @modelcontextprotocol/server-sequential-thinking >nul 2>&1
if !errorlevel! equ 0 (
    echo   ✓ sequential-thinking добавлен
) else (
    echo   ⚠ sequential-thinking уже существует или ошибка
)

REM 2. Context7
echo → Добавляю context7...
claude mcp add context7 -- npx -y @upstash/context7-mcp@latest >nul 2>&1
if !errorlevel! equ 0 (
    echo   ✓ context7 добавлен
) else (
    echo   ⚠ context7 уже существует или ошибка
)

REM 3. Filesystem
echo → Добавляю filesystem...
claude mcp add filesystem -- npx -y @modelcontextprotocol/server-filesystem "%PROJECT_PATH%" >nul 2>&1
if !errorlevel! equ 0 (
    echo   ✓ filesystem добавлен
) else (
    echo   ⚠ filesystem уже существует или ошибка
)

REM 4. Memory
echo → Добавляю memory...
claude mcp add memory -- npx -y @modelcontextprotocol/server-memory >nul 2>&1
if !errorlevel! equ 0 (
    echo   ✓ memory добавлен
) else (
    echo   ⚠ memory уже существует или ошибка
)

REM 5. SQLite
echo → Добавляю sqlite...
claude mcp add sqlite -- npx -y @modelcontextprotocol/server-sqlite --db-path "%DB_PATH%" >nul 2>&1
if !errorlevel! equ 0 (
    echo   ✓ sqlite добавлен
) else (
    echo   ⚠ sqlite уже существует или ошибка
)

REM 6. Git
echo → Добавляю git...
claude mcp add git -- npx -y @modelcontextprotocol/server-git --repository "%PROJECT_PATH%" >nul 2>&1
if !errorlevel! equ 0 (
    echo   ✓ git добавлен
) else (
    echo   ⚠ git уже существует или ошибка
)

REM 7. Fetch
echo → Добавляю fetch...
claude mcp add fetch -- npx -y @modelcontextprotocol/server-fetch >nul 2>&1
if !errorlevel! equ 0 (
    echo   ✓ fetch добавлен
) else (
    echo   ⚠ fetch уже существует или ошибка
)

echo.
echo ════════════════════════════════════════════════════════════
echo.
echo ✅ Базовая конфигурация установлена!
echo.

REM Проверка серверов
echo 📊 Проверка установленных серверов...
echo.
claude mcp list
echo.

echo ════════════════════════════════════════════════════════════
echo.
echo 📋 Дополнительные серверы (установить вручную):
echo.
echo 1️⃣  GitHub MCP:
echo    winget install --id GitHub.cli
echo    gh auth login
echo    set GITHUB_TOKEN=^^(gh auth token^^)
echo    claude mcp add github -e GITHUB_PERSONAL_ACCESS_TOKEN=%%GITHUB_TOKEN%% -- npx -y @modelcontextprotocol/server-github
echo.
echo 2️⃣  Brave Search:
echo    → Получить ключ: https://brave.com/search/api/
echo    claude mcp add brave-search -e BRAVE_API_KEY=ваш_ключ -- npx -y @modelcontextprotocol/server-brave-search
echo.
echo ════════════════════════════════════════════════════════════
echo.
echo 🎉 Готово! MCP серверы настроены для:
echo    %PROJECT_PATH%
echo.
echo 📚 Документация:
echo    %PROJECT_PATH%\Doc\MCP_SERVERS_SETUP.md
echo    %PROJECT_PATH%\Doc\MCP_CHEATSHEET.md
echo.

pause
