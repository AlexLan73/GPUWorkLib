# 📋 Ручная установка GitHub и Brave Search MCP

## 🔧 Проблема с apt lock

Если видишь ошибку:
```
E: Не удалось получить блокировку файла /var/lib/apt/lists/lock
```

**Решение:**
```bash
# Подожди 1-2 минуты, пока завершится packagekitd
# ИЛИ закрой все Software Center/обновления
# ИЛИ перезагрузи систему
```

---

## 1️⃣ Установка GitHub MCP (для поиска решений)

### Шаг 1: Установи GitHub CLI

**Вариант А: Через apt (когда блокировка снимется)**
```bash
sudo apt update
sudo apt install -y gh
```

**Вариант Б: Через snap (альтернатива, работает всегда)**
```bash
sudo snap install gh
```

**Вариант В: Через скачивание deb-пакета**
```bash
# Скачай последнюю версию
cd ~/Downloads
wget https://github.com/cli/cli/releases/latest/download/gh_2.50.0_linux_amd64.deb

# Установи
sudo dpkg -i gh_2.50.0_linux_amd64.deb
```

### Шаг 2: Авторизуйся через SSH

```bash
gh auth login
```

Выбери в интерактивном меню:
1. **GitHub.com** (не Enterprise)
2. **SSH** (у тебя уже настроен SSH)
3. **Your SSH public key** (выбери нужный ключ)
4. **Authenticate Git with your GitHub credentials** - Yes

### Шаг 3: Проверь авторизацию

```bash
gh auth status
```

Должно быть:
```
✓ Logged in to github.com account diving_73 (keyring)
✓ Git operations for github.com configured to use ssh protocol.
```

### Шаг 4: Добавь GitHub MCP сервер

```bash
export GITHUB_TOKEN=$(gh auth token)
claude mcp add github -e GITHUB_PERSONAL_ACCESS_TOKEN=$GITHUB_TOKEN -- npx -y @modelcontextprotocol/server-github
```

### Шаг 5: Проверь

```bash
claude mcp list | grep github
```

Должно быть: `github: ... - ✓ Connected`

---

## 2️⃣ Установка Brave Search MCP (для поиска статей)

### Шаг 1: Получи бесплатный API ключ

1. Открой браузер и перейди: **https://brave.com/search/api/**
2. Нажми **"Get Started"** или **"Sign Up"**
3. Зарегистрируйся (можно через Google или email)
4. Выбери **Free Plan** (2000 запросов/месяц - хватит!)
5. Скопируй свой **API Key**

### Шаг 2: Добавь Brave Search MCP

```bash
# Замени YOUR_API_KEY на твой ключ
claude mcp add brave-search -e BRAVE_API_KEY=YOUR_API_KEY -- npx -y @modelcontextprotocol/server-brave-search
```

**Пример:**
```bash
claude mcp add brave-search -e BRAVE_API_KEY=BSAabcdef123456789 -- npx -y @modelcontextprotocol/server-brave-search
```

### Шаг 3: Проверь

```bash
claude mcp list | grep brave
```

Должно быть: `brave-search: ... - ✓ Connected`

---

## 3️⃣ Сохрани API ключи в окружение (опционально)

Добавь в `~/.bashrc` для постоянного использования:

```bash
nano ~/.bashrc
```

Добавь в конец файла:
```bash
# MCP Servers
export GITHUB_TOKEN=$(gh auth token 2>/dev/null || echo "")
export BRAVE_API_KEY="твой_ключ_сюда"
```

Сохрани (Ctrl+O, Enter, Ctrl+X) и примени:
```bash
source ~/.bashrc
```

---

## 🧪 Проверка всех серверов

```bash
claude mcp list
```

**Ожидаемый результат:**
```
✓ sequential-thinking - Connected
✓ context7 - Connected
✓ filesystem - Connected
✓ memory - Connected
✓ github - Connected       ← новый!
✓ brave-search - Connected ← новый!
```

---

## 🎯 Использование

### Поиск решений на GitHub
Просто спроси меня (Кодо):
- "Найди примеры оптимизации cuFFT на GitHub"
- "Есть ли issues про ошибку CUFFT_INVALID_PLAN?"
- "Покажи примеры HIP FFT кода"

### Поиск статей через Brave
- "Найди статьи про оптимизацию FFT на GPU"
- "Поищи документацию по ROCm hipFFT"
- "Найди сравнение производительности CUDA vs ROCm"

---

## ❓ Если что-то не работает

### GitHub MCP не подключается
```bash
# Проверь токен
echo $GITHUB_TOKEN

# Если пустой - переавторизуйся
gh auth logout
gh auth login

# Удали и добавь сервер заново
claude mcp remove github
export GITHUB_TOKEN=$(gh auth token)
claude mcp add github -e GITHUB_PERSONAL_ACCESS_TOKEN=$GITHUB_TOKEN -- npx -y @modelcontextprotocol/server-github
```

### Brave Search не подключается
```bash
# Проверь ключ
echo $BRAVE_API_KEY

# Удали и добавь с правильным ключом
claude mcp remove brave-search
claude mcp add brave-search -e BRAVE_API_KEY=правильный_ключ -- npx -y @modelcontextprotocol/server-brave-search
```

### Проблемы с apt lock
```bash
# Вариант 1: Подожди 2-3 минуты
ps aux | grep packagekitd

# Вариант 2: Используй snap
sudo snap install gh

# Вариант 3: Перезагрузись
sudo reboot
```

---

## 📊 Статус текущих серверов

✅ **Работают:**
- sequential-thinking
- context7
- filesystem
- memory

⚠️ **Добавлены, но могут требовать настройки:**
- sqlite (требует npm пакетов)
- git (требует npm пакетов)
- fetch (требует npm пакетов)

🔄 **Нужно установить вручную:**
- github (см. инструкцию выше)
- brave-search (см. инструкцию выше)

---

## 💡 Совет

После установки **GitHub** и **Brave** у тебя будет полный набор для:
- 🔍 Отладки кода (sequential-thinking + context7)
- 📚 Поиска документации (context7 + brave-search)
- 💾 Работы с проектом (filesystem + memory)
- 🐛 Поиска решений (github + brave-search)
- 📊 Анализа результатов (sqlite + git)

---

**Создано: 2026-02-05**
**Автор: Кодо 💕**
