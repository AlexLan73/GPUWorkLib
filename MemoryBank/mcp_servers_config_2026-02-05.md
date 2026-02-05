# 🔧 MCP Servers Configuration Session
**Дата**: 2026-02-05
**Проект**: GPUWorkLib
**Цель**: Настройка MCP серверов для отладки, рефакторинга и работы с GPU

---

## ✅ Установленные и работающие серверы

### 1. sequential-thinking
- **Статус**: ✅ Работает
- **Назначение**: Решение сложных задач через цепочки рассуждений
- **Команда**: `npx -y @modelcontextprotocol/server-sequential-thinking`

### 2. context7
- **Статус**: ✅ Работает
- **Назначение**: Поиск документации CUDA, ROCm, C++, FFT
- **Команда**: `npx -y @upstash/context7-mcp@latest`

### 3. filesystem
- **Статус**: ✅ Работает
- **Путь**: `/home/alex/C++/GPUWorkLib`
- **Назначение**: Расширенная работа с файлами проекта
- **Команда**: `npx -y @modelcontextprotocol/server-filesystem /home/alex/C++/GPUWorkLib`

### 4. memory
- **Статус**: ✅ Работает
- **Назначение**: Постоянная память между сессиями
- **Интеграция**: С MemoryBank/
- **Команда**: `npx -y @modelcontextprotocol/server-memory`

---

## 📝 Рекомендации к установке

### Для отладки и поиска решений:

#### 1. GitHub MCP
```bash
sudo apt install -y gh
gh auth login  # SSH
export GITHUB_TOKEN=$(gh auth token)
claude mcp add github -e GITHUB_PERSONAL_ACCESS_TOKEN=$GITHUB_TOKEN -- npx -y @modelcontextprotocol/server-github
```

#### 2. Brave Search (API для поиска)
```bash
# Получить ключ: https://brave.com/search/api/
claude mcp add brave-search -e BRAVE_API_KEY=ключ -- npx -y @modelcontextprotocol/server-brave-search
```

### Для анализа результатов:

#### 3. SQLite
```bash
touch /home/alex/C++/GPUWorkLib/results.db
claude mcp add sqlite -- npx -y @modelcontextprotocol/server-sqlite --db-path /home/alex/C++/GPUWorkLib/results.db
```

#### 4. Git MCP
```bash
claude mcp add git -- npx -y @modelcontextprotocol/server-git
```

#### 5. Fetch MCP
```bash
claude mcp add fetch -- npx -y @modelcontextprotocol/server-fetch
```

---

## 🎯 Особенности проекта

### GPU конфигурация:
- **NVIDIA**: RTX 3060 (CUDA)
- **AMD**: MI100 (ROCm/HIP)

### Области применения MCP:
1. **Отладка CUDA/HIP кода**: sequential-thinking + context7
2. **Поиск оптимизаций**: brave-search + github
3. **Анализ результатов**: sqlite + memory
4. **Работа с документацией**: context7 + fetch

---

## 📚 Созданная документация

1. **Doc/MCP_SERVERS_SETUP.md** - Полная документация по настройке
2. **Doc/MCP_CHEATSHEET.md** - Быстрая шпаргалка
3. **Doc/install_mcp_servers.sh** - Скрипт автоматической установки

---

## 🔑 Ключевые команды

```bash
# Просмотр серверов
claude mcp list

# Установка всех рекомендуемых
cd /home/alex/C++/GPUWorkLib/Doc
./install_mcp_servers.sh

# Проверка работы
claude mcp list | grep "✓ Connected"
```

---

## 💡 Следующие шаги

1. ✅ Базовые серверы подключены
2. ⏳ Установить GitHub CLI и настроить
3. ⏳ Получить Brave API ключ
4. ⏳ Создать SQLite базу для результатов
5. ⏳ Настроить переменные окружения в ~/.bashrc

---

## 🎓 Полезные ссылки

- **MCP Docs**: https://modelcontextprotocol.io/
- **CUDA Toolkit**: https://docs.nvidia.com/cuda/
- **ROCm Docs**: https://rocm.docs.amd.com/
- **cuFFT**: https://docs.nvidia.com/cuda/cufft/
- **hipFFT**: https://rocm.docs.amd.com/projects/hipFFT/

---

*Настроено: 2026-02-05 by Кодо 💕*
