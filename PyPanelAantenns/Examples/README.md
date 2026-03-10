# Field Viewer — Dear PyGui + UDP

Схематичная визуализация антенного поля с ячейками в реальном времени.

## Файлы

| Файл | Назначение |
|------|-----------|
| `main.py` | 🖥️ Главный viewer — Dear PyGui + UDP + анимация |
| `udp_server_test.py` | 📡 Тестовый генератор данных (UDP) |
| `geometry.py` | 📐 Геометрия: hex, rect, circle, hit-testing |
| `color_map.py` | 🎨 Цветовые карты + плавная анимация (lerp) |
| `data_models.py` | 📦 Структуры: Field, Cell, Element |

## Быстрый старт

**Через скрипт (из корня PyPanelAantenns):**
```bash
./run_example.sh         # все окна (main_dock): Cell List, Field, Detail, Table, Controls, Log
./run_example.sh simple  # упрощённый (main.py): Field + Color scale
```

Скрипт сам запускает UDP-сервер и viewer.

**Вручную (два терминала):**
```bash
# Терминал 1 — тестовый UDP генератор
python udp_server_test.py

# Терминал 2 — визуализатор
python main.py          # или main_dock.py (все окна)
```

**Окна main_dock:** Cell List | Field View | Antenna Detail | Antenna Table | Controls | Log.  
Если окно скрыто → меню **View** → включить нужную панель.

## Аргументы

### main.py
```
--port 5005        UDP порт для приёма данных (default: 5005)
--cmap heat        Цветовая карта: heat | cool | plasma | radar
--hz 60            Target FPS (default: 60)
```

### udp_server_test.py
```
--host 127.0.0.1   Адрес назначения
--port 5005        UDP порт
--hz 20            Пакетов в секунду
--cells 20         Количество ячеек
--elems 7          Элементов в ячейке
```

## UDP протокол

JSON пакет (UTF-8):
```json
{
  "timestamp": 1234.56,
  "cells": {
    "0": {
      "value": 75.3,
      "elements": {
        "0": 45.2,
        "1": 80.1,
        "2": 33.7
      }
    },
    "1": { "value": 42.0, "elements": {...} }
  }
}
```

- `value`: [0..100] — значение ячейки → цвет
- Ключи — строки или числа, оба варианта поддерживаются

## Управление

| Действие | Результат |
|---------|-----------|
| **ЛКМ по ячейке** | Открывает детальный вид элементов |
| **Закрыть детальное окно** | Кнопка ✕ окна |

## Цветовые карты

| Карта | Цвета | Назначение |
|-------|-------|-----------|
| `heat` | 🟢→🟡→🔴 | Тепловая карта (default) |
| `cool` | 🔵→💠→⚪ | Холодная |
| `plasma` | 🔵→🟣→🟡 | Plasma |
| `radar` | ⚫→🟢→⚪ | Стиль радара |

## Анимация

- **Плавные переходы** — экспоненциальный lerp (скорость 0.12/кадр)
- **Вспышка** — яркая подсветка при резком изменении (>5 ед/кадр)
- **GPU-рендеринг** — Dear PyGui использует DirectX/Vulkan

## Интеграция с реальным источником

Замените `_run()` в `UDPClient` на свой протокол:

```python
# UDP (уже реализовано)
raw, addr = sock.recvfrom(65536)
data = json.loads(raw)

# TCP — аналогично
data = json.loads(tcp_socket.recv(65536))

# ZeroMQ
data = json.loads(zmq_socket.recv())
```
