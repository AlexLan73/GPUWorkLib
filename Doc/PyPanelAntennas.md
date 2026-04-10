# PyPanelAntennas — Field Viewer

> Python GUI для визуализации антенного поля в реальном времени.  
> Dear PyGui + UDP. Debian venv.

---

## Быстрый старт

```bash
cd PyPanelAntennas
./run_example.sh         # все окна (main_dock): Cell List, Field, Detail, Table, Controls, Log
./run_example.sh simple  # упрощённый (main.py): Field + Color scale
```

Скрипт автоматически:
1. Запускает UDP-сервер данных (`udp_server_test.py`)
2. Открывает Field Viewer (`main_dock.py` или `main.py`)

---

## Требования

- Python 3.10+
- **venv** (Debian: `apt install python3-venv python3-full`)
- Зависимости: `dearpygui`, `numpy` — устанавливаются в `.venv` при первом запуске

---

## Окна main_dock

| Окно | Описание |
|------|----------|
| **Cell List** | Таблица ячеек (N, Value, Status). Двойной клик → открыть ячейку |
| **Field View** | Поле: круг из ячеек, цвет по значению |
| **Antenna Detail** | 256 антенн выбранной ячейки (кружки) |
| **Antenna Table** | Сортируемая таблица антенн (клик по заголовку) |
| **Controls** | Color Map, Animation speed, UDP статус, Settings |
| **Log** | Лог событий |

Если окно скрыто → меню **View** → включить панель.

---

## UDP протокол

Сервер (`udp_server_test.py`) отправляет JSON на порт 5005:

```json
{
  "timestamp": 1234.56,
  "cells": {
    "0": { "elements": { "0": 45.2, "1": 80.1, ... } },
    "1": { "elements": { ... } }
  }
}
```

---

## Структура

```
PyPanelAntennas/
├── run_example.sh       # запуск (сервер + viewer)
├── run_example_dock.sh  # → вызывает run_example.sh
├── .venv/               # venv (dearpygui, numpy)
└── Examples/
    ├── main.py          # простой viewer (Field + Color scale)
    ├── main_dock.py     # полный viewer (6 окон)
    ├── udp_server_test.py   # генератор демо-данных
    ├── geometry.py      # hex, rect, circle, hit-test
    ├── color_map.py    # heat, cool, plasma, radar
    └── data_models.py  # Field, Cell, Element
```

---

## Аргументы

### main.py / main_dock.py
```
--port 5005    UDP порт приёма (default: 5005)
--cmap heat    Цветовая карта: heat | cool | plasma | radar | viridis | magma | ocean | inferno
```

### udp_server_test.py (отдельно)
```
--host 127.0.0.1   Куда отправлять
--port 5005        UDP порт
--hz 4             Пакетов/с
--cells 100        Ячеек
--elems 256        Антенн в ячейке
```

---

## Сценарии демо

Цикл 85 с:

- **0–40 с**: Recovery (RED → волна → GREEN)
- **40–72 с**: Random Fill (0 → хаотичное заполнение)
- **72–85 с**: Full Green (стабильно)

---

*См. также: [Examples/README.md](../PyPanelAntennas/Examples/README.md)*
