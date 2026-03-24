# TASK: GPU + Mellanox Topology Detector
**Статус**: 🟡 Phase 1 — Code Review DONE, ожидает тест на сервере
**Создан**: 2026-03-24
**Обновлён**: 2026-03-24
**Приоритет**: High
**Папки**:
- `Examples/GetGPU_and_Mellanox/` — standalone отладочный проект (Phase 1)
- `DrvGPU/services/` — интеграция (Phase 2, будущее)

---

## 🎯 Цель задачи

Создать **авто-детектор пар GPU + Mellanox NIC** по физическому расположению в PCIe слотах.
При старте DrvGPU на сервере (MI100 / 6049GP-TRT) — стабильно находить GPU по PCI-адресу
(не по HIP-индексу, который меняется при перезагрузке).

**Архитектурная цель**: маленький, легкий, независимый код.
Любой может скопировать `Examples/GetGPU_and_Mellanox/` и запустить.

---

## ✅ Code Review (2026-03-24, Кодо)

### Что сделано хорошо

| # | Что | Оценка |
|---|-----|--------|
| 1 | **Header-only** — весь детектор в одном файле `gpu_mellanox_detector.hpp` | ✅ Отлично |
| 2 | **Алгоритм** — sysfs slots → bus ranges → greedy pairing по nearest bus | ✅ Корректен |
| 3 | **Bifurcation** — правильно обрабатывает 2×GPU в одном слоте (2A/2B, 5A/5B) | ✅ Корректен |
| 4 | **ROCm** — `#ifdef ENABLE_ROCM` + `hipDeviceGetPCIBusId` для HIP index | ✅ Работает |
| 5 | **Device name tables** — MI100..MI325X, CX-4..CX-8 | ✅ Достаточно |
| 6 | **JSON export** — формат совместим с configGPU.json | ✅ Готово |
| 7 | **CMakeLists** — multi-version ROCm search (/opt/rocm-6.0..7.2) | ✅ Гибко |
| 8 | **README.md** — алгоритм, пример вывода, integration guide | ✅ Полный |

### Что нужно улучшить перед тестом на сервере

| # | Проблема | Серьёзность | Что делать |
|---|----------|-------------|-----------|
| 1 | **Нет verbose-режима** — при первом запуске на сервере нужно видеть что sysfs читается, какие слоты найдены, какие bus ranges, почему что-то пропущено | 🟡 Medium | Добавить `bool verbose = false` в `Detect()` и вывод диагностики |
| 2 | **`#ifdef` в параметрах функции** — `PairInSlot` имеет `#ifdef ENABLE_ROCM` прямо в списке параметров. Компилируется, но хрупко | 🟢 Low | Передавать `hip_map` всегда (пустой без ROCm), или передать через struct |
| 3 | **Нет NUMA node** — для будущей RDMA-оптимизации полезно знать NUMA-принадлежность GPU и NIC | 🟢 Low | Читать `/sys/bus/pci/devices/{addr}/numa_node`, добавить в `GpuNicPair` |
| 4 | **Нет GPU memory/compute info** — полезно для серверного инвентаря | 🟢 Low | Через `hipDeviceProperties` или sysfs |
| 5 | **main.cpp всегда сохраняет JSON** — без аргумента сохраняет `gpu_map.json` | 🟢 Low | Сохранять только при `--save` или при явном указании пути |

### Решение по таску: ТАСК ФАЙЛ описывает multi-file (include/, src/), а реальный код — header-only

**Вердикт**: header-only ЛУЧШЕ для цели "маленький, легкий, независимый". Обновляем таск под реальность.

---

## 📐 Фактическая архитектура (Phase 1)

```
Examples/GetGPU_and_Mellanox/        ← standalone, скопировал и запустил
├── gpu_mellanox_detector.hpp        ← ВСЁ в одном header-only файле (527 строк)
│   ├── namespace gpu_mellanox
│   │   ├── GpuNicPair              ← данные пары GPU+NIC
│   │   ├── ServerTopology          ← результат детекции
│   │   ├── detail::                ← internal helpers
│   │   │   ├── ReadLine/ReadHex/ReadDec  ← sysfs I/O
│   │   │   ├── PciDevice           ← распознанное PCIe устройство
│   │   │   ├── ReadAllDevices()    ← /sys/bus/pci/devices/
│   │   │   ├── ReadSlots()         ← /sys/bus/pci/slots/
│   │   │   ├── BuildHipPciMap()    ← #ifdef ENABLE_ROCM
│   │   │   └── PairInSlot()        ← greedy pairing
│   │   ├── Detect()                ← PUBLIC: главная функция
│   │   ├── PrintTable()            ← PUBLIC: вывод таблицы
│   │   └── SaveJson()              ← PUBLIC: JSON экспорт
├── main.cpp                         ← CLI: detect → print → save
├── CMakeLists.txt                   ← standalone build
└── README.md                        ← документация
```

---

## 🖥️ ИНСТРУКЦИЯ: Тестирование на сервере kc-vse-4-debian

### Предварительные требования
- Сервер: SuperServer 6049GP-TRT (Debian 12/13)
- 6× AMD Instinct MI100 + 6× Mellanox ConnectX-5
- ROCm 6.x или 7.x установлен (`/opt/rocm/` или `/opt/rocm-X.Y.Z/`)
- sudo доступ (для чтения sysfs bus numbers)

### Шаг 1 — Скопировать файлы на сервер

```bash
# Вариант A: если git clone доступен
git clone <repo_url>
cd GPUWorkLib/Examples/GetGPU_and_Mellanox

# Вариант B: скопировать только 4 файла
scp gpu_mellanox_detector.hpp main.cpp CMakeLists.txt README.md user@server:~/get_gpu/
```

### Шаг 2 — Собрать

```bash
cd ~/get_gpu   # или Examples/GetGPU_and_Mellanox
mkdir build && cd build

# С ROCm (рекомендуется — заполнит HIP column):
cmake .. -DCMAKE_PREFIX_PATH=/opt/rocm
make -j$(nproc)

# Без ROCm (если ROCm не установлен):
cmake ..
make -j$(nproc)
```

**Ожидаемый вывод cmake:**
```
[GetGPU_and_Mellanox] ROCm HIP found — ENABLE_ROCM=1, hip_id column active
```

### Шаг 3 — Запустить (НУЖЕН SUDO!)

```bash
sudo ./get_gpu_mellanox
```

### Шаг 4 — Проверить вывод

**Ожидаемый результат (6 GPU + 6 NIC):**

```
=== GPU + Mellanox Topology ===

  ID  Slot    GPU PCI             GPU Model                  NIC PCI             NIC Model             HIP
 ---  ------  ------------------  -------------------------  ------------------  --------------------  -----
   0  1       0000:1e:00.0        Instinct MI100             0000:20:00.0        ConnectX-5            2
   1  2A      0000:3f:00.0        Instinct MI100             0000:40:00.0        ConnectX-5            0
   2  2B      0000:45:00.0        Instinct MI100             0000:42:00.0        ConnectX-5            1
   3  5A      0000:8a:00.0        Instinct MI100             0000:8b:00.0        ConnectX-5            3
   4  5B      0000:90:00.0        Instinct MI100             0000:8d:00.0        ConnectX-5            4
   5  6       0000:b5:00.0        Instinct MI100             0000:b7:00.0        ConnectX-5            5
```

### Шаг 5 — Чеклист проверки

```
[ ] Все 6 GPU найдены
[ ] Все 6 NIC найдены и спарены
[ ] PCI адреса совпадают с `lspci | grep Instinct`
[ ] PCI адреса NIC совпадают с `lspci | grep Mellanox` (только .0 порты)
[ ] HIP column заполнена (не '-') — если собрано с ROCm
[ ] Slot labels: 1, 2A, 2B, 5A, 5B, 6
[ ] gpu_map.json создан и корректен
[ ] Повторный запуск даёт тот же результат
```

### Шаг 6 — Проверка перекрёстная (lspci)

```bash
# GPU PCI адреса:
lspci | grep Instinct
# Ожидаем: 1e:00.0, 3f:00.0, 45:00.0, 8a:00.0, 90:00.0, b5:00.0

# NIC PCI адреса:
lspci | grep Mellanox | grep "00.0"
# Ожидаем: 20:00.0, 40:00.0, 42:00.0, 8b:00.0, 8d:00.0, b7:00.0

# HIP перекрёстная проверка:
/opt/rocm/bin/rocm-smi --showbus
# Сравнить HIP idx ↔ PCI addr с нашей таблицей
```

### Шаг 7 — Если что-то не работает

| Симптом | Причина | Решение |
|---------|---------|---------|
| "Server topology NOT detected" | Нет sudo | `sudo ./get_gpu_mellanox` |
| "Server topology NOT detected" | Нет `/sys/bus/pci/slots/` | Проверить: `ls /sys/bus/pci/slots/` |
| 0 GPU найдено | sysfs `class` или `vendor` не читается | `cat /sys/bus/pci/devices/0000:1e:00.0/vendor` (должно быть `0x1002`) |
| HIP column всё '-' | Собрано без ROCm | Пересобрать: `cmake .. -DCMAKE_PREFIX_PATH=/opt/rocm` |
| Неправильные пары GPU↔NIC | Bus range не совпадает | Ручная проверка: `cat /sys/bus/pci/devices/0000:17:00.0/secondary_bus_number` |
| NIC не найдены | Mellanox vendor_id другой | `cat /sys/bus/pci/devices/0000:20:00.0/vendor` (должно быть `0x15b3`) |

### Шаг 8 — Сохранить результаты

```bash
# Сохранить JSON для проекта:
sudo ./get_gpu_mellanox gpu_map.json

# Скопировать обратно:
scp user@server:~/get_gpu/build/gpu_map.json .
# → Использовать как основу для configGPU.json
```

---

## 🗺️ Roadmap

### Phase 1 — Standalone (ТЕКУЩАЯ)
- [x] `gpu_mellanox_detector.hpp` — header-only детектор
- [x] `main.cpp` — CLI
- [x] `CMakeLists.txt` — standalone build
- [x] `README.md` — документация
- [x] **Code Review** (2026-03-24)
- [ ] Добавить verbose-режим для отладки
- [ ] Убрать `#ifdef` из параметров `PairInSlot`
- [ ] **Тест на сервере kc-vse-4-debian** (по инструкции выше)

### Phase 2 — DrvGPU integration (ПОСЛЕ тестирования)
- [ ] `DrvGPU/services/server_topology_service.hpp`
- [ ] Авто-переключение: `IsServerTopology()` → PCI mapping / old index mode
- [ ] Обновить `GPUManager::Initialize()`

### Phase 3 — Config + Backends (ПОСЛЕ Phase 2)
- [ ] `config_types.hpp` — добавить `gpu_pci`, `nic_pci` поля
- [ ] `rocm_backend.cpp` — `FindHipIndexByPci()`
- [ ] `opencl_backend.cpp` — `FindOpenCLDeviceByPci()`

---

## 📊 Целевая топология (kc-vse-4-debian)

| our_id | gpu_pci        | nic_pci        | GPU model    | NIC          | Slot |
|--------|----------------|----------------|--------------|--------------|------|
| 0      | 0000:1e:00.0   | 0000:20:00.0   | MI100        | CX-5         | 1    |
| 1      | 0000:3f:00.0   | 0000:40:00.0   | MI100        | CX-5         | 2A   |
| 2      | 0000:45:00.0   | 0000:42:00.0   | MI100        | CX-5         | 2B   |
| 3      | 0000:8a:00.0   | 0000:8b:00.0   | MI100        | CX-5         | 5A   |
| 4      | 0000:90:00.0   | 0000:8d:00.0   | MI100        | CX-5         | 5B   |
| 5      | 0000:b5:00.0   | 0000:b7:00.0   | MI100        | CX-5         | 6    |

---

## 🔑 Ключевые нюансы

1. **sudo** — нужен для `secondary_bus_number` / `subordinate_bus_number` на Debian 12/13
2. **ROCm 6.x vs 7.x** — API `hipDeviceProp_t.pciBusID` одинаков, совместимость OK
3. **Mellanox CX-5** — 2 порта (.0 и .1), детектор берёт только .0
4. **Bifurcation** — слоты 2 и 5 = 2×GPU + 2×NIC, greedy pairing по nearest bus
5. **Слот 8191** — виртуальный ESM, фильтруется
6. **PCI адрес стабилен** — не меняется при перезагрузке (в отличие от HIP index)

---

*Создан: 2026-03-24 | Сервер: kc-vse-4-debian (SuperServer 6049GP-TRT)*
*Автор: Кодо*
