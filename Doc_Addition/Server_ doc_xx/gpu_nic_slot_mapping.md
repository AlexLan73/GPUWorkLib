# GPU + NIC → PCIe Slot Mapping
## SuperServer 6049GP-TRT | Debian 12/13 | ROCm

---

## Алгоритм рассуждения

### Проблема
GPU ID в коде (HIP/OpenCL) — это **порядковый индекс enumeration**, не физическое место.
При перезагрузке или замене карты индекс может измениться → код работает не с той картой.

### Исследование топологии

**Шаг 1**: Попробовали `physical_slot` в sysfs → не работает на Debian 12/13.

**Шаг 2**: Нашли `/sys/bus/pci/slots/` → есть, но адреса указывают на **Intel Root Port bridges**, не на GPU напрямую.

**Шаг 3**: Root Port имеет `secondary_bus_number` и `subordinate_bus_number` — диапазон всех bus номеров под ним. GPU bus попадает в этот диапазон → нашли привязку слот↔GPU.

**Шаг 4**: Обнаружили что на слотах 2 и 5 по **2 GPU + 2 NIC** — PCIe bifurcation (x16→2×x8). Каждый MI100 имеет **встроенный PCIe switch** (AMD 14a0→14a1). Слот не уникален для карты при bifurcation.

**Вывод**: Единственный надёжный стабильный идентификатор — **PCI адрес** (например `0000:1e:00.0`). Он определяется физическим местом в топологии, не картой.

### Топология сервера (реальные данные)

```
CPU (Intel Xeon Scalable / Sky Lake-E)
├── Слот 1  (Root Port 17:00) → AMD switch → MI100 (1e:00) + Mellanox (20:00)
├── Слот 2  (Root Port 3a:00) → bifurcation x16→2×x8
│   ├── Path A → AMD switch → MI100 (3f:00) + Mellanox (40:00)
│   └── Path B → AMD switch → MI100 (45:00) + Mellanox (42:00)
├── Слот 3  (Root Port 5d:00) → RAID LSI MegaRAID
├── Слот 4  (Root Port 5d:02) → Intel X722 NIC (встроенная)
├── Слот 5  (Root Port 85:00) → bifurcation x16→2×x8
│   ├── Path A → AMD switch → MI100 (8a:00) + Mellanox (8b:00)
│   └── Path B → AMD switch → MI100 (90:00) + Mellanox (8d:00)
└── Слот 6  (Root Port ae:00) → AMD switch → MI100 (b5:00) + Mellanox (b7:00)
```

---

## Команды диагностики

### Показать все GPU и NIC по физическим слотам
```bash
echo "=== GPU и NIC по физическим слотам ===" && \
for slot in $(ls /sys/bus/pci/slots/ | grep -v 8191 | sort -V); do
    addr=$(cat /sys/bus/pci/slots/$slot/address 2>/dev/null)
    [ -z "$addr" ] && continue
    sysfs="/sys/bus/pci/devices/${addr}.0"
    [ ! -d "$sysfs" ] && continue
    sec=$(cat "$sysfs/secondary_bus_number" 2>/dev/null)
    sub=$(cat "$sysfs/subordinate_bus_number" 2>/dev/null)
    [ -z "$sec" ] && continue
    result=$(lspci -nn | grep -v "PCI bridge" | while read line; do
        bus_dec=$((16#$(echo "$line" | cut -d: -f1)))
        [ "$bus_dec" -ge "$sec" ] && [ "$bus_dec" -le "$sub" ] && echo "  $line"
    done)
    [ -n "$result" ] && printf "\n=== Слот %s (%s) ===\n%s\n" "$slot" "$addr" "$result"
done
```

### Показать HIP устройства с PCI адресами
```bash
/opt/rocm/bin/rocm-smi --showbus
```

### Дерево PCI (топология)
```bash
lspci -tv | grep -v "Sky Lake-E CHA\|Sky Lake-E Cache\|Sky Lake-E PCU"
```

---

## Итоговая карта сервера (kc-vse-4-debian)

| GPU id | PCI адрес GPU  | PCI адрес NIC  | Модель GPU    | NIC           | Слот  |
|--------|----------------|----------------|---------------|---------------|-------|
| 0      | 0000:1e:00.0   | 0000:20:00.0   | MI100         | Mellanox CX-5 | 1     |
| 1      | 0000:3f:00.0   | 0000:40:00.0   | MI100         | Mellanox CX-5 | 2A    |
| 2      | 0000:45:00.0   | 0000:42:00.0   | MI100         | Mellanox CX-5 | 2B    |
| 3      | 0000:8a:00.0   | 0000:8b:00.0   | MI100         | Mellanox CX-5 | 5A    |
| 4      | 0000:90:00.0   | 0000:8d:00.0   | MI100         | Mellanox CX-5 | 5B    |
| 5      | 0000:b5:00.0   | 0000:b7:00.0   | MI100         | Mellanox CX-5 | 6     |

> ⚠️ HIP id (0..5) может меняться при перезагрузке. PCI адрес — стабилен.

---

## Python: утилита формирования таблицы соответствия

```python
#!/usr/bin/env python3
"""
gpu_slot_map.py — автоматическое построение таблицы GPU+NIC по слотам
Запуск: sudo python3 gpu_slot_map.py
Требует: ROCm, lspci

Выводит таблицу и сохраняет в gpu_map.json
"""

import subprocess
import json
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class DeviceInfo:
    pci_addr: str
    description: str
    device_type: str  # "GPU" | "NIC" | "OTHER"


@dataclass
class SlotEntry:
    pci_slot: str
    root_port_pci: str
    bus_range: tuple
    gpu: Optional[DeviceInfo] = None
    nics: list = field(default_factory=list)


def run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return ""


def get_all_pci_devices() -> dict[str, str]:
    """Возвращает dict: pci_addr → описание"""
    out = run(["lspci", "-nn"])
    devices = {}
    for line in out.splitlines():
        parts = line.split(" ", 1)
        if len(parts) == 2:
            devices["0000:" + parts[0]] = parts[1]
    return devices


def classify_device(description: str) -> str:
    desc = description.lower()
    if any(x in desc for x in ["display", "vga", "3d controller", "amd/ati"]):
        return "GPU"
    if any(x in desc for x in ["ethernet", "network", "infiniband", "mellanox", "connectx"]):
        return "NIC"
    if "pci bridge" in desc:
        return "BRIDGE"
    return "OTHER"


def get_bus_range(sysfs_path: str) -> Optional[tuple[int, int]]:
    """Читает secondary/subordinate bus range из sysfs"""
    try:
        sec = int(open(f"{sysfs_path}/secondary_bus_number").read().strip())
        sub = int(open(f"{sysfs_path}/subordinate_bus_number").read().strip())
        return (sec, sub)
    except Exception:
        return None


def get_slots() -> list[SlotEntry]:
    slots_dir = "/sys/bus/pci/slots"
    entries = []

    for slot_name in sorted(os.listdir(slots_dir)):
        if "8191" in slot_name:
            continue
        addr_file = f"{slots_dir}/{slot_name}/address"
        try:
            addr = open(addr_file).read().strip()  # "0000:17:00"
        except Exception:
            continue

        sysfs = f"/sys/bus/pci/devices/{addr}.0"
        if not os.path.isdir(sysfs):
            continue

        bus_range = get_bus_range(sysfs)
        if not bus_range:
            continue

        entries.append(SlotEntry(
            pci_slot=slot_name,
            root_port_pci=f"{addr}.0",
            bus_range=bus_range,
        ))

    return entries


def get_hip_pci_map() -> dict[str, int]:
    """Возвращает dict: pci_addr → HIP device index через rocm-smi"""
    out = run(["/opt/rocm/bin/rocm-smi", "--showbus"])
    hip_map = {}
    # Парсим строки вида: GPU[0]  : PCI Bus: 0000:1E:00.0
    for line in out.splitlines():
        m = re.search(r"GPU\[(\d+)\].*?:\s*([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9])", line)
        if m:
            hip_idx = int(m.group(1))
            pci = m.group(2).lower()
            hip_map[pci] = hip_idx
    return hip_map


def build_map() -> list[dict]:
    all_devices = get_all_pci_devices()
    slots = get_slots()
    hip_map = get_hip_pci_map()

    result = []

    for slot in slots:
        sec, sub = slot.bus_range
        gpus_in_slot = []
        nics_in_slot = []

        for pci_addr, desc in all_devices.items():
            # Извлекаем bus номер (hex)
            try:
                bus_hex = pci_addr.split(":")[1]
                bus_dec = int(bus_hex, 16)
            except Exception:
                continue

            if sec <= bus_dec <= sub:
                device_type = classify_device(desc)
                if device_type == "GPU":
                    gpus_in_slot.append({"pci": pci_addr, "desc": desc})
                elif device_type == "NIC":
                    nics_in_slot.append({"pci": pci_addr, "desc": desc})

        if not gpus_in_slot:
            continue

        # Сортируем по bus (для правильного порядка пар)
        gpus_in_slot.sort(key=lambda x: x["pci"])
        nics_in_slot.sort(key=lambda x: x["pci"])

        # Спариваем GPU↔NIC по порядку bus номеров
        for i, gpu in enumerate(gpus_in_slot):
            gpu_pci = gpu["pci"]
            # NIC с ближайшим bus после GPU
            gpu_bus = int(gpu_pci.split(":")[1], 16)
            paired_nic = None
            min_dist = 999
            for nic in nics_in_slot:
                nic_bus = int(nic["pci"].split(":")[1], 16)
                dist = abs(nic_bus - gpu_bus)
                if dist < min_dist:
                    min_dist = dist
                    paired_nic = nic

            slot_label = slot.pci_slot if len(gpus_in_slot) == 1 \
                else f"{slot.pci_slot}{'AB'[i]}"

            hip_idx = hip_map.get(gpu_pci, -1)

            entry = {
                "hip_id":     hip_idx,
                "pci_slot":   slot_label,
                "gpu_pci":    gpu_pci,
                "gpu_model":  gpu["desc"].split("[")[0].strip(),
                "nic_pci":    paired_nic["pci"] if paired_nic else "",
                "nic_model":  paired_nic["desc"].split("[")[0].strip() if paired_nic else "",
            }
            result.append(entry)

    result.sort(key=lambda x: x["gpu_pci"])
    return result


def print_table(entries: list[dict]):
    print(f"\n{'HIP':>4} {'Слот':<6} {'GPU PCI':<18} {'GPU модель':<35} {'NIC PCI':<18} {'NIC модель'}")
    print("-" * 110)
    for e in entries:
        print(f"{e['hip_id']:>4} {e['pci_slot']:<6} {e['gpu_pci']:<18} {e['gpu_model']:<35} {e['nic_pci']:<18} {e['nic_model']}")


def save_json(entries: list[dict], path: str = "gpu_map.json"):
    # Формируем в формате configGPU.json
    config = {
        "description": "GPU+NIC PCI mapping — auto-generated",
        "version": "1.1",
        "gpus": [
            {
                "id":          i,
                "name":        f"GPU_{e['pci_slot']}",
                "gpu_pci":     e["gpu_pci"],
                "nic_pci":     e["nic_pci"],
                "is_active":   True,
                "is_console":  i == 0,
                "is_logger":   True,
                "is_prof":     True,
                "max_memory_percent": 70,
                "log_level":   "INFO"
            }
            for i, e in enumerate(entries)
        ]
    }
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n✓ Сохранено в {path}")


if __name__ == "__main__":
    print("Сканирование PCIe топологии...")
    entries = build_map()
    print_table(entries)
    save_json(entries, "gpu_map.json")
    print("\nГотово! Скопируй gpu_map.json → configGPU.json в проект.")
```

---

## План работ и задачи

### Task 1 — Запустить утилиту и сформировать таблицу
- [ ] `sudo python3 gpu_slot_map.py` на kc-vse-4-debian
- [ ] Проверить вывод — все 6 GPU найдены, пары GPU↔NIC корректны
- [ ] Сохранить `gpu_map.json` → это основа для `configGPU.json`

### Task 2 — Обновить схему configGPU.json
- [ ] Добавить поля `gpu_pci` и `nic_pci` в `GPUConfigEntry` (`config_types.hpp`)
- [ ] Обновить `gpu_config.cpp` — парсинг новых полей
- [ ] Обратная совместимость — если `gpu_pci` пусто, использовать старый `id` как индекс

### Task 3 — Обновить DrvGPU ROCm backend
- [ ] Добавить `FindHipIndexByPci(const std::string& pci)` в `rocm_backend.cpp`
- [ ] При инициализации: если `gpu_pci` задан → искать по PCI, иначе по индексу
- [ ] Логировать: `"GPU found: hip_idx=2 pci=0000:1e:00.0 slot=SLOT1"`

### Task 4 — Обновить DrvGPU OpenCL backend
- [ ] Аналогично ROCm: `FindOpenCLDeviceByPci()` через `CL_DEVICE_TOPOLOGY_AMD`
- [ ] Приоритет: если `gpu_pci` задан в конфиге — использовать его

### Task 5 — Тестирование на kc-vse-4-debian
- [ ] Запустить с новым configGPU.json
- [ ] Проверить: каждый `gpu_id=N` в логах соответствует правильному физическому слоту
- [ ] Перезагрузить сервер → убедиться что маппинг не изменился
- [ ] Заменить одну карту (если возможно) → проверить что конфиг не нужно менять

---

*Дата: 2026-03-12 | Сервер: kc-vse-4-debian*
