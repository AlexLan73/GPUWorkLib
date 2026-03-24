# Определение PCIe слота программно — SuperServer 6049GP-TRT

## 1. `lspci` — быстрый способ

```bash
lspci -v | grep -i "VGA\|GPU\|Display\|AMD\|NVIDIA\|Radeon"
```

С адресами:
```bash
lspci -nn | grep -i "AMD\|Radeon\|Display"
# Пример: 0b:00.0 Display controller [0380]: Advanced Micro Devices...
# 0b:00.0 = bus:device.function
```

## 2. Физический номер слота через `lspci`

```bash
lspci -v | grep -A 20 "VGA\|Display" | grep -i "slot\|physical"
# или
lspci -vmm | grep -A 15 "Display"
```

Ищи поле `PhySlot:` — это физический номер слота.

## 3. sysfs — надёжный способ для программного доступа

```bash
# Все GPU со слотами:
for dev in /sys/bus/pci/devices/*/; do
    slot=$(cat "$dev/physical_slot" 2>/dev/null)
    [ -n "$slot" ] && echo "$dev -> slot $slot"
done
```

Или напрямую по адресу:
```bash
cat /sys/bus/pci/devices/0000:0b:00.0/physical_slot
```

## 4. `dmidecode` — по данным BIOS

```bash
sudo dmidecode -t slot
# Показывает все слоты: ID, ширину (x8/x16), статус (In Use / Available)

# Быстро — только занятые:
sudo dmidecode -t slot | grep -A5 "In Use"
```

## 5. Программное определение (Python + C++)

### Python
```python
import subprocess

def get_gpu_pci_slot(pci_addr: str) -> str:
    """pci_addr = '0000:0b:00.0'"""
    path = f"/sys/bus/pci/devices/{pci_addr}/physical_slot"
    try:
        with open(path) as f:
            return f.read().strip()
    except FileNotFoundError:
        return "unknown"

# Получить все GPU адреса:
out = subprocess.check_output(["lspci", "-nn"], text=True)
for line in out.splitlines():
    if "Display" in line or "VGA" in line:
        addr = line.split()[0]
        print(f"GPU {addr} → slot {get_gpu_pci_slot('0000:' + addr)}")
```

### C++ (через sysfs)
```cpp
#include <fstream>
#include <string>

std::string GetPciSlot(const std::string& pci_addr) {
    // pci_addr = "0000:0b:00.0"
    std::string path = "/sys/bus/pci/devices/" + pci_addr + "/physical_slot";
    std::ifstream f(path);
    if (!f) return "unknown";
    std::string slot;
    std::getline(f, slot);
    return slot;
}
```

---

## 6. Шаг 2 (альтернатива) — ROCm / HIP вариант

```cpp
// Требует: #include <hip/hip_runtime.h>

#include <hip/hip_runtime.h>
#include <fstream>
#include <string>

// Получить PCI адрес GPU по HIP device index
std::string GetPciAddressROCm(int hip_device_index) {
    hipDeviceProp_t props;
    hipError_t err = hipGetDeviceProperties(&props, hip_device_index);
    if (err != hipSuccess) return "";

    // props содержит: pciBusID, pciDeviceID, pciDomainID
    char buf[32];
    snprintf(buf, sizeof(buf), "%04x:%02x:%02x.0",
        props.pciDomainID,
        props.pciBusID,
        props.pciDeviceID);
    return buf;
}

// Получить физический номер PCIe слота из sysfs
int GetPhysicalSlotROCm(int hip_device_index) {
    std::string pci = GetPciAddressROCm(hip_device_index);
    if (pci.empty()) return -1;

    std::string path = "/sys/bus/pci/devices/" + pci + "/physical_slot";
    std::ifstream f(path);
    if (!f) return -1;
    int slot; f >> slot;
    return slot;
}

// Найти HIP device index по номеру физического слота
int FindHipDeviceBySlot(int target_slot) {
    int device_count = 0;
    hipGetDeviceCount(&device_count);

    for (int i = 0; i < device_count; i++) {
        if (GetPhysicalSlotROCm(i) == target_slot)
            return i;  // это и есть нужный HIP device index
    }
    return -1;  // слот не найден
}
```

### Использование

```cpp
// При инициализации DrvGPU ROCm backend:
int configured_slot = gpu_config_entry.pci_slot;  // из configGPU.json
int hip_idx = FindHipDeviceBySlot(configured_slot);

if (hip_idx < 0) {
    // GPU в слоте не найден — карту вынули?
    LOG_ERROR << "GPU not found in PCIe slot " << configured_slot;
    return false;
}

hipSetDevice(hip_idx);  // теперь работаем с правильной физической картой
```

### Сравнение OpenCL vs ROCm подходов

| | OpenCL (AMD ext) | ROCm / HIP |
|---|---|---|
| API | `clGetDeviceInfo` + `CL_DEVICE_TOPOLOGY_AMD` | `hipGetDeviceProperties` |
| Структура | `cl_device_topology_amd.pcie` | `hipDeviceProp_t.pciBusID` |
| Надёжность | Только AMD | Только AMD (ROCm) |
| Заголовок | `<CL/cl_ext.h>` | `<hip/hip_runtime.h>` |
| В нашем проекте | OpenCL backend | ROCm backend |

> **Итог**: оба варианта дают одинаковый PCI адрес → одинаковый `physical_slot` из sysfs. Выбор зависит от бэкенда.

---

## Специфика 6049GP-TRT

- **8 GPU слотов**: 4 × PCIe x16 + 4 × PCIe x8
- Физические слоты пронумерованы **1–8** в BIOS
- `dmidecode -t slot` даёт точное соответствие: слот ↔ PCI bus address

---

*Дата: 2026-03-12*
