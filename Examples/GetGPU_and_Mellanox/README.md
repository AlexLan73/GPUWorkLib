# GetGPU_and_Mellanox

Auto-detects AMD GPU + Mellanox NIC pairs by PCIe slot topology.
Outputs a stable ID mapping independent of reboot order.

**Target**: SuperServer 6049GP-TRT  (6× AMD Instinct MI100 + 6× Mellanox ConnectX-5)

---

## Files

```
gpu_mellanox_detector.hpp   ← entire detector (header-only, drop anywhere)
main.cpp                    ← CLI entry point
CMakeLists.txt
```

---

## Build

### With ROCm (recommended — fills HIP device index column)

```bash
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/opt/rocm
make -j$(nproc)
```

### Without ROCm (sysfs-only, hip_id column shows '-')

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

---

## Run

```bash
# Requires root to read /sys/bus/pci/devices/*/secondary_bus_number
sudo ./get_gpu_mellanox                     # detect and print
sudo ./get_gpu_mellanox -v                  # verbose: show sysfs reads, bus ranges
sudo ./get_gpu_mellanox --save              # detect, print, save gpu_map.json
sudo ./get_gpu_mellanox --save out.json     # save to custom path
sudo ./get_gpu_mellanox -v --save           # verbose + save
sudo ./get_gpu_mellanox -h                  # show help
```

### Expected output (kc-vse-4-debian)

```
Scanning PCIe topology...

=== GPU + Mellanox Topology ===

  ID  Slot    GPU PCI             GPU Model                  NIC PCI             NIC Model               HIP  NUMA
 ---  ------  ------------------  -------------------------  ------------------  --------------------  -----  ----
   0  1       0000:1e:00.0        Instinct MI100             0000:20:00.0        ConnectX-5                2  0
   1  2A      0000:3f:00.0        Instinct MI100             0000:40:00.0        ConnectX-5                0  0
   2  2B      0000:45:00.0        Instinct MI100             0000:42:00.0        ConnectX-5                1  0
   3  5A      0000:8a:00.0        Instinct MI100             0000:8b:00.0        ConnectX-5                3  1
   4  5B      0000:90:00.0        Instinct MI100             0000:8d:00.0        ConnectX-5                4  1
   5  6       0000:b5:00.0        Instinct MI100             0000:b7:00.0        ConnectX-5                5  1
 ---  ------  ------------------  -------------------------  ------------------  --------------------  -----  ----
 Total: 6 GPU+NIC pair(s)   Server topology: DETECTED
```

> **HIP column**: HIP device index returned by ROCm at this boot.
> Note: HIP indices may change after reboot. Our `ID` column is stable (sorted by GPU PCI address).
>
> **NUMA column**: NUMA node of the GPU (useful for CPU affinity and RDMA optimization).

### Verbose mode (`-v`)

Shows diagnostic output to stderr — useful for debugging on a new server:

```
[VERBOSE] Slot 1: root_port=0000:17:00.0 bus_range=[29..34]
[VERBOSE] Slot 2: root_port=0000:3a:00.0 bus_range=[58..72]
[VERBOSE] Found AMD GPU: 0000:1e:00.0 (Instinct MI100) bus=30
[VERBOSE] Found Mellanox NIC: 0000:20:00.0 (ConnectX-5) bus=32
[VERBOSE] HIP devices found: 6
[VERBOSE] Slot 1: 1 GPU + 1 NIC
[VERBOSE] Paired: id=0 slot=1 gpu=0000:1e:00.0 nic=0000:20:00.0 hip=2 numa=0
...
```

---

## Algorithm

```
1. /sys/bus/pci/slots/{N}/address
        → Root Port PCI ("0000:17:00.0")
        → secondary_bus_number / subordinate_bus_number
        → bus range [sec, sub] of this physical slot

2. /sys/bus/pci/devices/
        → vendor 0x1002 + class 0x03 → AMD GPU
        → vendor 0x15b3 + class 0x02 → Mellanox NIC (keep only port .0)

3. For each slot:
        → collect GPU and NIC devices with bus in [sec, sub]
        → greedy pair: GPU (sorted by bus) → nearest free NIC

4. (ENABLE_ROCM) hipDeviceGetPCIBusId → map PCI addr → HIP index

5. Assign our_id 0..N sorted by gpu_pci (stable across reboots)

6. Read /sys/bus/pci/devices/{addr}/numa_node for NUMA locality
```

### Why PCIe bus range, not physical_slot file?

On Debian 12/13 `physical_slot` in sysfs is not populated for GPU devices.
Instead we use the Root Port's `secondary_bus_number` / `subordinate_bus_number`
range — all devices with bus in that range physically belong to that slot.

### Bifurcation slots (slots 2 and 5 on 6049GP-TRT)

Slot 2 and 5 carry **two MI100 cards** via x16→2×x8 bifurcation.
Each MI100 has its own AMD PCIe switch, so there are 2 GPUs and 2 NICs
in the same slot's bus range. Greedy pairing by nearest bus correctly
assigns each GPU to its co-located NIC.

---

## Future integration into DrvGPU

```cpp
// DrvGPU/services/server_topology_service.hpp  (Phase 2)
#include "Examples/GetGPU_and_Mellanox/gpu_mellanox_detector.hpp"

// Rename namespace:  gpu_mellanox → drv_gpu_lib::topology

// DrvGPU/config/config_types.hpp  (Phase 3)
struct GPUConfigEntry {
    // ... existing fields ...
    std::string gpu_pci = "";   // "0000:1e:00.0" — locate GPU by stable PCI address
    std::string nic_pci = "";   // "0000:20:00.0" — for future RDMA integration
    int numa_node = -1;         // NUMA node for CPU affinity
};

// DrvGPU/backends/rocm/rocm_backend.cpp
// If gpu_pci set → hipDeviceGetPCIBusId → find matching hip_idx → hipSetDevice(hip_idx)
// Else           → hipSetDevice(entry.id)  [existing behaviour]
```

---

*Server: kc-vse-4-debian (SuperServer 6049GP-TRT, Debian 12, ROCm 7.2)*
