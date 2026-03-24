/**
 * @file gpu_mellanox_detector.hpp
 * @brief Auto-detector: AMD GPU + Mellanox NIC pairs by PCIe slot topology
 *
 * Header-only. No external dependencies beyond C++17 and (optionally) ROCm HIP.
 *
 * HOW IT WORKS:
 *   1. Read PCIe slot → bus range mapping from /sys/bus/pci/slots/
 *   2. Enumerate all GPU (AMD) and NIC (Mellanox) devices from /sys/bus/pci/devices/
 *   3. For each slot: collect devices in its bus range, pair GPU↔NIC by nearest bus number
 *   4. (if ENABLE_ROCM) map each GPU PCI address → HIP device index via hipDeviceGetPCIBusId
 *
 * REQUIREMENTS:
 *   - Linux, /sys/bus/pci/ available
 *   - Reading secondary_bus_number may require root (Debian 12/13)
 *   - C++17 (std::filesystem)
 *   - ROCm HIP (optional, for hip_id field)
 *
 * FUTURE INTEGRATION:
 *   #include "gpu_mellanox_detector.hpp"  ← drop into DrvGPU/services/
 *   namespace gpu_mellanox → rename to drv_gpu_lib::topology
 */

#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <unordered_map>
#include <set>
#include <cstdint>
#include <cstdio>
#include <cstdlib>  // abs
#include <climits>  // INT_MAX
#include <iostream>

#ifdef ENABLE_ROCM
#  include <hip/hip_runtime.h>
#endif

namespace gpu_mellanox {

// ═════════════════════════════════════════════════════════════════════════════
// Public data types
// ═════════════════════════════════════════════════════════════════════════════

/**
 * @brief One GPU + paired Mellanox NIC
 *
 * our_id: stable sequential index assigned by PCI address order —
 *         does NOT change across reboots (unlike hip_id)
 * hip_id: HIP device index as returned by ROCm at this boot (-1 if ENABLE_ROCM not set)
 */
struct GpuNicPair {
  int         our_id    = -1;   ///< our stable ID (0-based, ordered by gpu_pci)
  int         hip_id    = -1;   ///< ROCm HIP index (-1 if unavailable)
  std::string gpu_pci;          ///< "0000:1e:00.0"
  std::string nic_pci;          ///< "0000:20:00.0"  (empty if no NIC found)
  std::string gpu_name;         ///< "Instinct MI100"
  std::string nic_name;         ///< "ConnectX-5"
  std::string slot_label;       ///< "1"  "2A"  "2B"  "5A"  "5B"  "6"
  int         numa_node = -1;   ///< NUMA node of the GPU (-1 if unknown)
};

/**
 * @brief Full detected topology
 *
 * is_server_topology = true when AMD GPUs and Mellanox NICs share PCIe slots,
 * i.e. when the server-side detection path is meaningful.
 */
struct ServerTopology {
  bool                    is_server_topology = false;
  std::vector<GpuNicPair> pairs;             ///< sorted by gpu_pci
};

/// Verbose flag — set to true for diagnostic output during Detect()
inline bool& Verbose() {
  static bool v = false;
  return v;
}
inline void SetVerbose(bool v) { Verbose() = v; }


// ═════════════════════════════════════════════════════════════════════════════
// Internal helpers (not part of public API)
// ═════════════════════════════════════════════════════════════════════════════

namespace detail {

// ─── sysfs helpers ────────────────────────────────────────────────────────

/// Read first line of a file; returns "" on error (no exception)
inline std::string ReadLine(const std::string& path) {
  std::ifstream f(path);
  if (!f) return {};
  std::string s;
  std::getline(f, s);
  return s;
}

/// Read hex value from sysfs file, e.g. "0x1002\n" → 0x1002
inline uint32_t ReadHex(const std::string& path) {
  const auto s = ReadLine(path);
  if (s.empty()) return 0;
  try { return static_cast<uint32_t>(std::stoul(s, nullptr, 16)); }
  catch (...) { return 0; }
}

/// Read decimal integer, e.g. secondary_bus_number "29\n" → 29
inline int ReadDec(const std::string& path) {
  const auto s = ReadLine(path);
  if (s.empty()) return -1;
  try { return std::stoi(s); }
  catch (...) { return -1; }
}

/// Lowercase a PCI address string  ("0000:1E:00.0" → "0000:1e:00.0")
inline std::string LowerPci(std::string s) {
  for (char& c : s)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

/// Extract decimal bus number from "0000:1e:00.0" → 0x1e = 30
inline int BusFromAddr(const std::string& addr) {
  unsigned domain, bus, dev, func;
  if (sscanf(addr.c_str(), "%x:%x:%x.%x", &domain, &bus, &dev, &func) == 4)
    return static_cast<int>(bus);
  return -1;
}

// ─── PCI device ───────────────────────────────────────────────────────────

struct PciDevice {
  std::string addr;           ///< "0000:1e:00.0" (lowercase)
  int         bus     = -1;   ///< decimal bus number
  uint16_t    vendor  = 0;    ///< 0x1002 AMD, 0x15b3 Mellanox
  uint16_t    device  = 0;    ///< device id
  uint8_t     cls     = 0;    ///< base class: 0x03 display, 0x02 network
  uint8_t     subcls  = 0;    ///< sub class
};

inline bool IsAmdGpu(const PciDevice& d) {
  // vendor AMD + base class 0x03 (Display / 3D / Display Other)
  return d.vendor == 0x1002 && d.cls == 0x03;
}

inline bool IsMellanoxNic(const PciDevice& d) {
  // vendor Mellanox + base class 0x02 (Network)
  return d.vendor == 0x15b3 && d.cls == 0x02;
}

/// Return only .0 (primary) port of a multi-port NIC
inline bool IsPrimaryPort(const std::string& addr) {
  // "0000:20:00.0" → last char '0'  (function 0)
  // "0000:20:00.1" → last char '1'  (function 1 = second port)
  return !addr.empty() && addr.back() == '0';
}

// ─── Device name tables ───────────────────────────────────────────────────

inline std::string AmdGpuName(uint16_t dev_id) {
  // AMD Instinct MI-series + Radeon RX (expand as needed)
  switch (dev_id) {
    case 0x738c: return "Instinct MI100";
    case 0x7388: return "Instinct MI100";      // alternate PF
    case 0x740c: return "Instinct MI250X";
    case 0x7408: return "Instinct MI250";
    case 0x74a0: return "Instinct MI300X";
    case 0x74a1: return "Instinct MI300A";
    case 0x74b5: return "Instinct MI325X";
    case 0x1478: return "Radeon RX 9070 XT";
    case 0x1479: return "Radeon RX 9070";
    case 0x73bf: return "Radeon RX 6900 XT";
    default:     return "AMD GPU";
  }
}

inline std::string MellanoxNicName(uint16_t dev_id) {
  switch (dev_id) {
    case 0x1013: return "ConnectX-4";
    case 0x1015: return "ConnectX-4 Lx";
    case 0x1017: return "ConnectX-5";
    case 0x1019: return "ConnectX-5 Ex";
    case 0x101b: return "ConnectX-6";
    case 0x101d: return "ConnectX-6 Dx";
    case 0x101f: return "ConnectX-6 Lx";
    case 0x1021: return "ConnectX-7";
    case 0x1023: return "ConnectX-8";
    default:     return "Mellanox NIC";
  }
}

// ─── PCIe device enumeration ─────────────────────────────────────────────

/// Read NUMA node for a PCI device (-1 if unavailable)
inline int ReadNumaNode(const std::string& pci_addr) {
  return ReadDec("/sys/bus/pci/devices/" + pci_addr + "/numa_node");
}

/// Read all AMD GPU and Mellanox NIC devices from /sys/bus/pci/devices/
inline std::vector<PciDevice> ReadAllDevices() {
  std::vector<PciDevice> result;
  namespace fs = std::filesystem;
  const std::string base = "/sys/bus/pci/devices/";
  if (!fs::exists(base)) {
    if (Verbose()) std::cerr << "[VERBOSE] /sys/bus/pci/devices/ not found!\n";
    return result;
  }

  for (const auto& entry : fs::directory_iterator(base)) {
    const std::string addr = LowerPci(entry.path().filename().string());
    const std::string dp   = base + addr + "/";

    const uint32_t class_raw = ReadHex(dp + "class");
    const uint8_t  base_cls  = static_cast<uint8_t>((class_raw >> 16) & 0xFF);

    // Fast pre-filter: only display (0x03) or network (0x02) devices
    if (base_cls != 0x03 && base_cls != 0x02) continue;

    PciDevice d;
    d.addr   = addr;
    d.bus    = BusFromAddr(addr);
    d.vendor = static_cast<uint16_t>(ReadHex(dp + "vendor"));
    d.device = static_cast<uint16_t>(ReadHex(dp + "device"));
    d.cls    = base_cls;
    d.subcls = static_cast<uint8_t>((class_raw >> 8) & 0xFF);

    if (IsAmdGpu(d)) {
      if (Verbose())
        std::cerr << "[VERBOSE] Found AMD GPU: " << d.addr
                  << " (" << AmdGpuName(d.device) << ") bus=" << d.bus << "\n";
      result.push_back(d);
    } else if (IsMellanoxNic(d)) {
      if (Verbose())
        std::cerr << "[VERBOSE] Found Mellanox NIC: " << d.addr
                  << " (" << MellanoxNicName(d.device) << ") bus=" << d.bus << "\n";
      result.push_back(d);
    }
  }
  if (Verbose()) std::cerr << "[VERBOSE] Total GPU+NIC devices: " << result.size() << "\n";
  return result;
}

// ─── PCIe slot enumeration ───────────────────────────────────────────────

struct SlotInfo {
  std::string slot_name;    ///< "1", "2", "5", "6"
  std::string root_port;    ///< "0000:17:00.0"
  int         sec_bus = -1; ///< secondary bus  (decimal)
  int         sub_bus = -1; ///< subordinate bus (decimal)
};

/**
 * Read physical PCIe slots from /sys/bus/pci/slots/
 *
 * Each slot entry: address → Root Port PCI addr → secondary/subordinate bus range.
 * All PCIe devices with bus in [sec_bus, sub_bus] are physically in that slot.
 *
 * NOTE: reading secondary_bus_number may require root on Debian 12/13.
 */
inline std::vector<SlotInfo> ReadSlots() {
  std::vector<SlotInfo> result;
  namespace fs = std::filesystem;
  const std::string base = "/sys/bus/pci/slots/";
  if (!fs::exists(base)) {
    if (Verbose()) std::cerr << "[VERBOSE] /sys/bus/pci/slots/ not found! (need root?)\n";
    return result;
  }

  for (const auto& entry : fs::directory_iterator(base)) {
    const std::string slot_name = entry.path().filename().string();

    // Slot "8191" is a virtual ESM slot, skip it
    if (slot_name.find("8191") != std::string::npos) {
      if (Verbose()) std::cerr << "[VERBOSE] Skipping virtual slot: " << slot_name << "\n";
      continue;
    }

    // address file: "0000:17:00" (Root Port addr WITHOUT .0)
    const auto addr = ReadLine(entry.path().string() + "/address");
    if (addr.empty()) {
      if (Verbose()) std::cerr << "[VERBOSE] Slot " << slot_name << ": no address file\n";
      continue;
    }

    const std::string root_port = LowerPci(addr) + ".0";
    const std::string dp        = "/sys/bus/pci/devices/" + root_port + "/";

    SlotInfo s;
    s.slot_name = slot_name;
    s.root_port = root_port;
    s.sec_bus   = ReadDec(dp + "secondary_bus_number");
    s.sub_bus   = ReadDec(dp + "subordinate_bus_number");

    if (s.sec_bus < 0 || s.sub_bus < 0) {
      if (Verbose())
        std::cerr << "[VERBOSE] Slot " << slot_name << " (" << root_port
                  << "): can't read bus range (sec=" << s.sec_bus
                  << " sub=" << s.sub_bus << ") — need root?\n";
      continue;
    }

    if (Verbose())
      std::cerr << "[VERBOSE] Slot " << slot_name << ": root_port=" << root_port
                << " bus_range=[" << s.sec_bus << ".." << s.sub_bus << "]\n";
    result.push_back(s);
  }

  // Sort by sec_bus for deterministic GPU ID assignment
  std::sort(result.begin(), result.end(),
            [](const SlotInfo& a, const SlotInfo& b) {
              return a.sec_bus < b.sec_bus;
            });

  if (Verbose()) std::cerr << "[VERBOSE] Valid PCIe slots: " << result.size() << "\n";
  return result;
}

// ─── ROCm HIP index lookup ────────────────────────────────────────────────

#ifdef ENABLE_ROCM
/// Build map: lowercase pci_addr → HIP device index
/// hipDeviceGetPCIBusId returns UPPERCASE hex, we normalise to lowercase.
inline std::unordered_map<std::string, int> BuildHipPciMap() {
  std::unordered_map<std::string, int> hip_map;
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess || count == 0) return hip_map;

  for (int i = 0; i < count; ++i) {
    char bus_id[64] = {};
    if (hipDeviceGetPCIBusId(bus_id, sizeof(bus_id), i) == hipSuccess)
      hip_map[LowerPci(bus_id)] = i;
  }
  return hip_map;
}
#endif

// ─── Greedy GPU↔NIC pairing ──────────────────────────────────────────────

/**
 * Pair GPUs and NICs within one PCIe slot by nearest bus number.
 * Uses a greedy algorithm (iterate GPUs sorted by bus, assign nearest free NIC).
 *
 * This correctly handles bifurcation slots (2 GPUs + 2 NICs per slot).
 * Example — Slot 2 on kc-vse-4-debian:
 *   GPU 0x3f, GPU 0x45  |  NIC 0x40, NIC 0x42
 *   → GPU(0x3f) → NIC(0x40) dist=1  ✓
 *   → GPU(0x45) → NIC(0x42) dist=3  ✓  (NIC 0x40 already taken)
 */
inline void PairInSlot(
    std::vector<PciDevice>& gpus,
    std::vector<PciDevice>& nics,
    const std::string& slot_name,
    int& next_our_id,
    const std::unordered_map<std::string, int>& hip_map,
    std::vector<GpuNicPair>& out)
{
  std::sort(gpus.begin(), gpus.end(),
            [](const PciDevice& a, const PciDevice& b) { return a.bus < b.bus; });
  std::sort(nics.begin(), nics.end(),
            [](const PciDevice& a, const PciDevice& b) { return a.bus < b.bus; });

  std::set<std::string> used_nics;

  for (size_t gi = 0; gi < gpus.size(); ++gi) {
    const auto& gpu = gpus[gi];

    GpuNicPair pair;
    pair.our_id    = next_our_id++;
    pair.gpu_pci   = gpu.addr;
    pair.gpu_name  = AmdGpuName(gpu.device);
    pair.numa_node = ReadNumaNode(gpu.addr);

    // Slot label: "1" for single GPU slot, "1A"/"1B" for bifurcation
    if (gpus.size() == 1)
      pair.slot_label = slot_name;
    else
      pair.slot_label = slot_name + static_cast<char>('A' + gi);

    // Greedy: find nearest NIC not yet assigned
    int   min_dist = INT_MAX;
    const PciDevice* best = nullptr;
    for (const auto& nic : nics) {
      if (used_nics.count(nic.addr)) continue;
      const int dist = std::abs(nic.bus - gpu.bus);
      if (dist < min_dist) { min_dist = dist; best = &nic; }
    }
    if (best) {
      used_nics.insert(best->addr);
      pair.nic_pci  = best->addr;
      pair.nic_name = MellanoxNicName(best->device);
    }

    // HIP index lookup (hip_map is empty when compiled without ROCm)
    auto it = hip_map.find(gpu.addr);
    if (it != hip_map.end()) pair.hip_id = it->second;

    if (Verbose())
      std::cerr << "[VERBOSE] Paired: id=" << pair.our_id
                << " slot=" << pair.slot_label
                << " gpu=" << pair.gpu_pci
                << " nic=" << (pair.nic_pci.empty() ? "NONE" : pair.nic_pci)
                << " hip=" << pair.hip_id
                << " numa=" << pair.numa_node << "\n";

    out.push_back(std::move(pair));
  }
}

} // namespace detail


// ═════════════════════════════════════════════════════════════════════════════
// Public API
// ═════════════════════════════════════════════════════════════════════════════

/**
 * @brief Detect AMD GPU + Mellanox NIC pairs from PCIe topology
 *
 * Reads /sys/bus/pci/slots/ and /sys/bus/pci/devices/.
 * With ENABLE_ROCM: also maps each GPU PCI address to a HIP device index.
 *
 * Returns ServerTopology::is_server_topology = false if:
 *   - no AMD GPUs found
 *   - no Mellanox NICs found
 *   - /sys/bus/pci/slots/ is unavailable (no root?)
 */
inline ServerTopology Detect() {
  ServerTopology topo;

  const auto all_devices = detail::ReadAllDevices();
  const auto slots        = detail::ReadSlots();

  // Separate GPU and NIC lists; deduplicate NICs (keep only .0 port)
  std::vector<detail::PciDevice> all_gpus, all_nics;
  for (const auto& d : all_devices) {
    if (detail::IsAmdGpu(d))
      all_gpus.push_back(d);
    else if (detail::IsMellanoxNic(d) && detail::IsPrimaryPort(d.addr))
      all_nics.push_back(d);
  }

  if (all_gpus.empty() || all_nics.empty() || slots.empty()) {
    if (Verbose()) {
      std::cerr << "[VERBOSE] Early exit: gpus=" << all_gpus.size()
                << " nics=" << all_nics.size()
                << " slots=" << slots.size() << "\n";
    }
    return topo;  // is_server_topology = false
  }

  // HIP PCI map: populated with ROCm, empty otherwise
  std::unordered_map<std::string, int> hip_map;
#ifdef ENABLE_ROCM
  hip_map = detail::BuildHipPciMap();
  if (Verbose())
    std::cerr << "[VERBOSE] HIP devices found: " << hip_map.size() << "\n";
#else
  if (Verbose())
    std::cerr << "[VERBOSE] Compiled without ROCm — HIP column will be '-'\n";
#endif

  int next_our_id = 0;

  for (const auto& slot : slots) {
    std::vector<detail::PciDevice> slot_gpus, slot_nics;

    for (const auto& d : all_gpus)
      if (d.bus >= slot.sec_bus && d.bus <= slot.sub_bus)
        slot_gpus.push_back(d);

    for (const auto& d : all_nics)
      if (d.bus >= slot.sec_bus && d.bus <= slot.sub_bus)
        slot_nics.push_back(d);

    if (slot_gpus.empty()) continue;

    if (Verbose())
      std::cerr << "[VERBOSE] Slot " << slot.slot_name
                << ": " << slot_gpus.size() << " GPU + "
                << slot_nics.size() << " NIC\n";

    detail::PairInSlot(
        slot_gpus, slot_nics, slot.slot_name, next_our_id,
        hip_map, topo.pairs);
  }

  topo.is_server_topology = !topo.pairs.empty();
  return topo;
}


/**
 * @brief Print topology table to stdout
 */
inline void PrintTable(const ServerTopology& topo) {
  if (!topo.is_server_topology || topo.pairs.empty()) {
    std::cout << "\nServer topology (AMD GPU + Mellanox NIC) NOT detected.\n"
              << "Possible reasons:\n"
              << "  - No AMD GPUs or Mellanox NICs in system\n"
              << "  - /sys/bus/pci/slots/ unavailable (try: sudo)\n"
              << "  - secondary_bus_number not readable (try: sudo)\n";
    return;
  }

  // Check if HIP data is available
  const bool has_hip = std::any_of(topo.pairs.begin(), topo.pairs.end(),
                                   [](const GpuNicPair& p) { return p.hip_id >= 0; });

  const char* sep =
    " ---  ------  ------------------  -------------------------"
    "  ------------------  --------------------  -----  ----\n";

  std::cout << "\n=== GPU + Mellanox Topology ===\n\n";
  printf(" %3s  %-6s  %-18s  %-25s  %-18s  %-20s  %5s  %s\n",
         "ID", "Slot", "GPU PCI", "GPU Model",
         "NIC PCI", "NIC Model", "HIP", "NUMA");
  printf("%s", sep);

  for (const auto& p : topo.pairs) {
    const std::string hip_str  = (p.hip_id >= 0)    ? std::to_string(p.hip_id)    : "-";
    const std::string numa_str = (p.numa_node >= 0)  ? std::to_string(p.numa_node) : "-";
    printf(" %3d  %-6s  %-18s  %-25s  %-18s  %-20s  %5s  %s\n",
           p.our_id,
           p.slot_label.c_str(),
           p.gpu_pci.c_str(),
           p.gpu_name.c_str(),
           p.nic_pci.empty() ? "-" : p.nic_pci.c_str(),
           p.nic_name.empty() ? "-" : p.nic_name.c_str(),
           hip_str.c_str(),
           numa_str.c_str());
  }

  printf("%s", sep);
  printf(" Total: %zu GPU+NIC pair(s)   Server topology: %s\n\n",
         topo.pairs.size(),
         topo.is_server_topology ? "DETECTED" : "no");

#ifndef ENABLE_ROCM
  std::cout << " [HIP column is '-': rebuild with -DENABLE_ROCM=ON for HIP indices]\n\n";
#endif
}


/**
 * @brief Save topology as configGPU.json (DrvGPU format v1.1)
 *
 * The generated file can be used as configGPU.json directly.
 * Fields gpu_pci / nic_pci will be used by DrvGPU Phase 3 integration
 * to locate the correct physical GPU by stable PCI address.
 */
inline void SaveJson(const ServerTopology& topo,
                     const std::string& path = "gpu_map.json") {
  if (!topo.is_server_topology) {
    std::cerr << "SaveJson: topology not detected, nothing to save.\n";
    return;
  }

  std::ofstream f(path);
  if (!f) {
    std::cerr << "SaveJson: cannot open '" << path << "' for writing.\n";
    return;
  }

  f << "{\n"
    << "  \"description\": \"GPU+NIC PCI mapping — auto-generated by GetGPU_and_Mellanox\",\n"
    << "  \"version\": \"1.1\",\n"
    << "  \"gpus\": [\n";

  for (size_t i = 0; i < topo.pairs.size(); ++i) {
    const auto& p    = topo.pairs[i];
    const bool  last = (i + 1 == topo.pairs.size());
    f << "    {\n"
      << "      \"id\": "          << p.our_id << ",\n"
      << "      \"name\": \"GPU_" << p.slot_label << "\",\n"
      << "      \"gpu_pci\": \""  << p.gpu_pci << "\",\n"
      << "      \"nic_pci\": \""  << p.nic_pci << "\",\n"
      << "      \"numa_node\": "  << p.numa_node << ",\n"
      << "      \"is_active\": true,\n"
      << "      \"is_console\": " << (i == 0 ? "true" : "false") << ",\n"
      << "      \"is_logger\": true,\n"
      << "      \"is_prof\": true,\n"
      << "      \"max_memory_percent\": 70,\n"
      << "      \"log_level\": \"INFO\"\n"
      << "    }" << (last ? "\n" : ",\n");
  }

  f << "  ]\n}\n";

  std::cout << "Saved: " << path << "\n";
  std::cout << "Copy gpu_map.json → configGPU.json in your project.\n\n";
}

} // namespace gpu_mellanox
