/**
 * @file main.cpp
 * @brief GPU + Mellanox NIC topology detector — entry point
 *
 * Usage:
 *   sudo ./get_gpu_mellanox                    # detect and print table
 *   sudo ./get_gpu_mellanox -v                 # verbose: show sysfs reads
 *   sudo ./get_gpu_mellanox --save             # detect, print, save gpu_map.json
 *   sudo ./get_gpu_mellanox --save out.json    # detect, print, save to custom path
 *   sudo ./get_gpu_mellanox -v --save          # verbose + save
 *
 * Returns:
 *   0 — server topology detected
 *   1 — not detected (no GPU/NIC pairs, or /sys access denied)
 */

#include "gpu_mellanox_detector.hpp"
#include <cstring>

int main(int argc, char* argv[]) {
  bool save_json = false;
  std::string json_path = "gpu_map.json";

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "-v") == 0 || std::strcmp(argv[i], "--verbose") == 0) {
      gpu_mellanox::SetVerbose(true);
    } else if (std::strcmp(argv[i], "--save") == 0) {
      save_json = true;
      // Next arg (if exists and not a flag) is the output path
      if (i + 1 < argc && argv[i + 1][0] != '-') {
        json_path = argv[++i];
      }
    } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
      std::cout << "Usage: " << argv[0] << " [-v|--verbose] [--save [path.json]]\n"
                << "  -v, --verbose   Show diagnostic output (sysfs reads, bus ranges)\n"
                << "  --save [path]   Save result as JSON (default: gpu_map.json)\n"
                << "  -h, --help      Show this help\n";
      return 0;
    }
  }

  std::cout << "Scanning PCIe topology...\n";

  const auto topo = gpu_mellanox::Detect();

  gpu_mellanox::PrintTable(topo);

  if (topo.is_server_topology && save_json)
    gpu_mellanox::SaveJson(topo, json_path);

  return topo.is_server_topology ? 0 : 1;
}
