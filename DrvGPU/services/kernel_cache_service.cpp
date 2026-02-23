/**
 * @file kernel_cache_service.cpp
 * @brief On-disk kernel cache implementation
 *
 * Extracted from FormScriptGenerator (signal_generators).
 * Generic, storage-agnostic — works with filesystem directly.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-22
 */

#include "kernel_cache_service.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <chrono>
#include <iomanip>
#include <cstdio>

namespace fs = std::filesystem;

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Constructor
// ════════════════════════════════════════════════════════════════════════════

KernelCacheService::KernelCacheService(const std::string& base_dir,
                                       BackendType backend_type)
    : base_dir_(base_dir), backend_type_(backend_type) {
}

// ════════════════════════════════════════════════════════════════════════════
// Save
// ════════════════════════════════════════════════════════════════════════════

void KernelCacheService::Save(const std::string& name,
                               const std::string& cl_source,
                               const std::vector<uint8_t>& binary,
                               const std::string& metadata,
                               const std::string& comment) {
  if (name.empty()) {
    throw std::runtime_error(
        "KernelCacheService::Save: name cannot be empty");
  }

  std::string bin_dir = GetBinDir();
  fs::create_directories(bin_dir);

  // Version old files if they exist
  VersionOldFiles(name);

  // Save .cl source
  std::string cl_path = base_dir_ + "/" + name + ".cl";
  {
    std::ofstream f(cl_path);
    if (!f.is_open()) {
      throw std::runtime_error(
          "KernelCacheService::Save: cannot write " + cl_path);
    }
    f << cl_source;
  }

  // Save binary
  std::string bin_path = bin_dir + "/" + name + GetBinarySuffix();
  {
    std::ofstream f(bin_path, std::ios::binary);
    if (!f.is_open()) {
      throw std::runtime_error(
          "KernelCacheService::Save: cannot write " + bin_path);
    }
    f.write(reinterpret_cast<const char*>(binary.data()),
            static_cast<std::streamsize>(binary.size()));
  }

  // Update manifest
  WriteManifestEntry(name, metadata, comment);
}

// ════════════════════════════════════════════════════════════════════════════
// Load
// ════════════════════════════════════════════════════════════════════════════

KernelCacheService::CacheEntry
KernelCacheService::Load(const std::string& name) const {
  CacheEntry entry;

  std::string cl_path = base_dir_ + "/" + name + ".cl";
  std::string bin_path = GetBinDir() + "/" + name + GetBinarySuffix();

  // Try binary (fast path)
  if (fs::exists(bin_path)) {
    std::ifstream f(bin_path, std::ios::binary | std::ios::ate);
    if (f.is_open()) {
      auto size = f.tellg();
      f.seekg(0, std::ios::beg);
      entry.binary.resize(static_cast<size_t>(size));
      f.read(reinterpret_cast<char*>(entry.binary.data()),
             static_cast<std::streamsize>(size));
    }
  }

  // Try source
  if (fs::exists(cl_path)) {
    std::ifstream f(cl_path);
    if (f.is_open()) {
      std::ostringstream ss;
      ss << f.rdbuf();
      entry.source = ss.str();
    }
  }

  // Neither found
  if (!entry.has_binary() && !entry.has_source()) {
    throw std::runtime_error(
        "KernelCacheService::Load: kernel '" + name
        + "' not found (checked: " + bin_path + ", " + cl_path + ")");
  }

  return entry;
}

// ════════════════════════════════════════════════════════════════════════════
// ListKernels
// ════════════════════════════════════════════════════════════════════════════

std::vector<std::string> KernelCacheService::ListKernels() const {
  std::vector<std::string> names;
  std::string manifest_path = base_dir_ + "/manifest.json";

  if (!fs::exists(manifest_path)) return names;

  std::ifstream f(manifest_path);
  std::string content((std::istreambuf_iterator<char>(f)),
                       std::istreambuf_iterator<char>());

  // Simple JSON parsing: find all "name": "value" pairs
  std::string search = "\"name\"";
  size_t pos = 0;
  while ((pos = content.find(search, pos)) != std::string::npos) {
    pos += search.size();
    // Skip whitespace and colon
    while (pos < content.size() &&
           (content[pos] == ' ' || content[pos] == ':' || content[pos] == '"'))
      ++pos;
    size_t end = content.find('"', pos);
    if (end != std::string::npos) {
      names.push_back(content.substr(pos, end - pos));
      pos = end + 1;
    }
  }

  return names;
}

// ════════════════════════════════════════════════════════════════════════════
// GetBinDir
// ════════════════════════════════════════════════════════════════════════════

std::string KernelCacheService::GetBinDir() const {
  return base_dir_ + "/bin";
}

// ════════════════════════════════════════════════════════════════════════════
// GetBinarySuffix
// ════════════════════════════════════════════════════════════════════════════

std::string KernelCacheService::GetBinarySuffix() const {
  switch (backend_type_) {
    case BackendType::ROCm:
      return "_rocm.hsaco";
    case BackendType::OPENCL:
    default:
      return "_opencl.bin";
  }
}

// ════════════════════════════════════════════════════════════════════════════
// VersionOldFiles
// ════════════════════════════════════════════════════════════════════════════

void KernelCacheService::VersionOldFiles(const std::string& name) const {
  std::string cl_path = base_dir_ + "/" + name + ".cl";
  std::string bin_dir = GetBinDir();
  std::string bin_path = bin_dir + "/" + name + GetBinarySuffix();

  bool cl_exists = fs::exists(cl_path);
  bool bin_exists = fs::exists(bin_path);

  if (!cl_exists && !bin_exists) return;

  // Find next free suffix: _00, _01, ...
  int suffix = 0;
  while (suffix <= 99) {
    char buf[8];
    snprintf(buf, sizeof(buf), "_%02d", suffix);
    std::string s(buf);

    std::string old_cl = base_dir_ + "/" + name + s + ".cl";
    // Binary suffix: e.g. name_opencl_00.bin
    std::string suffix_str = GetBinarySuffix();
    // Insert version before extension: _opencl_00.bin
    auto dot_pos = suffix_str.rfind('.');
    std::string versioned_suffix = suffix_str.substr(0, dot_pos)
                                   + s + suffix_str.substr(dot_pos);
    std::string old_bin = bin_dir + "/" + name + versioned_suffix;

    if (!fs::exists(old_cl) && !fs::exists(old_bin)) {
      if (cl_exists)  fs::rename(cl_path, old_cl);
      if (bin_exists) fs::rename(bin_path, old_bin);
      return;
    }
    ++suffix;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// WriteManifestEntry
// ════════════════════════════════════════════════════════════════════════════

void KernelCacheService::WriteManifestEntry(
    const std::string& name,
    const std::string& metadata,
    const std::string& comment) const {

  std::string manifest_path = base_dir_ + "/manifest.json";
  std::string timestamp = GetTimestamp();

  // Read existing manifest or start fresh
  std::string content;
  if (fs::exists(manifest_path)) {
    std::ifstream f(manifest_path);
    content.assign((std::istreambuf_iterator<char>(f)),
                    std::istreambuf_iterator<char>());
  }

  // Build new entry JSON
  std::ostringstream entry;
  entry << "    {\n";
  entry << "      \"name\": \"" << name << "\",\n";
  entry << "      \"comment\": \"" << comment << "\",\n";
  entry << "      \"created\": \"" << timestamp << "\",\n";
  entry << "      \"params\": \"" << metadata << "\",\n";

  // Backend string
  const char* backend_str = "opencl";
  if (backend_type_ == BackendType::ROCm) backend_str = "rocm";
  entry << "      \"backend\": \"" << backend_str << "\"\n";
  entry << "    }";

  // Parse existing entries (skip entry with same name)
  std::vector<std::string> entries;
  if (!content.empty()) {
    size_t arr_start = content.find('[');
    size_t arr_end = content.rfind(']');
    if (arr_start != std::string::npos && arr_end != std::string::npos) {
      std::string arr = content.substr(arr_start + 1, arr_end - arr_start - 1);

      size_t pos = 0;
      while (true) {
        size_t obj_start = arr.find('{', pos);
        if (obj_start == std::string::npos) break;
        size_t obj_end = arr.find('}', obj_start);
        if (obj_end == std::string::npos) break;

        std::string obj = arr.substr(obj_start, obj_end - obj_start + 1);

        // Check if this is the same name
        bool same_name = false;
        std::string name_check = "\"name\": \"" + name + "\"";
        if (obj.find(name_check) != std::string::npos) same_name = true;
        name_check = "\"name\":\"" + name + "\"";
        if (obj.find(name_check) != std::string::npos) same_name = true;

        if (!same_name) {
          entries.push_back("    " + obj);
        }

        pos = obj_end + 1;
      }
    }
  }

  // Add new entry
  entries.push_back(entry.str());

  // Write manifest (binary mode — LF only)
  std::ofstream f(manifest_path, std::ios::binary);
  f << "{\n";
  f << "  \"version\": 1,\n";
  f << "  \"kernels\": [\n";
  for (size_t i = 0; i < entries.size(); ++i) {
    f << entries[i];
    if (i + 1 < entries.size()) f << ",";
    f << "\n";
  }
  f << "  ]\n";
  f << "}\n";
}

// ════════════════════════════════════════════════════════════════════════════
// GetTimestamp
// ════════════════════════════════════════════════════════════════════════════

std::string KernelCacheService::GetTimestamp() {
  auto now = std::chrono::system_clock::now();
  auto t = std::chrono::system_clock::to_time_t(now);
  std::tm tm_buf;
#ifdef _WIN32
  localtime_s(&tm_buf, &t);
#else
  localtime_r(&t, &tm_buf);
#endif

  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm_buf);
  return std::string(buf);
}

} // namespace drv_gpu_lib