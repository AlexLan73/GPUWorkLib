/**
 * @file file_storage_backend.cpp
 * @brief File-based IStorageBackend implementation
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-22
 */

#include "file_storage_backend.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <algorithm>

namespace fs = std::filesystem;

namespace drv_gpu_lib {

FileStorageBackend::FileStorageBackend(const std::string& base_dir)
    : base_dir_(base_dir) {
}

void FileStorageBackend::Save(const std::string& key,
                               const std::vector<uint8_t>& data) {
  std::string path = KeyToPath(key);

  // Create parent directories if needed
  fs::path p(path);
  if (p.has_parent_path()) {
    fs::create_directories(p.parent_path());
  }

  std::ofstream f(path, std::ios::binary);
  if (!f.is_open()) {
    throw std::runtime_error(
        "FileStorageBackend::Save: cannot write " + path);
  }
  f.write(reinterpret_cast<const char*>(data.data()),
          static_cast<std::streamsize>(data.size()));
}

std::vector<uint8_t> FileStorageBackend::Load(const std::string& key) const {
  std::string path = KeyToPath(key);

  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f.is_open()) {
    throw std::runtime_error(
        "FileStorageBackend::Load: cannot read " + path);
  }

  auto size = f.tellg();
  f.seekg(0, std::ios::beg);

  std::vector<uint8_t> data(static_cast<size_t>(size));
  f.read(reinterpret_cast<char*>(data.data()),
         static_cast<std::streamsize>(size));

  return data;
}

std::vector<std::string> FileStorageBackend::List(
    const std::string& prefix) const {
  std::vector<std::string> result;

  fs::path base(base_dir_);
  if (!fs::exists(base) || !fs::is_directory(base)) {
    return result;
  }

  for (const auto& entry : fs::recursive_directory_iterator(base)) {
    if (!entry.is_regular_file()) continue;

    // Get relative path as key
    std::string rel = fs::relative(entry.path(), base).generic_string();

    // Apply prefix filter
    if (!prefix.empty()) {
      if (rel.rfind(prefix, 0) != 0) continue;  // doesn't start with prefix
    }

    result.push_back(rel);
  }

  std::sort(result.begin(), result.end());
  return result;
}

bool FileStorageBackend::Exists(const std::string& key) const {
  return fs::exists(KeyToPath(key));
}

std::string FileStorageBackend::KeyToPath(const std::string& key) const {
  // Use generic_string for consistent '/' separators
  return (fs::path(base_dir_) / key).string();
}

} // namespace drv_gpu_lib