#pragma once

/**
 * @file file_storage_backend.hpp
 * @brief File-based implementation of IStorageBackend
 *
 * Stores data as files in base_dir. Keys with '/' create subdirectories.
 * Uses std::filesystem (C++17).
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-22
 */

#include "i_storage_backend.hpp"

#include <string>
#include <vector>
#include <cstdint>

namespace drv_gpu_lib {

class FileStorageBackend : public IStorageBackend {
public:
  /**
   * @brief Construct file storage with given root directory
   * @param base_dir Root directory for all stored files
   */
  explicit FileStorageBackend(const std::string& base_dir);

  void Save(const std::string& key, const std::vector<uint8_t>& data) override;
  std::vector<uint8_t> Load(const std::string& key) const override;
  std::vector<std::string> List(const std::string& prefix = "") const override;
  bool Exists(const std::string& key) const override;

  /// Get the base directory path
  const std::string& GetBaseDir() const { return base_dir_; }

private:
  std::string base_dir_;

  /// Build full filesystem path from key
  std::string KeyToPath(const std::string& key) const;
};

} // namespace drv_gpu_lib