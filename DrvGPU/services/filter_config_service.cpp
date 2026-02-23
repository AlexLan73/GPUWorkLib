/**
 * @file filter_config_service.cpp
 * @brief FilterConfigService implementation — JSON save/load for filter configs
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-22
 */

#include "filter_config_service.hpp"
#include "storage/file_storage_backend.hpp"

#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cstdio>

namespace fs = std::filesystem;

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Constructor
// ════════════════════════════════════════════════════════════════════════════

FilterConfigService::FilterConfigService(const std::string& base_dir)
    : base_dir_(base_dir),
      storage_(std::make_unique<FileStorageBackend>(base_dir)) {
}

// ════════════════════════════════════════════════════════════════════════════
// Save
// ════════════════════════════════════════════════════════════════════════════

void FilterConfigService::Save(const std::string& name,
                                const FilterConfigData& data,
                                const std::string& comment) {
  if (name.empty()) {
    throw std::runtime_error(
        "FilterConfigService::Save: name cannot be empty");
  }

  // Version old files
  VersionOldFiles(name);

  std::string timestamp = GetTimestamp();
  std::string effective_comment = comment.empty() ? data.comment : comment;
  std::string json = ToJson(name, data, effective_comment, timestamp);

  // Convert string to bytes and save
  std::vector<uint8_t> bytes(json.begin(), json.end());
  storage_->Save("filters/" + name + ".json", bytes);
}

// ════════════════════════════════════════════════════════════════════════════
// Load
// ════════════════════════════════════════════════════════════════════════════

FilterConfigData FilterConfigService::Load(const std::string& name) const {
  std::string key = "filters/" + name + ".json";
  if (!storage_->Exists(key)) {
    throw std::runtime_error(
        "FilterConfigService::Load: filter '" + name + "' not found");
  }

  auto bytes = storage_->Load(key);
  std::string json(bytes.begin(), bytes.end());
  return FromJson(json);
}

// ════════════════════════════════════════════════════════════════════════════
// ListFilters
// ════════════════════════════════════════════════════════════════════════════

std::vector<std::string> FilterConfigService::ListFilters() const {
  auto keys = storage_->List("filters/");
  std::vector<std::string> names;

  for (const auto& key : keys) {
    // Extract name from "filters/{name}.json"
    if (key.size() <= 13) continue;  // "filters/" + ".json" = 13
    std::string name = key.substr(8);  // remove "filters/"
    // Remove .json extension
    auto dot = name.rfind(".json");
    if (dot != std::string::npos) {
      name = name.substr(0, dot);
    }
    // Skip versioned files (_00, _01, ...)
    if (name.size() >= 3) {
      std::string tail = name.substr(name.size() - 3);
      if (tail[0] == '_' && std::isdigit(tail[1]) && std::isdigit(tail[2])) {
        continue;
      }
    }
    names.push_back(name);
  }

  return names;
}

// ════════════════════════════════════════════════════════════════════════════
// Exists
// ════════════════════════════════════════════════════════════════════════════

bool FilterConfigService::Exists(const std::string& name) const {
  return storage_->Exists("filters/" + name + ".json");
}

// ════════════════════════════════════════════════════════════════════════════
// VersionOldFiles
// ════════════════════════════════════════════════════════════════════════════

void FilterConfigService::VersionOldFiles(const std::string& name) const {
  std::string key = "filters/" + name + ".json";
  if (!storage_->Exists(key)) return;

  // Find next free suffix
  for (int i = 0; i <= 99; ++i) {
    char buf[8];
    snprintf(buf, sizeof(buf), "_%02d", i);
    std::string versioned_key = "filters/" + name + std::string(buf) + ".json";
    if (!storage_->Exists(versioned_key)) {
      // Rename: load old, save as versioned, (old will be overwritten by caller)
      auto data = storage_->Load(key);
      storage_->Save(versioned_key, data);

      // Delete original (via filesystem since IStorageBackend has no Delete)
      fs::path original = fs::path(base_dir_) / key;
      if (fs::exists(original)) {
        fs::remove(original);
      }
      return;
    }
  }
}

// ════════════════════════════════════════════════════════════════════════════
// GetTimestamp
// ════════════════════════════════════════════════════════════════════════════

std::string FilterConfigService::GetTimestamp() {
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

// ════════════════════════════════════════════════════════════════════════════
// JSON Serialization
// ════════════════════════════════════════════════════════════════════════════

std::string FilterConfigService::ToJson(
    const std::string& name,
    const FilterConfigData& data,
    const std::string& comment,
    const std::string& timestamp) {

  std::ostringstream j;
  j << "{\n";
  j << "  \"name\": \"" << name << "\",\n";
  j << "  \"type\": \"" << data.type << "\",\n";
  j << "  \"comment\": \"" << comment << "\",\n";
  j << "  \"created\": \"" << timestamp << "\"";

  if (data.type == "fir" && !data.coefficients.empty()) {
    j << ",\n  \"coefficients\": [";
    for (size_t i = 0; i < data.coefficients.size(); ++i) {
      if (i > 0) j << ", ";
      j << std::scientific << data.coefficients[i];
    }
    j << "]";
  }

  if (data.type == "iir" && !data.sections.empty()) {
    j << ",\n  \"sections\": [\n";
    for (size_t i = 0; i < data.sections.size(); ++i) {
      const auto& s = data.sections[i];
      j << "    {\"b0\": " << s[0]
        << ", \"b1\": " << s[1]
        << ", \"b2\": " << s[2]
        << ", \"a1\": " << s[3]
        << ", \"a2\": " << s[4] << "}";
      if (i + 1 < data.sections.size()) j << ",";
      j << "\n";
    }
    j << "  ]";
  }

  j << "\n}\n";
  return j.str();
}

FilterConfigData FilterConfigService::FromJson(const std::string& json) {
  FilterConfigData data;

  // Parse name
  auto parse_string = [&json](const std::string& key) -> std::string {
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos);
    if (pos == std::string::npos) return "";
    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return "";
    pos++;
    auto end = json.find('"', pos);
    if (end == std::string::npos) return "";
    return json.substr(pos, end - pos);
  };

  data.name = parse_string("name");
  data.type = parse_string("type");
  data.comment = parse_string("comment");
  data.created = parse_string("created");

  if (data.type == "fir") {
    // Parse coefficients array
    auto pos = json.find("\"coefficients\"");
    if (pos != std::string::npos) {
      auto arr_start = json.find('[', pos);
      auto arr_end = json.find(']', arr_start);
      if (arr_start != std::string::npos && arr_end != std::string::npos) {
        std::string arr = json.substr(arr_start + 1, arr_end - arr_start - 1);
        std::string num;
        for (char c : arr) {
          if (c == '-' || c == '.' || (c >= '0' && c <= '9') ||
              c == 'e' || c == 'E' || c == '+') {
            num += c;
          } else if (!num.empty()) {
            data.coefficients.push_back(std::stof(num));
            num.clear();
          }
        }
        if (!num.empty()) data.coefficients.push_back(std::stof(num));
      }
    }
  } else if (data.type == "iir") {
    // Parse sections: [{b0:..., b1:..., b2:..., a1:..., a2:...}, ...]
    auto parse_float = [](const std::string& s, const std::string& key) -> float {
      auto pos = s.find("\"" + key + "\"");
      if (pos == std::string::npos) return 0.0f;
      pos = s.find(':', pos);
      if (pos == std::string::npos) return 0.0f;
      pos++;
      while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t')) pos++;
      std::string num;
      for (; pos < s.size(); pos++) {
        char c = s[pos];
        if (c == '-' || c == '.' || (c >= '0' && c <= '9') ||
            c == 'e' || c == 'E' || c == '+') {
          num += c;
        } else break;
      }
      return num.empty() ? 0.0f : std::stof(num);
    };

    auto pos = json.find("\"sections\"");
    if (pos != std::string::npos) {
      size_t search_from = pos;
      while (true) {
        auto brace_start = json.find('{', search_from);
        if (brace_start == std::string::npos || brace_start < pos) {
          search_from = (brace_start != std::string::npos) ? brace_start + 1 : json.size();
          if (search_from >= json.size()) break;
          continue;
        }
        auto brace_end = json.find('}', brace_start);
        if (brace_end == std::string::npos) break;

        std::string sec_str = json.substr(brace_start, brace_end - brace_start + 1);

        // Check this looks like a section (has b0)
        if (sec_str.find("\"b0\"") != std::string::npos) {
          std::array<float, 5> sec;
          sec[0] = parse_float(sec_str, "b0");
          sec[1] = parse_float(sec_str, "b1");
          sec[2] = parse_float(sec_str, "b2");
          sec[3] = parse_float(sec_str, "a1");
          sec[4] = parse_float(sec_str, "a2");
          data.sections.push_back(sec);
        }

        search_from = brace_end + 1;
      }
    }
  }

  return data;
}

} // namespace drv_gpu_lib
