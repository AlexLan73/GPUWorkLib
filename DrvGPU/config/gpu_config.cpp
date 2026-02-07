/**
 * @file gpu_config.cpp
 * @brief GPUConfig implementation - JSON-based GPU configuration manager
 *
 * ============================================================================
 * IMPLEMENTATION NOTES:
 *
 * JSON Deserialization Strategy:
 *   Uses nlohmann/json with value() method for optional fields.
 *   value(key, default) returns the default if key is missing.
 *   This ensures forward compatibility when new fields are added.
 *
 * Thread Safety:
 *   All public methods acquire mutex before accessing data_.
 *   Read-only methods use const mutex lock.
 *
 * Error Handling:
 *   - Parse errors: logged to stderr, return false
 *   - Missing fields: use defaults from GPUConfigEntry
 *   - Missing file in LoadOrCreate: create default and save
 * ============================================================================
 *
 * @author Codo (AI Assistant)
 * @date 2026-02-07
 */

#include "gpu_config.hpp"

// nlohmann/json - header-only JSON library
#include "../../third_party/nlohmann/json.hpp"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <filesystem>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace drv_gpu_lib {

// ============================================================================
// JSON Serialization / Deserialization helpers
// ============================================================================

/**
 * @brief Deserialize GPUConfigEntry from JSON object
 *
 * Uses value() with defaults for EVERY field.
 * If a field is missing in JSON, the default from GPUConfigEntry is used.
 *
 * Example:
 *   JSON: { "id": 1, "name": "Evgeni" }
 *   Result: { id=1, name="Evgeni", is_prof=false, is_logger=false, ... }
 */
static GPUConfigEntry ParseGPUEntry(const json& j) {
    GPUConfigEntry entry;

    // Core identification
    entry.id                = j.value("id", entry.id);
    entry.name              = j.value("name", entry.name);

    // Feature flags (all default to false)
    entry.is_prof           = j.value("is_prof", entry.is_prof);
    entry.is_logger         = j.value("is_logger", entry.is_logger);
    entry.is_console        = j.value("is_console", entry.is_console);
    entry.is_active         = j.value("is_active", entry.is_active);
    entry.is_db             = j.value("is_db", entry.is_db);

    // Resource limits
    entry.max_memory_percent = j.value("max_memory_percent", entry.max_memory_percent);

    // Logging settings
    entry.log_level         = j.value("log_level", entry.log_level);

    return entry;
}

/**
 * @brief Serialize GPUConfigEntry to JSON object
 *
 * Writes ALL fields to JSON (including defaults).
 * This makes the config file self-documenting.
 */
static json SerializeGPUEntry(const GPUConfigEntry& entry) {
    json j;

    j["id"]                 = entry.id;
    j["name"]               = entry.name;
    j["is_prof"]            = entry.is_prof;
    j["is_logger"]          = entry.is_logger;
    j["is_console"]         = entry.is_console;
    j["is_active"]          = entry.is_active;
    j["is_db"]              = entry.is_db;
    j["max_memory_percent"] = entry.max_memory_percent;
    j["log_level"]          = entry.log_level;

    return j;
}

/**
 * @brief Serialize full GPUConfigData to JSON
 */
static json SerializeConfigData(const GPUConfigData& data) {
    json root;

    root["version"]     = data.version;
    root["description"] = data.description;

    json gpus_array = json::array();
    for (const auto& entry : data.gpus) {
        gpus_array.push_back(SerializeGPUEntry(entry));
    }
    root["gpus"] = gpus_array;

    return root;
}

// ============================================================================
// Singleton Implementation
// ============================================================================

GPUConfig& GPUConfig::GetInstance() {
    static GPUConfig instance;
    return instance;
}

GPUConfig::GPUConfig() {
    // Initialize with default config
    data_ = CreateDefaultConfig();
}

// ============================================================================
// Loading and Saving
// ============================================================================

bool GPUConfig::Load(const std::string& file_path) {
    std::lock_guard<std::mutex> lock(mutex_);

    try {
        // Open and parse JSON file
        std::ifstream file(file_path);
        if (!file.is_open()) {
            std::cerr << "[GPUConfig] ERROR: Cannot open file: " << file_path << "\n";
            return false;
        }

        json root = json::parse(file);

        // Parse root fields
        GPUConfigData new_data;
        new_data.version     = root.value("version", new_data.version);
        new_data.description = root.value("description", new_data.description);

        // Parse GPU entries
        if (root.contains("gpus") && root["gpus"].is_array()) {
            for (const auto& gpu_json : root["gpus"]) {
                new_data.gpus.push_back(ParseGPUEntry(gpu_json));
            }
        }

        // Validate: at least one GPU
        if (new_data.gpus.empty()) {
            std::cerr << "[GPUConfig] WARNING: No GPUs in config, adding default\n";
            GPUConfigEntry default_gpu;
            default_gpu.id = 0;
            default_gpu.name = "TEST";
            default_gpu.is_prof = true;
            default_gpu.is_logger = true;
            new_data.gpus.push_back(default_gpu);
        }

        // Apply
        data_ = std::move(new_data);
        file_path_ = file_path;
        loaded_ = true;

        std::cout << "[GPUConfig] Loaded " << data_.gpus.size()
                  << " GPU config(s) from: " << file_path << "\n";

        return true;

    } catch (const json::parse_error& e) {
        std::cerr << "[GPUConfig] JSON parse error: " << e.what() << "\n";
        return false;
    } catch (const std::exception& e) {
        std::cerr << "[GPUConfig] Load error: " << e.what() << "\n";
        return false;
    }
}

bool GPUConfig::LoadOrCreate(const std::string& file_path) {
    // Try to load existing file
    if (fs::exists(file_path)) {
        return Load(file_path);
    }

    // File doesn't exist - create default
    std::cout << "[GPUConfig] Config file not found, creating default: " << file_path << "\n";

    {
        std::lock_guard<std::mutex> lock(mutex_);
        data_ = CreateDefaultConfig();
        file_path_ = file_path;
        loaded_ = true;
    }

    // Save default to file
    return Save(file_path);
}

bool GPUConfig::Save(const std::string& file_path) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::string path = file_path.empty() ? file_path_ : file_path;
    if (path.empty()) {
        std::cerr << "[GPUConfig] ERROR: No file path specified for Save()\n";
        return false;
    }

    try {
        // Create parent directories if needed
        fs::path dir = fs::path(path).parent_path();
        if (!dir.empty() && !fs::exists(dir)) {
            fs::create_directories(dir);
        }

        // Serialize to JSON
        json root = SerializeConfigData(data_);

        // Write to file (pretty-printed with 2-space indent)
        std::ofstream file(path);
        if (!file.is_open()) {
            std::cerr << "[GPUConfig] ERROR: Cannot create file: " << path << "\n";
            return false;
        }

        file << root.dump(2) << "\n";
        file.close();

        file_path_ = path;

        std::cout << "[GPUConfig] Saved " << data_.gpus.size()
                  << " GPU config(s) to: " << path << "\n";

        return true;

    } catch (const std::exception& e) {
        std::cerr << "[GPUConfig] Save error: " << e.what() << "\n";
        return false;
    }
}

bool GPUConfig::IsLoaded() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return loaded_;
}

// ============================================================================
// Configuration Access
// ============================================================================

const GPUConfigEntry& GPUConfig::GetConfig(int gpu_id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    const GPUConfigEntry* found = FindConfig(gpu_id);
    if (found) {
        return *found;
    }

    // Return default entry with requested id
    default_entry_ = GPUConfigEntry();
    default_entry_.id = gpu_id;
    return default_entry_;
}

const std::vector<GPUConfigEntry>& GPUConfig::GetAllConfigs() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return data_.gpus;
}

const GPUConfigData& GPUConfig::GetData() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return data_;
}

std::vector<int> GPUConfig::GetActiveGPUIDs() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<int> active_ids;
    for (const auto& entry : data_.gpus) {
        if (entry.is_active) {
            active_ids.push_back(entry.id);
        }
    }
    return active_ids;
}

bool GPUConfig::IsProfilingEnabled(int gpu_id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    const GPUConfigEntry* found = FindConfig(gpu_id);
    return found ? found->is_prof : false;
}

bool GPUConfig::IsLoggingEnabled(int gpu_id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    const GPUConfigEntry* found = FindConfig(gpu_id);
    return found ? found->is_logger : false;
}

bool GPUConfig::IsConsoleEnabled(int gpu_id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    const GPUConfigEntry* found = FindConfig(gpu_id);
    return found ? found->is_console : false;
}

size_t GPUConfig::GetMaxMemoryPercent(int gpu_id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    const GPUConfigEntry* found = FindConfig(gpu_id);
    return found ? found->max_memory_percent : 70;
}

// ============================================================================
// Modification
// ============================================================================

void GPUConfig::SetConfig(const GPUConfigEntry& entry) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Find existing entry with same id
    for (auto& existing : data_.gpus) {
        if (existing.id == entry.id) {
            existing = entry;
            return;
        }
    }

    // Not found - add new entry
    data_.gpus.push_back(entry);
}

void GPUConfig::ResetToDefault() {
    std::lock_guard<std::mutex> lock(mutex_);
    data_ = CreateDefaultConfig();
    loaded_ = false;
    file_path_.clear();
}

// ============================================================================
// Utilities
// ============================================================================

std::string GPUConfig::GetFilePath() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return file_path_;
}

void GPUConfig::Print() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║              GPU Configuration                      ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n";
    std::cout << "  Version: " << data_.version << "\n";
    std::cout << "  File: " << (file_path_.empty() ? "(not saved)" : file_path_) << "\n";
    std::cout << "  GPUs: " << data_.gpus.size() << "\n";
    std::cout << "\n";

    for (const auto& entry : data_.gpus) {
        std::cout << "  ┌─ GPU " << entry.id << ": \"" << entry.name << "\"\n";
        std::cout << "  │  Active:  " << (entry.is_active ? "YES" : "NO") << "\n";
        std::cout << "  │  Prof:    " << (entry.is_prof ? "ON" : "off") << "\n";
        std::cout << "  │  Logger:  " << (entry.is_logger ? "ON" : "off") << "\n";
        std::cout << "  │  Console: " << (entry.is_console ? "ON" : "off") << "\n";
        std::cout << "  │  DB:      " << (entry.is_db ? "ON" : "off") << "\n";
        std::cout << "  │  MaxMem:  " << entry.max_memory_percent << "%\n";
        std::cout << "  │  LogLvl:  " << entry.log_level << "\n";
        std::cout << "  └───────────────────────────────────\n";
    }
    std::cout << "\n";
}

// ============================================================================
// Private Methods
// ============================================================================

GPUConfigData GPUConfig::CreateDefaultConfig() const {
    GPUConfigData data;
    data.version = "1.0";
    data.description = "GPU Configuration for DrvGPU";

    // Default: single GPU with profiling and logging enabled
    GPUConfigEntry default_gpu;
    default_gpu.id = 0;
    default_gpu.name = "TEST";
    default_gpu.is_prof = true;
    default_gpu.is_logger = true;
    default_gpu.is_console = true;
    default_gpu.is_active = true;

    data.gpus.push_back(default_gpu);

    return data;
}

const GPUConfigEntry* GPUConfig::FindConfig(int gpu_id) const {
    // NOTE: Caller must hold mutex_!
    for (const auto& entry : data_.gpus) {
        if (entry.id == gpu_id) {
            return &entry;
        }
    }
    return nullptr;
}

} // namespace drv_gpu_lib
