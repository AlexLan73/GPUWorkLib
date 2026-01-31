#include "module_registry.hpp"
#include <iostream>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Конструктор и деструктор
// ════════════════════════════════════════════════════════════════════════════

ModuleRegistry::ModuleRegistry() = default;

ModuleRegistry::~ModuleRegistry() {
    Clear();
}

// ════════════════════════════════════════════════════════════════════════════
// Move конструктор и оператор
// ════════════════════════════════════════════════════════════════════════════

ModuleRegistry::ModuleRegistry(ModuleRegistry&& other) noexcept {
    std::lock_guard<std::mutex> lock(other.mutex_);
    modules_ = std::move(other.modules_);
}

ModuleRegistry& ModuleRegistry::operator=(ModuleRegistry&& other) noexcept {
    if (this != &other) {
        std::lock_guard<std::mutex> lock_this(mutex_);
        std::lock_guard<std::mutex> lock_other(other.mutex_);
        modules_ = std::move(other.modules_);
    }
    return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// Регистрация модулей
// ════════════════════════════════════════════════════════════════════════════

void ModuleRegistry::RegisterModule(const std::string& name, 
                                     std::shared_ptr<IComputeModule> module) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (modules_.find(name) != modules_.end()) {
        throw std::runtime_error("ModuleRegistry: module '" + name + "' already registered");
    }
    
    modules_[name] = module;
}

bool ModuleRegistry::UnregisterModule(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    return modules_.erase(name) > 0;
}

bool ModuleRegistry::HasModule(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return modules_.find(name) != modules_.end();
}

// ════════════════════════════════════════════════════════════════════════════
// Доступ к модулям
// ════════════════════════════════════════════════════════════════════════════

std::shared_ptr<IComputeModule> ModuleRegistry::GetModule(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = modules_.find(name);
    if (it == modules_.end()) {
        throw std::runtime_error("ModuleRegistry: module '" + name + "' not found");
    }
    
    return it->second;
}

std::shared_ptr<const IComputeModule> ModuleRegistry::GetModule(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = modules_.find(name);
    if (it == modules_.end()) {
        throw std::runtime_error("ModuleRegistry: module '" + name + "' not found");
    }
    
    return it->second;
}

// ════════════════════════════════════════════════════════════════════════════
// Информация о реестре
// ════════════════════════════════════════════════════════════════════════════

size_t ModuleRegistry::GetModuleCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return modules_.size();
}

std::vector<std::string> ModuleRegistry::GetModuleNames() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::string> names;
    names.reserve(modules_.size());
    
    for (const auto& pair : modules_) {
        names.push_back(pair.first);
    }
    
    return names;
}

void ModuleRegistry::PrintModules() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::cout << "\n" << std::string(50, '=') << "\n";
    std::cout << "ModuleRegistry - Registered Modules\n";
    std::cout << std::string(50, '=') << "\n";
    
    if (modules_.empty()) {
        std::cout << "No modules registered.\n";
    } else {
        for (const auto& pair : modules_) {
            std::cout << "  - " << pair.first << "\n";
        }
    }
    
    std::cout << std::string(50, '=') << "\n";
}

// ════════════════════════════════════════════════════════════════════════════
// Очистка
// ════════════════════════════════════════════════════════════════════════════

void ModuleRegistry::Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    modules_.clear();
}

} // namespace drv_gpu_lib
