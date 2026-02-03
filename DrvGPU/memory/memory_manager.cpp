/**
 * @file memory_manager.cpp
 * @brief Реализация MemoryManager - менеджер памяти GPU
 * 
 * @author DrvGPU Team
 * @date 2026-01-31
 * @fixed 2026-02-02 - Deadlock fix (TrackAllocation/TrackFree БЕЗ mutex lock)
 */

#include "memory/memory_manager.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Конструктор и деструктор
// ════════════════════════════════════════════════════════════════════════════

MemoryManager::MemoryManager(IBackend* backend)
    : backend_(backend)
    , total_allocations_(0)
    , total_frees_(0)
    , current_allocations_(0)
    , total_bytes_allocated_(0)
    , peak_bytes_allocated_(0)
{
    if (!backend_) {
        throw std::invalid_argument("MemoryManager: backend cannot be null");
    }
}

MemoryManager::~MemoryManager() {
    Cleanup();
}

MemoryManager::MemoryManager(MemoryManager&& other) noexcept
    : backend_(other.backend_)
    , total_allocations_(other.total_allocations_)
    , total_frees_(other.total_frees_)
    , current_allocations_(other.current_allocations_)
    , total_bytes_allocated_(other.total_bytes_allocated_)
    , peak_bytes_allocated_(other.peak_bytes_allocated_)
{
    other.backend_ = nullptr;
}

MemoryManager& MemoryManager::operator=(MemoryManager&& other) noexcept {
    if (this != &other) {
        Cleanup();
        
        backend_ = other.backend_;
        total_allocations_ = other.total_allocations_;
        total_frees_ = other.total_frees_;
        current_allocations_ = other.current_allocations_;
        total_bytes_allocated_ = other.total_bytes_allocated_;
        peak_bytes_allocated_ = other.peak_bytes_allocated_;
        
        other.backend_ = nullptr;
    }
    return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// Прямое выделение памяти
// ════════════════════════════════════════════════════════════════════════════

void* MemoryManager::Allocate(size_t size_bytes, unsigned int flags) {
    if (!backend_) {
        throw std::runtime_error("MemoryManager: backend is null");
    }
    
    void* ptr = backend_->Allocate(size_bytes, flags);
    
    if (ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        TrackAllocation(size_bytes);
    }
    
    return ptr;
}

void MemoryManager::Free(void* ptr) {
    if (!ptr) return;
    
    // TODO: Если нужна статистика Free, храните map<void*, size_t>
    // и вызывайте TrackFree(size_bytes) здесь
    
    if (backend_) {
        backend_->Free(ptr);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Статистика
// ════════════════════════════════════════════════════════════════════════════

size_t MemoryManager::GetAllocationCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_allocations_;
}

size_t MemoryManager::GetTotalAllocatedBytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_bytes_allocated_;
}

void MemoryManager::PrintStatistics() const {
    std::cout << GetStatistics();
}

std::string MemoryManager::GetStatistics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::ostringstream oss;
    oss << "\n" << std::string(60, '=') << "\n";
    oss << "MemoryManager Statistics\n";
    oss << std::string(60, '=') << "\n";
    oss << std::left << std::setw(30) << "Total Allocations:" 
        << total_allocations_ << "\n";
    oss << std::left << std::setw(30) << "Total Frees:" 
        << total_frees_ << "\n";
    oss << std::left << std::setw(30) << "Current Allocations:" 
        << current_allocations_ << "\n";
    oss << std::left << std::setw(30) << "Total Allocated:" 
        << std::fixed << std::setprecision(2)
        << (total_bytes_allocated_ / (1024.0 * 1024.0)) << " MB\n";
    oss << std::left << std::setw(30) << "Peak Allocated:" 
        << std::fixed << std::setprecision(2)
        << (peak_bytes_allocated_ / (1024.0 * 1024.0)) << " MB\n";
    oss << std::string(60, '=') << "\n";
    
    return oss.str();
}

void MemoryManager::ResetStatistics() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    total_allocations_ = 0;
    total_frees_ = 0;
    current_allocations_ = 0;
    total_bytes_allocated_ = 0;
    peak_bytes_allocated_ = 0;
}

// ════════════════════════════════════════════════════════════════════════════
// Очистка
// ════════════════════════════════════════════════════════════════════════════

void MemoryManager::Cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Буферы управляются через shared_ptr и освобождаются автоматически
    // Здесь можно добавить логирование, если остались неосвобождённые буферы
    
    if (current_allocations_ > 0) {
        std::cerr << "[MemoryManager] WARNING: " << current_allocations_ 
                  << " allocations still active during cleanup!\n";
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Приватные методы
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Отслеживать выделение памяти (внутренний метод)
 * @param size_bytes Размер выделенной памяти
 * 
 * ⚠️ КРИТИЧЕСКИЙ МОМЕНТ (DEADLOCK FIX):
 * Этот метод НЕ захватывает mutex_!
 * Он вызывается ТОЛЬКО из мест, где mutex_ УЖЕ захвачен:
 * - CreateBuffer() в memory_manager.hpp (template метод)
 * - Возможно из других мест под lock
 * 
 * НИКОГДА не добавляйте std::lock_guard здесь - это приведёт к deadlock!
 */
void MemoryManager::TrackAllocation(size_t size_bytes) {
    // ⚠️ DEADLOCK FIX: НЕ добавляем std::lock_guard!
    // Этот метод вызывается ТОЛЬКО под уже захваченным mutex_
    
    total_allocations_++;
    current_allocations_++;
    total_bytes_allocated_ += size_bytes;
    
    if (total_bytes_allocated_ > peak_bytes_allocated_) {
        peak_bytes_allocated_ = total_bytes_allocated_;
    }
}

/**
 * @brief Отслеживать освобождение памяти (внутренний метод)
 * @param size_bytes Размер освобождённой памяти
 * 
 * ⚠️ КРИТИЧЕСКИЙ МОМЕНТ (DEADLOCK FIX):
 * Этот метод НЕ захватывает mutex_!
 * Вызывается только под уже захваченным lock.
 */
void MemoryManager::TrackFree(size_t size_bytes) {
    // ⚠️ DEADLOCK FIX: НЕ добавляем std::lock_guard!
    
    total_frees_++;
    current_allocations_--;
    total_bytes_allocated_ -= size_bytes;
}

} // namespace drv_gpu_lib