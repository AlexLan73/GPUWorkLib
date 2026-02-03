/**
 * @file antenna_module.cpp - ЧАСТЬ 1/4
 * @brief Реализация Antenna FFT Module
 * 
 * ЧАСТЬ 1: Конструктор, деструктор, Initialize, вспомогательные функции
 */

#include "antenna_module.hpp"
#include "common/logger.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <iostream>

#ifndef ANTENNA_KERNELS_PATH
#define ANTENNA_KERNELS_PATH "kernels"
#endif

namespace drv_gpu_lib {
namespace antenna {

// ════════════════════════════════════════════════════════════════════════════
// Конструктор / Деструктор
// ════════════════════════════════════════════════════════════════════════════

AntennaModule::AntennaModule(IBackend* backend, const AntennaParams& params)
    : backend_(backend)
    , params_(params)
    , batch_config_()
    , initialized_(false)
    , nFFT_(0)
    , context_(nullptr)
    , queue_(nullptr)
    , device_(nullptr)
    , main_plan_handle_(0)
    , batch_plan_handle_(0)
    , batch_plan_beams_(0)
    , padding_kernel_(nullptr)
    , post_kernel_(nullptr)
    , reduction_kernel_(nullptr)
    , program_(nullptr)
    , batch_buffers_size_(0)
{
    if (!backend_) {
        throw std::invalid_argument("AntennaModule: backend cannot be null");
    }
    
    if (!params_.IsValid()) {
        throw std::invalid_argument("AntennaModule: invalid parameters");
    }
    
    DRVGPU_LOG_INFO("AntennaModule", "Created (not initialized)");
}

AntennaModule::~AntennaModule() {
    Cleanup();
}

AntennaModule::AntennaModule(AntennaModule&& other) noexcept
    : backend_(other.backend_)
    , params_(other.params_)
    , batch_config_(other.batch_config_)
    , initialized_(other.initialized_)
    , nFFT_(other.nFFT_)
    , context_(other.context_)
    , queue_(other.queue_)
    , device_(other.device_)
    , main_plan_handle_(other.main_plan_handle_)
    , batch_plan_handle_(other.batch_plan_handle_)
    , batch_plan_beams_(other.batch_plan_beams_)
    , padding_kernel_(other.padding_kernel_)
    , post_kernel_(other.post_kernel_)
    , reduction_kernel_(other.reduction_kernel_)
    , program_(other.program_)
    , buffer_fft_input_(std::move(other.buffer_fft_input_))
    , buffer_fft_output_(std::move(other.buffer_fft_output_))
    , batch_fft_input_(std::move(other.batch_fft_input_))
    , batch_fft_output_(std::move(other.batch_fft_output_))
    , batch_buffers_size_(other.batch_buffers_size_)
    , buffer_maxima_(std::move(other.buffer_maxima_))
{
    other.main_plan_handle_ = 0;
    other.batch_plan_handle_ = 0;
    other.padding_kernel_ = nullptr;
    other.post_kernel_ = nullptr;
    other.reduction_kernel_ = nullptr;
    other.program_ = nullptr;
    other.initialized_ = false;
}

AntennaModule& AntennaModule::operator=(AntennaModule&& other) noexcept {
    if (this != &other) {
        Cleanup();
        
        backend_ = other.backend_;
        params_ = other.params_;
        batch_config_ = other.batch_config_;
        initialized_ = other.initialized_;
        nFFT_ = other.nFFT_;
        context_ = other.context_;
        queue_ = other.queue_;
        device_ = other.device_;
        main_plan_handle_ = other.main_plan_handle_;
        batch_plan_handle_ = other.batch_plan_handle_;
        batch_plan_beams_ = other.batch_plan_beams_;
        padding_kernel_ = other.padding_kernel_;
        post_kernel_ = other.post_kernel_;
        reduction_kernel_ = other.reduction_kernel_;
        program_ = other.program_;
        buffer_fft_input_ = std::move(other.buffer_fft_input_);
        buffer_fft_output_ = std::move(other.buffer_fft_output_);
        batch_fft_input_ = std::move(other.batch_fft_input_);
        batch_fft_output_ = std::move(other.batch_fft_output_);
        batch_buffers_size_ = other.batch_buffers_size_;
        buffer_maxima_ = std::move(other.buffer_maxima_);
        
        other.main_plan_handle_ = 0;
        other.batch_plan_handle_ = 0;
        other.padding_kernel_ = nullptr;
        other.post_kernel_ = nullptr;
        other.reduction_kernel_ = nullptr;
        other.program_ = nullptr;
        other.initialized_ = false;
    }
    return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// Жизненный цикл: Initialize
// ════════════════════════════════════════════════════════════════════════════

void AntennaModule::Initialize() {
    if (initialized_) {
        DRVGPU_LOG_WARNING("AntennaModule", "Already initialized");
        return;
    }
    
    DRVGPU_LOG_INFO("AntennaModule", "Initializing...");
    
    // Получить OpenCL ресурсы из backend
    context_ = static_cast<cl_context>(backend_->GetNativeContext());
    device_ = static_cast<cl_device_id>(backend_->GetNativeDevice());
    queue_ = static_cast<cl_command_queue>(backend_->GetNativeQueue());
    
    if (!context_ || !device_ || !queue_) {
        throw std::runtime_error("AntennaModule: Invalid OpenCL handles from backend");
    }
    
    // Вычислить nFFT
    nFFT_ = CalculateNFFT(params_.count_points);
    
    // Инициализировать clFFT
    clfftSetupData fftSetup;
    clfftInitSetupData(&fftSetup);
    clfftStatus status = clfftSetup(&fftSetup);
    if (status != CLFFT_SUCCESS) {
        throw std::runtime_error("AntennaModule: clfftSetup failed");
    }
    
    // Создать kernels
    CreateKernels();
    
    initialized_ = true;
    DRVGPU_LOG_INFO("AntennaModule", "Initialized successfully ✅");
}

// ════════════════════════════════════════════════════════════════════════════
// Жизненный цикл: Cleanup
// ════════════════════════════════════════════════════════════════════════════

void AntennaModule::Cleanup() {
    if (!initialized_) {
        return;
    }
    
    DRVGPU_LOG_INFO("AntennaModule", "Cleanup...");
    
    ReleaseFFTPlan();
    ReleaseKernels();
    
    initialized_ = false;
    DRVGPU_LOG_INFO("AntennaModule", "Cleanup complete");
}

// ════════════════════════════════════════════════════════════════════════════
// Вспомогательные методы: nFFT вычисление
// ════════════════════════════════════════════════════════════════════════════

size_t AntennaModule::CalculateNFFT(size_t count_points) const {
    if (!IsPowerOf2(count_points)) {
        count_points = NextPowerOf2(count_points);
    }
    return count_points * 2;
}

bool AntennaModule::IsPowerOf2(size_t n) const {
    return n > 0 && (n & (n - 1)) == 0;
}

size_t AntennaModule::NextPowerOf2(size_t n) const {
    if (n == 0) return 1;
    if (IsPowerOf2(n)) return n;
    
    size_t power = 1;
    while (power < n) {
        power <<= 1;
    }
    return power;
}

// ════════════════════════════════════════════════════════════════════════════
// Вспомогательные методы: Память и стратегия
// ════════════════════════════════════════════════════════════════════════════

size_t AntennaModule::EstimateRequiredMemory() const {
    // Входные данные
    size_t input_size = params_.beam_count * params_.count_points * sizeof(std::complex<float>);
    
    // FFT буферы (input + output)
    size_t fft_buffers = params_.beam_count * nFFT_ * sizeof(std::complex<float>) * 2;
    
    // Post-processing буферы (выбранные точки + magnitude)
    size_t post_buffers = params_.beam_count * params_.out_count_points_fft *
                         (sizeof(std::complex<float>) + sizeof(float));
    
    // Результаты (MaxValue структуры)
    size_t result_size = params_.beam_count * params_.max_peaks_count * 32; // 32 bytes per MaxValue
    
    return input_size + fft_buffers + post_buffers + result_size;
}

bool AntennaModule::CheckAvailableMemory(size_t required_memory) const {
    // Получить размер глобальной памяти GPU
    cl_ulong global_memory = 0;
    clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_memory, nullptr);
    
    // Рассчитать доступную память с учётом порога
    size_t available_memory = static_cast<size_t>(global_memory * batch_config_.memory_usage_limit);
    
    std::cout << " ┌─────────────────────────────────────────────────────────────┐\n";
    std::cout << " │ MEMORY CHECK │\n";
    std::cout << " └─────────────────────────────────────────────────────────────┘\n";
    printf(" │ GPU Global Memory │ %10zu MB │\n", global_memory / (1024 * 1024));
    printf(" │ Threshold (%.0f%%) │ %10zu MB │\n", 
           batch_config_.memory_usage_limit * 100, available_memory / (1024 * 1024));
    printf(" │ Required Memory │ %10zu MB │\n", required_memory / (1024 * 1024));
    printf(" │ Status │ %s │\n",
           required_memory <= available_memory ? " OK ✅ " : " BATCH ⚠️ ");
    std::cout << "\n";
    
    return required_memory <= available_memory;
}

size_t AntennaModule::CalculateBatchSize(size_t total_beams) const {
    size_t batch_size = static_cast<size_t>(total_beams * batch_config_.batch_size_ratio);
    
    if (batch_size < 1) batch_size = 1;
    if (batch_size > total_beams) batch_size = total_beams;
    
    return batch_size;
}

// ════════════════════════════════════════════════════════════════════════════
// UpdateParams
// ════════════════════════════════════════════════════════════════════════════

void AntennaModule::UpdateParams(const AntennaParams& params) {
    if (!params.IsValid()) {
        throw std::invalid_argument("AntennaModule::UpdateParams: invalid parameters");
    }
    
    size_t old_nFFT = nFFT_;
    params_ = params;
    nFFT_ = CalculateNFFT(params_.count_points);
    
    // Если nFFT изменился, нужно пересоздать планы
    if (old_nFFT != nFFT_) {
        ReleaseFFTPlan();
        // Планы будут пересозданы при следующем вызове ProcessNew
    }
}

// КОНЕЦ ЧАСТИ 1/4
// Следующая часть: LoadKernelSource, CreateKernels, ReleaseKernels
