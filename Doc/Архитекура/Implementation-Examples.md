# Примеры реализации и использования LibGPU

## Оглавление
1. [Базовое использование](#базовое-использование)
2. [Создание модуля](#создание-модуля)
3. [Pipeline обработки](#pipeline-обработки)
4. [Переключение бэкендов](#переключение-бэкендов)
5. [Оптимизация производительности](#оптимизация-производительности)
6. [Примеры OpenCL kernels](#примеры-opencl-kernels)

---

## Базовое использование

### Инициализация системы

```cpp
#include <gpu_lib/DrvGPU.hpp>
#include <gpu_lib/modules/FFTPostProcessing.hpp>
#include <iostream>

int main() {
    try {
        // Получение единственного экземпляра DrvGPU
        auto& drv = gpu_lib::core::DrvGPU::getInstance();
        
        // Инициализация с OpenCL backend
        drv.initialize(gpu_lib::BackendType::OpenCL, 0);
        
        // Вывод информации об устройстве
        auto device_info = drv.getDeviceInfo();
        std::cout << "Device: " << device_info.name << "\n";
        std::cout << "Compute Units: " << device_info.compute_units << "\n";
        std::cout << "Global Memory: " << device_info.global_memory_size / (1024*1024) << " MB\n";
        std::cout << "Max Work Group Size: " << device_info.max_work_group_size << "\n";
        
        // Регистрация модулей
        auto& registry = drv.getModuleRegistry();
        registry.registerModule("FFTPostProcessing", []() {
            return std::make_unique<gpu_lib::modules::FFTPostProcessing>();
        });
        
        // Создание модуля
        auto fft_module = registry.getOrCreateModule("FFTPostProcessing");
        std::cout << "Module " << fft_module->getName() 
                  << " v" << fft_module->getVersion() << " initialized\n";
        
        // Ваша логика работы с модулем...
        
        // Очистка
        drv.shutdown();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
```

### Работа с памятью GPU

```cpp
#include <gpu_lib/DrvGPU.hpp>
#include <vector>

void memory_example() {
    auto& drv = gpu_lib::core::DrvGPU::getInstance();
    auto& mem_mgr = drv.getMemoryManager();
    
    // 1. Выделение памяти на GPU
    const size_t buffer_size = 1024 * 1024; // 1MB
    auto gpu_buffer = mem_mgr.allocate(
        buffer_size,
        gpu_lib::MemoryFlags::ReadWrite
    );
    
    std::cout << "Allocated " << gpu_buffer.size() << " bytes on GPU\n";
    
    // 2. Копирование данных CPU -> GPU
    std::vector<float> host_data(buffer_size / sizeof(float), 1.0f);
    gpu_buffer->copyFrom(host_data.data(), buffer_size);
    
    // 3. Mapping для прямого доступа (если поддерживается)
    {
        void* mapped_ptr = gpu_buffer->map(gpu_lib::MapFlags::Read);
        // Работа с mapped_ptr...
        // Автоматический unmap при выходе из scope
    }
    
    // 4. Shared буфер между модулями
    auto shared_buffer = mem_mgr.allocateShared("fft_result", buffer_size);
    
    // Другой модуль может получить тот же буфер
    auto same_buffer = mem_mgr.getShared("fft_result");
    
    // 5. Использование пула для часто выделяемых размеров
    auto pooled_buffer = mem_mgr.allocateFromPool(4096);
    
    // 6. Статистика памяти
    auto stats = mem_mgr.getStats();
    std::cout << "Memory allocated: " << stats.total_allocated / (1024*1024) << " MB\n";
    std::cout << "Peak usage: " << stats.peak_usage / (1024*1024) << " MB\n";
    std::cout << "Pool size: " << stats.pool_size << "\n";
    std::cout << "Shared buffers: " << stats.shared_buffers_count << "\n";
    
    // 7. Дефрагментация (при необходимости)
    if (stats.total_allocated > stats.total_available * 0.8) {
        mem_mgr.defragment();
    }
    
    // Автоматическая очистка через RAII
}
```

### Использование FFT Post-Processing модуля

```cpp
#include <gpu_lib/DrvGPU.hpp>
#include <gpu_lib/modules/FFTPostProcessing.hpp>
#include <complex>
#include <vector>

void fft_postprocessing_example() {
    auto& drv = gpu_lib::core::DrvGPU::getInstance();
    auto& mem_mgr = drv.getMemoryManager();
    auto& registry = drv.getModuleRegistry();
    
    // Получение модуля
    auto fft_module = std::dynamic_pointer_cast<gpu_lib::modules::FFTPostProcessing>(
        registry.getOrCreateModule("FFTPostProcessing")
    );
    
    // Подготовка тестовых FFT данных (например, после cuFFT)
    const size_t fft_size = 2048;
    std::vector<std::complex<float>> fft_result(fft_size);
    
    // Заполнение тестовыми данными (сигнал с 3 частотами)
    for (size_t i = 0; i < fft_size; ++i) {
        float magnitude = 0.0f;
        // Пик 1 на частоте 100 Hz
        if (i >= 95 && i <= 105) {
            magnitude = 10.0f * (1.0f - std::abs(static_cast<int>(i) - 100) / 5.0f);
        }
        // Пик 2 на частоте 500 Hz
        if (i >= 495 && i <= 505) {
            magnitude = 15.0f * (1.0f - std::abs(static_cast<int>(i) - 500) / 5.0f);
        }
        // Пик 3 на частоте 1000 Hz
        if (i >= 995 && i <= 1005) {
            magnitude = 8.0f * (1.0f - std::abs(static_cast<int>(i) - 1000) / 5.0f);
        }
        fft_result[i] = std::complex<float>(magnitude, 0.0f);
    }
    
    // Копирование на GPU
    auto gpu_fft_buffer = mem_mgr.allocate(fft_size * sizeof(std::complex<float>));
    gpu_fft_buffer->copyFrom(fft_result.data(), fft_size * sizeof(std::complex<float>));
    
    // Конфигурация поиска пиков
    gpu_lib::modules::FFTPostProcessing::Config config;
    config.enable_parabolic_interpolation = true;
    config.threshold = 1.0f;  // Порог для поиска
    config.max_peaks = 3;     // Топ-3 пика
    
    // Поиск топ-3 пиков
    auto peaks = fft_module->findTopPeaks(gpu_fft_buffer, fft_size, config);
    
    std::cout << "Found " << peaks.size() << " peaks:\n";
    for (const auto& peak : peaks) {
        std::cout << "  Frequency: " << peak.frequency << " Hz\n";
        std::cout << "    Magnitude: " << peak.magnitude << "\n";
        std::cout << "    Bin Index: " << peak.bin_index << "\n";
        std::cout << "    Interpolated Freq: " << peak.interpolated_freq << " Hz\n";
        std::cout << "    Phase: " << peak.phase << " rad\n";
    }
    
    // Поиск всех локальных максимумов выше порога
    auto all_peaks = fft_module->findAllPeaks(gpu_fft_buffer, fft_size, 2.0f);
    std::cout << "\nAll peaks above threshold: " << all_peaks.size() << "\n";
    
    // Уточнение конкретного пика
    if (!peaks.empty()) {
        float sample_rate = 48000.0f;  // 48 kHz
        float refined_freq = fft_module->refineFrequency(
            gpu_fft_buffer, 
            peaks[0].bin_index,
            sample_rate
        );
        std::cout << "Refined frequency for peak 0: " << refined_freq << " Hz\n";
    }
}
```

---

## Создание модуля

### Пример: Signal Statistics Module

```cpp
// modules/signal_stats/include/gpu_lib/modules/SignalStatistics.hpp
#pragma once

#include <gpu_lib/IComputeModule.hpp>
#include <gpu_lib/MemoryManager.hpp>

namespace gpu_lib {
namespace modules {

/**
 * @brief Модуль для вычисления статистических характеристик сигнала
 */
class SignalStatistics : public ComputeModuleBase {
public:
    SignalStatistics() 
        : ComputeModuleBase("SignalStatistics", "1.0.0") {}
    
    virtual ~SignalStatistics() = default;
    
    /**
     * @brief Вычисление среднего значения
     */
    float computeMean(const core::GPUMemoryHandle& buffer, size_t size);
    
    /**
     * @brief Вычисление медианы
     */
    float computeMedian(const core::GPUMemoryHandle& buffer, size_t size);
    
    /**
     * @brief Вычисление магнитуды комплексного сигнала
     */
    void computeMagnitude(
        const core::GPUMemoryHandle& complex_buffer,
        core::GPUMemoryHandle& magnitude_buffer,
        size_t size
    );
    
    /**
     * @brief Вычисление RMS (Root Mean Square)
     */
    float computeRMS(const core::GPUMemoryHandle& buffer, size_t size);
    
    /**
     * @brief Вычисление дисперсии
     */
    float computeVariance(const core::GPUMemoryHandle& buffer, size_t size, float mean = -1.0f);
    
    /**
     * @brief Вычисление стандартного отклонения
     */
    float computeStdDev(const core::GPUMemoryHandle& buffer, size_t size);
    
    /**
     * @brief Batch обработка всех статистик за один проход
     */
    struct Statistics {
        float mean;
        float median;
        float rms;
        float variance;
        float std_dev;
        float min;
        float max;
    };
    
    Statistics computeAll(const core::GPUMemoryHandle& buffer, size_t size);
    
protected:
    void onInitialize() override;
    void onShutdown() override;
    
private:
    // Kernels
    std::unique_ptr<backend::IKernel> mean_kernel_;
    std::unique_ptr<backend::IKernel> median_kernel_;
    std::unique_ptr<backend::IKernel> magnitude_kernel_;
    std::unique_ptr<backend::IKernel> rms_kernel_;
    std::unique_ptr<backend::IKernel> variance_kernel_;
    std::unique_ptr<backend::IKernel> reduce_kernel_;
    std::unique_ptr<backend::IKernel> sort_kernel_;
    
    // Temporary buffers для reduction операций
    core::GPUMemoryHandle temp_buffer_;
    core::GPUMemoryHandle sorted_buffer_;
    
    // Максимальный размер для оптимизации
    size_t max_buffer_size_ = 0;
    
    void compileKernels();
    void createTempBuffers(size_t max_size);
    
    // Helper: Parallel reduction на GPU
    float parallelReduce(
        const core::GPUMemoryHandle& input,
        size_t size,
        backend::IKernel& kernel
    );
};

} // namespace modules
} // namespace gpu_lib
```

### Реализация модуля

```cpp
// modules/signal_stats/src/SignalStatistics.cpp
#include <gpu_lib/modules/SignalStatistics.hpp>
#include <cmath>
#include <algorithm>

namespace gpu_lib {
namespace modules {

void SignalStatistics::onInitialize() {
    compileKernels();
    // Temp buffers создаются по требованию
}

void SignalStatistics::onShutdown() {
    // Kernels автоматически удаляются через unique_ptr
    // Temp buffers удаляются через RAII
}

void SignalStatistics::compileKernels() {
    // Загрузка OpenCL кода (обычно из embedded string или файла)
    const char* kernel_source = R"CL(
        // Mean kernel - parallel sum reduction
        __kernel void compute_sum(
            __global const float* input,
            __global float* partial_sums,
            const int size
        ) {
            int gid = get_global_id(0);
            int lid = get_local_id(0);
            int group_size = get_local_size(0);
            
            __local float local_sum[256];
            
            // Каждый work-item суммирует свою часть
            float sum = 0.0f;
            for (int i = gid; i < size; i += get_global_size(0)) {
                sum += input[i];
            }
            local_sum[lid] = sum;
            
            barrier(CLK_LOCAL_MEM_FENCE);
            
            // Tree reduction в local memory
            for (int s = group_size / 2; s > 0; s >>= 1) {
                if (lid < s) {
                    local_sum[lid] += local_sum[lid + s];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            
            // Первый work-item записывает результат группы
            if (lid == 0) {
                partial_sums[get_group_id(0)] = local_sum[0];
            }
        }
        
        // Magnitude kernel
        __kernel void compute_magnitude(
            __global const float2* complex_input,
            __global float* magnitude_output,
            const int size
        ) {
            int gid = get_global_id(0);
            if (gid >= size) return;
            
            float2 c = complex_input[gid];
            magnitude_output[gid] = sqrt(c.x * c.x + c.y * c.y);
        }
        
        // RMS kernel - sum of squares
        __kernel void compute_sum_of_squares(
            __global const float* input,
            __global float* partial_sums,
            const int size
        ) {
            int gid = get_global_id(0);
            int lid = get_local_id(0);
            int group_size = get_local_size(0);
            
            __local float local_sum[256];
            
            float sum = 0.0f;
            for (int i = gid; i < size; i += get_global_size(0)) {
                float val = input[i];
                sum += val * val;
            }
            local_sum[lid] = sum;
            
            barrier(CLK_LOCAL_MEM_FENCE);
            
            for (int s = group_size / 2; s > 0; s >>= 1) {
                if (lid < s) {
                    local_sum[lid] += local_sum[lid + s];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            
            if (lid == 0) {
                partial_sums[get_group_id(0)] = local_sum[0];
            }
        }
        
        // Variance kernel
        __kernel void compute_variance_sum(
            __global const float* input,
            __global float* partial_sums,
            const float mean,
            const int size
        ) {
            int gid = get_global_id(0);
            int lid = get_local_id(0);
            int group_size = get_local_size(0);
            
            __local float local_sum[256];
            
            float sum = 0.0f;
            for (int i = gid; i < size; i += get_global_size(0)) {
                float diff = input[i] - mean;
                sum += diff * diff;
            }
            local_sum[lid] = sum;
            
            barrier(CLK_LOCAL_MEM_FENCE);
            
            for (int s = group_size / 2; s > 0; s >>= 1) {
                if (lid < s) {
                    local_sum[lid] += local_sum[lid + s];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            
            if (lid == 0) {
                partial_sums[get_group_id(0)] = local_sum[0];
            }
        }
    )CL";
    
    // Компиляция kernels
    mean_kernel_ = backend().createKernel(kernel_source, "compute_sum");
    rms_kernel_ = backend().createKernel(kernel_source, "compute_sum_of_squares");
    magnitude_kernel_ = backend().createKernel(kernel_source, "compute_magnitude");
    variance_kernel_ = backend().createKernel(kernel_source, "compute_variance_sum");
}

float SignalStatistics::computeMean(
    const core::GPUMemoryHandle& buffer, 
    size_t size
) {
    if (!isInitialized()) {
        throw std::runtime_error("Module not initialized");
    }
    
    // Parallel reduction для вычисления суммы
    float sum = parallelReduce(buffer, size, *mean_kernel_);
    
    return sum / static_cast<float>(size);
}

void SignalStatistics::computeMagnitude(
    const core::GPUMemoryHandle& complex_buffer,
    core::GPUMemoryHandle& magnitude_buffer,
    size_t size
) {
    // Установка аргументов kernel
    magnitude_kernel_->setArgBuffer(0, *complex_buffer.get());
    magnitude_kernel_->setArgBuffer(1, *magnitude_buffer.get());
    magnitude_kernel_->setArg(2, static_cast<int>(size));
    
    // Вычисление work sizes
    size_t local_size = 256;
    size_t global_size = ((size + local_size - 1) / local_size) * local_size;
    
    // Запуск kernel
    magnitude_kernel_->execute(
        backend::WorkSize(global_size),
        backend::WorkSize(local_size)
    );
    
    // Синхронизация (опционально, если нужен результат сразу)
    backend().synchronize();
}

float SignalStatistics::computeRMS(
    const core::GPUMemoryHandle& buffer,
    size_t size
) {
    // Parallel reduction для суммы квадратов
    float sum_of_squares = parallelReduce(buffer, size, *rms_kernel_);
    
    return std::sqrt(sum_of_squares / static_cast<float>(size));
}

float SignalStatistics::computeVariance(
    const core::GPUMemoryHandle& buffer,
    size_t size,
    float mean
) {
    // Если mean не передан, вычисляем
    if (mean < 0.0f) {
        mean = computeMean(buffer, size);
    }
    
    // Устанавливаем mean как аргумент kernel
    variance_kernel_->setArgBuffer(0, *buffer.get());
    
    // Создаем temporary buffer для partial sums
    size_t num_groups = (size + 255) / 256;
    if (!temp_buffer_.get() || temp_buffer_.size() < num_groups * sizeof(float)) {
        temp_buffer_ = memoryManager().allocate(num_groups * sizeof(float));
    }
    
    variance_kernel_->setArgBuffer(1, *temp_buffer_.get());
    variance_kernel_->setArg(2, mean);
    variance_kernel_->setArg(3, static_cast<int>(size));
    
    // Запуск
    variance_kernel_->execute(
        backend::WorkSize((size + 255) / 256 * 256),
        backend::WorkSize(256)
    );
    
    // Reduction partial sums на CPU (или второй pass на GPU)
    std::vector<float> partial_sums(num_groups);
    temp_buffer_->copyTo(partial_sums.data(), num_groups * sizeof(float));
    
    float sum = 0.0f;
    for (float val : partial_sums) {
        sum += val;
    }
    
    return sum / static_cast<float>(size);
}

float SignalStatistics::computeStdDev(
    const core::GPUMemoryHandle& buffer,
    size_t size
) {
    return std::sqrt(computeVariance(buffer, size));
}

SignalStatistics::Statistics SignalStatistics::computeAll(
    const core::GPUMemoryHandle& buffer,
    size_t size
) {
    Statistics stats;
    
    // Вычисляем mean один раз
    stats.mean = computeMean(buffer, size);
    
    // RMS
    stats.rms = computeRMS(buffer, size);
    
    // Variance и std_dev (используем уже вычисленный mean)
    stats.variance = computeVariance(buffer, size, stats.mean);
    stats.std_dev = std::sqrt(stats.variance);
    
    // Median требует сортировки - самая дорогая операция
    stats.median = computeMedian(buffer, size);
    
    // Min/Max можно добавить через отдельный kernel
    // TODO: Implement min/max kernel
    stats.min = 0.0f;
    stats.max = 0.0f;
    
    return stats;
}

float SignalStatistics::parallelReduce(
    const core::GPUMemoryHandle& input,
    size_t size,
    backend::IKernel& kernel
) {
    // Two-pass reduction
    // Pass 1: Reduce большой массив до небольшого количества partial sums
    size_t local_size = 256;
    size_t num_groups = (size + local_size - 1) / local_size;
    
    if (!temp_buffer_.get() || temp_buffer_.size() < num_groups * sizeof(float)) {
        temp_buffer_ = memoryManager().allocate(num_groups * sizeof(float));
    }
    
    kernel.setArgBuffer(0, *input.get());
    kernel.setArgBuffer(1, *temp_buffer_.get());
    kernel.setArg(2, static_cast<int>(size));
    
    kernel.execute(
        backend::WorkSize(num_groups * local_size),
        backend::WorkSize(local_size)
    );
    
    // Pass 2: Final reduction на CPU (или повторить на GPU если num_groups большой)
    std::vector<float> partial_sums(num_groups);
    temp_buffer_->copyTo(partial_sums.data(), num_groups * sizeof(float));
    
    float result = 0.0f;
    for (float val : partial_sums) {
        result += val;
    }
    
    return result;
}

} // namespace modules
} // namespace gpu_lib
```

### Unit Test для модуля

```cpp
// modules/signal_stats/tests/test_signal_statistics.cpp
#include <gtest/gtest.h>
#include <gpu_lib/DrvGPU.hpp>
#include <gpu_lib/modules/SignalStatistics.hpp>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

class SignalStatisticsTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto& drv = gpu_lib::core::DrvGPU::getInstance();
        
        if (!drv.isInitialized()) {
            drv.initialize(gpu_lib::BackendType::OpenCL, 0);
        }
        
        auto& registry = drv.getModuleRegistry();
        registry.registerModule("SignalStatistics", []() {
            return std::make_unique<gpu_lib::modules::SignalStatistics>();
        });
        
        module_ = std::dynamic_pointer_cast<gpu_lib::modules::SignalStatistics>(
            registry.getOrCreateModule("SignalStatistics")
        );
        
        ASSERT_NE(module_, nullptr);
        ASSERT_TRUE(module_->isInitialized());
    }
    
    void TearDown() override {
        module_.reset();
    }
    
    std::shared_ptr<gpu_lib::modules::SignalStatistics> module_;
};

TEST_F(SignalStatisticsTest, ComputeMean_SimpleArray) {
    auto& drv = gpu_lib::core::DrvGPU::getInstance();
    auto& mem_mgr = drv.getMemoryManager();
    
    // Подготовка данных
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float expected_mean = 3.0f;
    
    // Копирование на GPU
    auto gpu_buffer = mem_mgr.allocate(data.size() * sizeof(float));
    gpu_buffer->copyFrom(data.data(), data.size() * sizeof(float));
    
    // Вычисление
    float result = module_->computeMean(gpu_buffer, data.size());
    
    // Проверка
    EXPECT_NEAR(result, expected_mean, 1e-5);
}

TEST_F(SignalStatisticsTest, ComputeMagnitude) {
    auto& drv = gpu_lib::core::DrvGPU::getInstance();
    auto& mem_mgr = drv.getMemoryManager();
    
    // Комплексные данные
    std::vector<std::complex<float>> complex_data = {
        {3.0f, 4.0f},   // magnitude = 5.0
        {1.0f, 0.0f},   // magnitude = 1.0
        {0.0f, 1.0f},   // magnitude = 1.0
        {-3.0f, -4.0f}  // magnitude = 5.0
    };
    
    std::vector<float> expected_magnitudes = {5.0f, 1.0f, 1.0f, 5.0f};
    
    // GPU buffers
    auto complex_buffer = mem_mgr.allocate(complex_data.size() * sizeof(std::complex<float>));
    auto magnitude_buffer = mem_mgr.allocate(complex_data.size() * sizeof(float));
    
    complex_buffer->copyFrom(complex_data.data(), complex_data.size() * sizeof(std::complex<float>));
    
    // Compute
    module_->computeMagnitude(complex_buffer, magnitude_buffer, complex_data.size());
    
    // Verify
    std::vector<float> result(complex_data.size());
    magnitude_buffer->copyTo(result.data(), complex_data.size() * sizeof(float));
    
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_NEAR(result[i], expected_magnitudes[i], 1e-5) << "at index " << i;
    }
}

TEST_F(SignalStatisticsTest, ComputeRMS) {
    auto& drv = gpu_lib::core::DrvGPU::getInstance();
    auto& mem_mgr = drv.getMemoryManager();
    
    // RMS для синусоиды amplitude A = A / sqrt(2)
    std::vector<float> sine_wave(1000);
    const float amplitude = 10.0f;
    for (size_t i = 0; i < sine_wave.size(); ++i) {
        sine_wave[i] = amplitude * std::sin(2.0f * M_PI * i / sine_wave.size());
    }
    
    float expected_rms = amplitude / std::sqrt(2.0f);
    
    auto gpu_buffer = mem_mgr.allocate(sine_wave.size() * sizeof(float));
    gpu_buffer->copyFrom(sine_wave.data(), sine_wave.size() * sizeof(float));
    
    float rms = module_->computeRMS(gpu_buffer, sine_wave.size());
    
    EXPECT_NEAR(rms, expected_rms, expected_rms * 0.01);  // 1% tolerance
}
```

---

## Pipeline обработки

### Создание Pipeline Builder

```cpp
// core/include/gpu_lib/Pipeline.hpp
#pragma once

#include <gpu_lib/IComputeModule.hpp>
#include <gpu_lib/MemoryManager.hpp>
#include <functional>
#include <vector>
#include <memory>

namespace gpu_lib {
namespace core {

/**
 * @brief Builder для создания pipeline обработки
 * 
 * Design Pattern: Builder
 * Позволяет строить сложные цепочки операций декларативно
 */
class Pipeline {
public:
    using StageFunction = std::function<void()>;
    
    struct Stage {
        std::string name;
        StageFunction function;
        std::vector<std::string> inputs;   // Имена input буферов
        std::vector<std::string> outputs;  // Имена output буферов
    };
    
    Pipeline(MemoryManager& memory_manager)
        : memory_manager_(memory_manager) {}
    
    /**
     * @brief Добавление стадии в pipeline
     */
    Pipeline& addStage(
        const std::string& name,
        StageFunction function,
        const std::vector<std::string>& inputs = {},
        const std::vector<std::string>& outputs = {}
    ) {
        stages_.push_back(Stage{name, function, inputs, outputs});
        return *this;
    }
    
    /**
     * @brief Регистрация буфера в pipeline
     */
    Pipeline& registerBuffer(const std::string& name, GPUMemoryHandle buffer) {
        buffers_[name] = std::move(buffer);
        return *this;
    }
    
    /**
     * @brief Получение буфера по имени
     */
    GPUMemoryHandle& getBuffer(const std::string& name) {
        auto it = buffers_.find(name);
        if (it == buffers_.end()) {
            throw std::runtime_error("Buffer not found: " + name);
        }
        return it->second;
    }
    
    /**
     * @brief Выполнение всего pipeline
     */
    void execute() {
        for (const auto& stage : stages_) {
            try {
                stage.function();
            } catch (const std::exception& e) {
                throw std::runtime_error(
                    "Pipeline stage '" + stage.name + "' failed: " + e.what()
                );
            }
        }
    }
    
    /**
     * @brief Получение списка стадий
     */
    const std::vector<Stage>& getStages() const { return stages_; }
    
    /**
     * @brief Очистка pipeline
     */
    void clear() {
        stages_.clear();
        buffers_.clear();
    }
    
private:
    MemoryManager& memory_manager_;
    std::vector<Stage> stages_;
    std::unordered_map<std::string, GPUMemoryHandle> buffers_;
};

} // namespace core
} // namespace gpu_lib
```

---

## Переключение бэкендов

### Runtime Backend Switching

```cpp
#include <gpu_lib/DrvGPU.hpp>
#include <iostream>

void backend_switching_example() {
    auto& drv = gpu_lib::core::DrvGPU::getInstance();
    
    // Инициализация с OpenCL
    std::cout << "Initializing with OpenCL...\n";
    drv.initialize(gpu_lib::BackendType::OpenCL, 0);
    
    auto info = drv.getDeviceInfo();
    std::cout << "OpenCL Device: " << info.name << "\n";
    std::cout << "Backend: " << drv.getBackend().getBackendName() << "\n";
    
    // Работа с OpenCL
    auto& mem_mgr = drv.getMemoryManager();
    auto buffer1 = mem_mgr.allocate(1024 * 1024);  // 1MB
    
    std::vector<float> data(1024 * 1024 / sizeof(float), 42.0f);
    buffer1->copyFrom(data.data(), data.size() * sizeof(float));
    
    std::cout << "Memory allocated on OpenCL: " 
              << mem_mgr.getStats().total_allocated / (1024*1024) << " MB\n";
    
    // Переключение на ROCm (если доступен)
    if (gpu_lib::backend::BackendFactory::isBackendAvailable(gpu_lib::BackendType::ROCm)) {
        std::cout << "\nSwitching to ROCm backend...\n";
        
        // ВАЖНО: switchBackend() пересоздает все буферы!
        // Старые handles становятся невалидными
        drv.switchBackend(gpu_lib::BackendType::ROCm);
        
        info = drv.getDeviceInfo();
        std::cout << "ROCm Device: " << info.name << "\n";
        std::cout << "Backend: " << drv.getBackend().getBackendName() << "\n";
        
        // Нужно пересоздать буферы
        auto buffer2 = mem_mgr.allocate(1024 * 1024);
        buffer2->copyFrom(data.data(), data.size() * sizeof(float));
        
        std::cout << "Memory allocated on ROCm: "
                  << mem_mgr.getStats().total_allocated / (1024*1024) << " MB\n";
        
        // Модули автоматически пересоздают свои kernels при переключении
        auto& registry = drv.getModuleRegistry();
        auto module = registry.getOrCreateModule("FFTPostProcessing");
        
        // Модуль готов к работе с новым бэкендом
        std::cout << "Module " << module->getName() << " ready with ROCm\n";
    } else {
        std::cout << "\nROCm backend not available, staying with OpenCL\n";
    }
    
    drv.shutdown();
}
```

---

## Оптимизация производительности

### Асинхронное выполнение

```cpp
// Пример асинхронного pipeline с перекрытием CPU-GPU
#include <gpu_lib/DrvGPU.hpp>
#include <future>

void async_pipeline_example() {
    auto& drv = gpu_lib::core::DrvGPU::getInstance();
    auto& backend = drv.getBackend();
    
    // Создание нескольких command queues для параллелизма
    auto queue1 = backend.createCommandQueue();
    auto queue2 = backend.createCommandQueue();
    
    // Подготовка данных
    auto& mem_mgr = drv.getMemoryManager();
    auto input1 = mem_mgr.allocate(1024 * 1024);
    auto input2 = mem_mgr.allocate(1024 * 1024);
    auto output1 = mem_mgr.allocate(1024 * 1024);
    auto output2 = mem_mgr.allocate(1024 * 1024);
    
    // Kernel для обработки
    const char* kernel_src = R"CL(
        __kernel void process(__global float* input, __global float* output, int size) {
            int gid = get_global_id(0);
            if (gid < size) {
                output[gid] = input[gid] * 2.0f;  // Пример обработки
            }
        }
    )CL";
    
    auto kernel = backend.createKernel(kernel_src, "process");
    
    // Асинхронное выполнение в двух очередях
    auto future1 = std::async(std::launch::async, [&]() {
        kernel->setArgBuffer(0, *input1.get());
        kernel->setArgBuffer(1, *output1.get());
        kernel->setArg(2, 1024 * 1024 / sizeof(float));
        
        queue1->enqueueKernel(*kernel, 
            gpu_lib::backend::WorkSize(1024 * 1024 / sizeof(float)),
            gpu_lib::backend::WorkSize(256)
        );
        
        queue1->finish();
        std::cout << "Queue 1 finished\n";
    });
    
    auto future2 = std::async(std::launch::async, [&]() {
        kernel->setArgBuffer(0, *input2.get());
        kernel->setArgBuffer(1, *output2.get());
        kernel->setArg(2, 1024 * 1024 / sizeof(float));
        
        queue2->enqueueKernel(*kernel,
            gpu_lib::backend::WorkSize(1024 * 1024 / sizeof(float)),
            gpu_lib::backend::WorkSize(256)
        );
        
        queue2->finish();
        std::cout << "Queue 2 finished\n";
    });
    
    // Ожидание завершения обеих очередей
    future1.wait();
    future2.wait();
    
    std::cout << "All async operations completed\n";
}
```

---

## Примеры OpenCL kernels

### Fractional Delay Kernel (Farrow Structure)

```opencl
// Дробная задержка через структуру Фарроу
// Использует кубическую интерполяцию для subsample точности

__kernel void fractional_delay_farrow(
    __global const float* input,
    __global float* output,
    const float delay_samples,  // Может быть дробным, например 2.75
    const int size
) {
    int gid = get_global_id(0);
    if (gid >= size) return;
    
    // Разделение delay на целую и дробную части
    int delay_int = (int)floor(delay_samples);
    float delay_frac = delay_samples - delay_int;
    
    // Индекс входного сэмпла с учетом целой задержки
    int input_idx = gid - delay_int;
    
    if (input_idx < 0 || input_idx + 3 >= size) {
        output[gid] = 0.0f;  // Zero padding
        return;
    }
    
    // Получение 4 точек для кубической интерполяции
    float x0 = input[input_idx];
    float x1 = input[input_idx + 1];
    float x2 = input[input_idx + 2];
    float x3 = input[input_idx + 3];
    
    // Farrow structure coefficients для кубической интерполяции
    float a0 = x1;
    float a1 = 0.5f * (x2 - x0);
    float a2 = x0 - 2.5f * x1 + 2.0f * x2 - 0.5f * x3;
    float a3 = 0.5f * (x3 - x0) + 1.5f * (x1 - x2);
    
    // Вычисление выходного значения через Горнера
    float result = a0 + delay_frac * (a1 + delay_frac * (a2 + delay_frac * a3));
    
    output[gid] = result;
}
```

### Heterodyne (Complex Mixer)

```opencl
// Гетеродинное преобразование: умножение на комплексный NCO сигнал
// Используется для сдвига частоты сигнала

__kernel void heterodyne_mixer(
    __global const float2* input,      // Комплексный входной сигнал
    __global float2* output,           // Комплексный выходной сигнал
    const float center_freq,           // Центральная частота в Hz
    const float sample_rate,           // Частота дискретизации в Hz
    const float phase_offset,          // Начальная фаза в радианах
    const int size
) {
    int gid = get_global_id(0);
    if (gid >= size) return;
    
    // Вычисление фазы для текущего сэмпла
    float phase = 2.0f * M_PI_F * center_freq * gid / sample_rate + phase_offset;
    
    // Генерация NCO (Numerically Controlled Oscillator)
    float nco_real = cos(phase);
    float nco_imag = sin(phase);
    
    // Комплексное умножение: output = input * nco
    float2 in = input[gid];
    
    float out_real = in.x * nco_real - in.y * nco_imag;
    float out_imag = in.x * nco_imag + in.y * nco_real;
    
    output[gid] = (float2)(out_real, out_imag);
}
```

### Parabolic Interpolation для FFT пиков

```opencl
// Параболическая интерполяция для субсэмпловой точности пика FFT
// Использует 3 точки вокруг пика для вычисления вершины параболы

__kernel void parabolic_interpolation(
    __global const float* magnitude,   // Магнитуда FFT
    __global const int* peak_indices,  // Индексы найденных пиков
    __global float* refined_bins,      // Уточненные позиции бинов (дробные)
    const int num_peaks,
    const int fft_size
) {
    int peak_id = get_global_id(0);
    if (peak_id >= num_peaks) return;
    
    int peak_idx = peak_indices[peak_id];
    
    // Проверка границ
    if (peak_idx <= 0 || peak_idx >= fft_size - 1) {
        refined_bins[peak_id] = (float)peak_idx;
        return;
    }
    
    // Три точки вокруг пика
    float y_minus = magnitude[peak_idx - 1];
    float y_zero = magnitude[peak_idx];
    float y_plus = magnitude[peak_idx + 1];
    
    // Формула параболической интерполяции
    float numerator = y_minus - y_plus;
    float denominator = y_minus - 2.0f * y_zero + y_plus;
    
    float delta = 0.0f;
    if (fabs(denominator) > 1e-6f) {
        delta = 0.5f * numerator / denominator;
        
        // Ограничение delta в разумных пределах
        delta = clamp(delta, -0.5f, 0.5f);
    }
    
    // Уточненная позиция бина
    refined_bins[peak_id] = peak_idx + delta;
}
```

### Signal Statistics - Parallel Reduction

```opencl
// Параллельная редукция для вычисления суммы (используется для mean)
// Two-stage reduction: local memory -> global memory

__kernel void parallel_sum_stage1(
    __global const float* input,
    __global float* partial_sums,
    __local float* local_sums,
    const int size
) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int group_id = get_group_id(0);
    int local_size = get_local_size(0);
    
    // Каждый work-item суммирует свою часть
    float sum = 0.0f;
    for (int i = gid; i < size; i += get_global_size(0)) {
        sum += input[i];
    }
    
    local_sums[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Tree reduction в local memory
    for (int stride = local_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_sums[lid] += local_sums[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Первый work-item записывает результат группы
    if (lid == 0) {
        partial_sums[group_id] = local_sums[0];
    }
}
```

Эти примеры покрывают основные паттерны использования библиотеки и показывают, как создавать расширяемую архитектуру с хорошими практиками ООП и GPU программирования!
