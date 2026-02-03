/**
 * @file test_search_3_module.hpp
 * @brief Тесты для Antenna FFT Module
 * 
 * Тест 1: CPU -> SVM -> GPU (zero-copy)
 * Тест 2: CPU -> External OpenCL -> Import to DrvGPU -> GPU
 */

#include "search_3_module.hpp"
#include "drv_gpu.hpp"
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <iomanip>

namespace test_find_3_max {


// ═══════════════════════════════════════════════════════════════════════════
// Параметры теста
// ═══════════════════════════════════════════════════════════════════════════

const size_t NUM_BEAMS = 10;           // Количество лучей (антенн)
const size_t POINTS_PER_BEAM = 1500;   // Точек на луч
const size_t FFT_EXPAND_FACTOR = 2;    // Коэффициент расширения FFT
const size_t OUT_COUNT_FFT = 1000;     // Точек FFT для анализа
const size_t MAX_PEAKS = 3;            // Максимумов для поиска

const float BASE_FREQUENCY_FACTOR = 3.0f;  // w0 = points_per_beam / 3
const float FREQUENCY_MULTIPLIER = 1.5f;   // Увеличение частоты на луч
const float PHASE_SHIFT_DEGREES = 5.0f;    // Сдвиг фазы на луч (градусы)

// ═══════════════════════════════════════════════════════════════════════════
// Генерация синусоидального сигнала
// ═══════════════════════════════════════════════════════════════════════════

/**
 * @brief Сгенерировать комплексный синусоидальный сигнал
 * 
 * Формула для луча i:
 * - Частота: w_i = w0 * (1.5^i), где w0 = points_per_beam / 3
 * - Фаза: phi_i = i * 5° (в радианах)
 * - Сигнал: signal[i][t] = exp(j * (2π * t * w_i / points_per_beam + phi_i))
 * 
 * @param num_beams Количество лучей
 * @param points_per_beam Точек на луч
 * @return Вектор комплексных чисел [num_beams * points_per_beam]
 */
std::vector<std::complex<float>> GenerateSinusoidSignal(
    size_t num_beams, 
    size_t points_per_beam) 
{
    std::vector<std::complex<float>> signal(num_beams * points_per_beam);
    
    const float base_w0 = static_cast<float>(points_per_beam) / BASE_FREQUENCY_FACTOR;
    const float phase_shift_rad = PHASE_SHIFT_DEGREES * M_PI / 180.0f;
    
    std::cout << "  Генерация сигнала:\n";
    std::cout << "    - Базовая частота w0: " << base_w0 << " отсчётов\n";
    std::cout << "    - Сдвиг фазы: " << PHASE_SHIFT_DEGREES << "° (" 
              << phase_shift_rad << " рад)\n\n";
    
    for (size_t beam = 0; beam < num_beams; ++beam) {
        // Частота для текущего луча
        float w_beam = base_w0 * std::pow(FREQUENCY_MULTIPLIER, static_cast<float>(beam));
        
        // Фаза для текущего луча
        float phi_beam = static_cast<float>(beam) * phase_shift_rad;
        
        std::cout << "    Луч " << beam << ": w=" << std::fixed << std::setprecision(2) 
                  << w_beam << ", φ=" << (phi_beam * 180.0f / M_PI) << "°\n";
        
        // Генерировать точки для луча
        for (size_t t = 0; t < points_per_beam; ++t) {
            // Угол: 2π * t * w_beam / points_per_beam + phi_beam
            float angle = 2.0f * M_PI * static_cast<float>(t) * w_beam / 
                         static_cast<float>(points_per_beam) + phi_beam;
            
            // Комплексная экспонента: e^(j*angle) = cos(angle) + j*sin(angle)
            float real = std::cos(angle);
            float imag = std::sin(angle);
            
            signal[beam * points_per_beam + t] = std::complex<float>(real, imag);
        }
    }
    
    std::cout << "\n";
    return signal;
}

// ═══════════════════════════════════════════════════════════════════════════
// Вывод результатов
// ═══════════════════════════════════════════════════════════════════════════

void PrintResults(const drv_gpu_lib::search_3_::Search3FFTResult& result) {
    std::cout << "\n┌─────────────────────────────────────────────────────────────┐\n";
    std::cout << "│ РЕЗУЛЬТАТЫ FFT ОБРАБОТКИ                                    │\n";
    std::cout << "└─────────────────────────────────────────────────────────────┘\n\n";
    
    std::cout << "  nFFT используемый: " << result.nFFT << "\n";
    std::cout << "  Обработано лучей: " << result.results.size() << "\n\n";
    
    std::cout << "┌────────┬─────────┬────────────────┬─────────────┬────────────────┐\n";
    std::cout << "│  Луч   │ Peak #  │   Amplitude    │  Phase (°)  │     Index      │\n";
    std::cout << "├────────┼─────────┼────────────────┼─────────────┼────────────────┤\n";
    
    for (size_t beam = 0; beam < result.results.size(); ++beam) {
        const auto& beam_result = result.results[beam];
        
        if (beam_result.max_values.empty()) {
            std::cout << "│ " << std::setw(6) << beam << " │   N/A   │      N/A       │     N/A     │      N/A       │\n";
        } else {
            for (size_t peak = 0; peak < beam_result.max_values.size(); ++peak) {
                const auto& max_val = beam_result.max_values[peak];
                
                if (peak == 0) {
                    std::cout << "│ " << std::setw(6) << beam;
                } else {
                    std::cout << "│        ";
                }
                
                float phase_deg = max_val.phase * 180.0f / M_PI;
                
                std::cout << " │ " << std::setw(7) << (peak + 1)
                          << " │ " << std::setw(14) << std::fixed << std::setprecision(4) 
                          << max_val.amplitude
                          << " │ " << std::setw(11) << std::setprecision(2) << phase_deg
                          << " │ " << std::setw(14) << max_val.index_point
                          << " │\n";
            }
        }
        
        if (beam < result.results.size() - 1) {
            std::cout << "├────────┼─────────┼────────────────┼─────────────┼────────────────┤\n";
        }
    }
    
    std::cout << "└────────┴─────────┴────────────────┴─────────────┴────────────────┘\n\n";
}

// ═══════════════════════════════════════════════════════════════════════════
// ТЕСТ 1: CPU -> SVM -> GPU (Zero-Copy)
// ═══════════════════════════════════════════════════════════════════════════

void test_search_3_module_svm() {
    std::cout << "\n╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ ТЕСТ 1: CPU -> SVM -> GPU (Zero-Copy)                        ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n\n";
    
    try {
        // ═══ 1. Инициализация DrvGPU ═══
        std::cout << "═══ Шаг 1: Инициализация DrvGPU ═══\n";
        drv_gpu_lib::DrvGPU gpu(drv_gpu_lib::BackendType::OPENCL, 0);
        gpu.Initialize();
        std::cout << "✅ DrvGPU инициализирован\n\n";
        
        // ═══ 2. Создать Search3 модуль ═══
        std::cout << "═══ Шаг 2: Создание Search3 модуля ═══\n";
        drv_gpu_lib::search_3_::Search3Params params(
            NUM_BEAMS,
            POINTS_PER_BEAM,
            OUT_COUNT_FFT,
            MAX_PEAKS,
            "test_svm",
            "search_3_test"
        );
        
        std::cout << "  Параметры:\n";
        std::cout << "    - Лучей: " << NUM_BEAMS << "\n";
        std::cout << "    - Точек/луч: " << POINTS_PER_BEAM << "\n";
        std::cout << "    - FFT expand: " << FFT_EXPAND_FACTOR << "x\n";
        std::cout << "    - Out FFT: " << OUT_COUNT_FFT << "\n";
        std::cout << "    - Max peaks: " << MAX_PEAKS << "\n\n";
        
        auto search_3_module = std::make_shared<drv_gpu_lib::search_3_::Search3Module>(
            &gpu.GetBackend(), params);
        search_3_module->Initialize();
        std::cout << "✅ Search3 модуль инициализирован\n";
        std::cout << "   nFFT вычислен: " << search_3_module->GetNFFT() << "\n\n";
        
        // ═══ 3. Генерация сигнала на CPU ═══
        std::cout << "═══ Шаг 3: Генерация синусоидального сигнала на CPU ═══\n";
        std::vector<std::complex<float>> cpu_signal = 
            GenerateSinusoidSignal(NUM_BEAMS, POINTS_PER_BEAM);
        std::cout << "✅ Сигнал сгенерирован: " << cpu_signal.size() 
                  << " комплексных точек\n\n";
        
        // ═══ 4. Создать GPU буфер с данными ═══
        std::cout << "═══ Шаг 4: Создание GPU буфера с данными ═══\n";
        auto& mem_mgr = gpu.GetMemoryManager();
        auto gpu_buffer = mem_mgr.CreateBuffer<std::complex<float>>(
            cpu_signal.data(), cpu_signal.size());
        std::cout << "✅ GPU буфер создан: " << gpu_buffer->GetNumElements() 
                  << " элементов\n\n";
        
        // ═══ 5. Обработка FFT на GPU ═══
        std::cout << "═══ Шаг 5: Обработка FFT на GPU (ProcessNew) ═══\n";
        auto result = search_3_module->ProcessNew(gpu_buffer);
        std::cout << "✅ FFT обработка завершена\n\n";
        
        // ═══ 7. Вывод результатов ═══
        std::cout << "═══ Шаг 7: Результаты ═══\n";
        PrintResults(result);
        
        std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
        std::cout << "║ ТЕСТ 1 ЗАВЕРШЁН УСПЕШНО ✅                                    ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════════════╝\n\n";
        
    } catch (const std::exception& e) {
        std::cerr << "❌ ОШИБКА в тесте 1: " << e.what() << "\n";
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ТЕСТ 2: CPU -> External OpenCL -> Import to DrvGPU -> GPU
// ═══════════════════════════════════════════════════════════════════════════

void test_search_3_module_external_opencl() {
    std::cout << "\n╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ ТЕСТ 2: CPU -> External OpenCL -> Import to DrvGPU -> GPU    ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n\n";
    
    try {
        // ═══ 1. Генерация сигнала на CPU ═══
        std::cout << "═══ Шаг 1: Генерация синусоидального сигнала на CPU ═══\n";
        std::vector<std::complex<float>> cpu_signal = 
            GenerateSinusoidSignal(NUM_BEAMS, POINTS_PER_BEAM);
        std::cout << "✅ Сигнал сгенерирован: " << cpu_signal.size() 
                  << " комплексных точек\n\n";
        
        // ═══ 2. Создать внешний OpenCL контекст и буфер ═══
        std::cout << "═══ Шаг 2: Создание внешнего OpenCL буфера ═══\n";
        
        // Получить платформу и устройство
        cl_platform_id platform;
        cl_device_id device;
        cl_int err;
        
        err = clGetPlatformIDs(1, &platform, nullptr);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to get OpenCL platform");
        }
        
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to get OpenCL device");
        }
        
        // Создать контекст
        cl_context external_context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to create external context");
        }
        
        // Создать command queue
        cl_command_queue external_queue = clCreateCommandQueue(external_context, device, 0, &err);
        if (err != CL_SUCCESS) {
            clReleaseContext(external_context);
            throw std::runtime_error("Failed to create external queue");
        }
        
        std::cout << "✅ Внешний OpenCL контекст создан\n";
        
        // Создать cl_mem буфер
        size_t buffer_size = cpu_signal.size() * sizeof(std::complex<float>);
        cl_mem external_buffer = clCreateBuffer(
            external_context,
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            buffer_size,
            cpu_signal.data(),
            &err);
        
        if (err != CL_SUCCESS) {
            clReleaseCommandQueue(external_queue);
            clReleaseContext(external_context);
            throw std::runtime_error("Failed to create external buffer");
        }
        
        std::cout << "✅ Внешний cl_mem буфер создан и заполнен данными\n";
        std::cout << "   Размер: " << (buffer_size / (1024 * 1024)) << " MB\n\n";
        
        // ═══ 3. Инициализация DrvGPU ═══
        std::cout << "═══ Шаг 3: Инициализация DrvGPU ═══\n";
        drv_gpu_lib::DrvGPU gpu(drv_gpu_lib::BackendType::OPENCL, 0);
        gpu.Initialize();
        std::cout << "✅ DrvGPU инициализирован\n\n";
        
        // ═══ 4. Создать Search3 модуль ═══
        std::cout << "═══ Шаг 4: Создание Search3 модуля ═══\n";
        drv_gpu_lib::search_3_::Search3Params params(
            NUM_BEAMS,
            POINTS_PER_BEAM,
            OUT_COUNT_FFT,
            MAX_PEAKS,
            "test_external_cl",
            "search_3_test"
        );
        
        auto search_3_module = std::make_shared<drv_gpu_lib::search_3_::Search3Module>(
            &gpu.GetBackend(), params);
        search_3_module->Initialize();
        std::cout << "✅ Search3 модуль инициализирован\n";
        std::cout << "   nFFT вычислен: " << search_3_module->GetNFFT() << "\n\n";
        
        // ═══ 5. ВАРИАНТ A: Передать cl_mem напрямую ═══
        std::cout << "═══ Шаг 5A: Обработка через cl_mem (прямая передача) ═══\n";
        auto result_clmem = search_3_module->ProcessNew(external_buffer);
        std::cout << "✅ FFT обработка через cl_mem завершена\n\n";
        
        std::cout << "═══ Результаты (cl_mem вариант) ═══\n";
        PrintResults(result_clmem);
        
        // ═══ 6. ВАРИАНТ B: Импорт через GPUBuffer ═══
        std::cout << "═══ Шаг 5B: Обработка через GPUBuffer wrapper ═══\n";
        auto& mem_mgr = gpu.GetMemoryManager();
        
        // Создать GPUBuffer
        auto gpu_buffer = mem_mgr.CreateBuffer<std::complex<float>>(cpu_signal.size());
        
        // Скопировать из external_buffer в gpu_buffer
        cl_command_queue queue = static_cast<cl_command_queue>(
            gpu.GetBackend().GetNativeQueue());
        cl_mem gpu_buffer_mem = static_cast<cl_mem>(gpu_buffer->GetPtr());
        
        err = clEnqueueCopyBuffer(
            queue,
            external_buffer,        // src
            gpu_buffer_mem,         // dst
            0, 0,                   // offsets
            buffer_size,            // size
            0, nullptr, nullptr);
        
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to copy buffer");
        }
        
        clFinish(queue);
        std::cout << "✅ Данные скопированы в DrvGPU буфер\n\n";
        
        auto result_gpubuffer = search_3_module->ProcessNew(gpu_buffer);
        std::cout << "✅ FFT обработка через GPUBuffer завершена\n\n";
        
        std::cout << "═══ Результаты (GPUBuffer вариант) ═══\n";
        PrintResults(result_gpubuffer);
        
        // ═══ 7. Очистка внешних ресурсов ═══
        clReleaseMemObject(external_buffer);
        clReleaseCommandQueue(external_queue);
        clReleaseContext(external_context);
        
        std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
        std::cout << "║ ТЕСТ 2 ЗАВЕРШЁН УСПЕШНО ✅                                    ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════════════╝\n\n";
        
    } catch (const std::exception& e) {
        std::cerr << "❌ ОШИБКА в тесте 2: " << e.what() << "\n";
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════

int run() {
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║          ТЕСТЫ FIND 3 MAX FFT MODULE                             ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
    
    // Запустить оба теста
    test_search_3_module_svm();
    test_search_3_module_external_opencl();
    
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║          ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ                                  ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
    
    return 0;
}
}