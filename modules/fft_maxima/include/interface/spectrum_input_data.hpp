#pragma once

/**
 * @file spectrum_input_data.hpp
 * @brief Новый универсальный API для SpectrumMaximaFinder
 *
 * Часть API Refactoring (MemoryBank/specs/api_refactoring.md)
 *
 * Поддерживаемые типы данных (T):
 * - std::vector<std::complex<float>> — данные на CPU
 * - cl_mem — OpenCL буфер (от заказчика)
 * - void* — SVM pointer
 * - GPUBuffer<std::complex<float>>* — наша RAII обёртка
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-11
 */

#include <complex>
#include <vector>
#include <cstdint>

namespace antenna_fft {

// ═══════════════════════════════════════════════════════════════════════════
// InputData<T> — универсальная структура входных данных
// ═══════════════════════════════════════════════════════════════════════════

/**
 * @struct InputData
 * @brief Шаблонная структура для входных данных любого типа
 *
 * @tparam T Тип данных:
 *   - std::vector<std::complex<float>> — CPU вектор
 *   - cl_mem — OpenCL буфер
 *   - void* — SVM указатель
 *
 * Пример использования:
 * @code
 * // CPU данные
 * InputData<std::vector<std::complex<float>>> cpu_input{
 *     .antenna_count = 256,
 *     .n_point = 1300000,
 *     .data = my_vector
 * };
 *
 * // GPU данные (cl_mem)
 * InputData<cl_mem> gpu_input{
 *     .antenna_count = 256,
 *     .n_point = 1300000,
 *     .data = my_cl_mem_buffer
 * };
 * @endcode
 */
template<typename T>
struct InputData {
    uint32_t antenna_count = 0;     ///< Количество антенн (лучей)
    uint32_t n_point = 0;           ///< Точек на антенну (комплексных float)
    T data{};                       ///< Данные в любом формате
    size_t gpu_memory_bytes = 0;    ///< Реальный размер буфера на GPU (для cl_mem)
                                    ///< Может быть больше SizeBytes() из-за выравнивания/padding
                                    ///< 0 = использовать SizeBytes() как fallback

    /// Общее количество точек
    size_t TotalPoints() const { return static_cast<size_t>(antenna_count) * n_point; }

    /// Расчётный размер данных в байтах (для complex<float> = 8 bytes per point)
    /// @note Реальный размер буфера может быть больше — см. gpu_memory_bytes
    size_t SizeBytes() const { return TotalPoints() * sizeof(std::complex<float>); }

    /// Реальный размер занятой GPU памяти
    /// @return gpu_memory_bytes если указан, иначе SizeBytes()
    size_t ActualGpuMemory() const {
        return (gpu_memory_bytes > 0) ? gpu_memory_bytes : SizeBytes();
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// ProcessingParams — параметры обработки
// ═══════════════════════════════════════════════════════════════════════════

/**
 * @struct ProcessingParams
 * @brief Параметры обработки FFT
 *
 * Пример:
 * @code
 * ProcessingParams params{
 *     .repeat_count = 2,
 *     .sample_rate = 1000.0f,
 *     .memory_limit = 0.80f
 * };
 * @endcode
 */
struct ProcessingParams {
    uint32_t repeat_count = 2;      ///< Множитель FFT: nFFT = NextPowerOf2(n_point) × repeat_count
    float sample_rate = 1000.0f;    ///< Частота дискретизации (Гц)
    uint32_t search_range = 0;      ///< Диапазон поиска максимума (0 = авто = nFFT/4)
    float memory_limit = 0.80f;     ///< Доля свободной памяти GPU для batch (0.0-1.0)
};

// ═══════════════════════════════════════════════════════════════════════════
// BackendType — тип GPU backend'а
// ═══════════════════════════════════════════════════════════════════════════

/**
 * @enum DriverType
 * @brief Тип драйвера/backend'а для обработки
 *
 * Используется для выбора реализации (OpenCL vs ROCm/HIP)
 */
enum class DriverType {
    AUTO,       ///< Автоматический выбор (по умолчанию)
    OPENCL,     ///< OpenCL (clFFT)
    ROCM        ///< ROCm/HIP (hipFFT)
};

// ═══════════════════════════════════════════════════════════════════════════
// Вспомогательные type traits
// ═══════════════════════════════════════════════════════════════════════════

/// Проверка: это CPU вектор?
template<typename T>
struct is_cpu_vector : std::false_type {};

template<>
struct is_cpu_vector<std::vector<std::complex<float>>> : std::true_type {};

template<typename T>
inline constexpr bool is_cpu_vector_v = is_cpu_vector<T>::value;

/// Проверка: это SVM указатель?
template<typename T>
struct is_svm_pointer : std::false_type {};

template<>
struct is_svm_pointer<void*> : std::true_type {};

template<typename T>
inline constexpr bool is_svm_pointer_v = is_svm_pointer<T>::value;

// Примечание: cl_mem проверяется через std::is_pointer или отдельный trait
// в реализации Process(), т.к. cl_mem = void* в OpenCL

} // namespace antenna_fft
