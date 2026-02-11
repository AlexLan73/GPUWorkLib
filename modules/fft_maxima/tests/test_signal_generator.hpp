#pragma once

/**
 * @file test_signal_generator.hpp
 * @brief Генератор тестовых сигналов на GPU (OpenCL)
 *
 * ВРЕМЕННО в tests/! Позже переедет в библиотеку синтезаторов сигналов.
 *
 * Генерирует синусоиды с заданными частотами прямо на GPU,
 * без передачи данных через CPU. Используется для тестирования
 * ProcessFromGPU() — обработки данных уже находящихся на GPU.
 *
 * Использование:
 * @code
 * // Контекст уже создан (ExternalOpenCLContext или DrvGPU)
 * TestSignalGenerator generator(ctx, queue, device);
 *
 * // Генерируем 256 антенн × 1.3M точек на GPU
 * cl_mem gpu_data = generator.GenerateSinusoids(256, 1300000, 1000.0f, 2.5f, 0.25f);
 *
 * // Используем gpu_data в SpectrumMaximaFinder::Process(InputData<cl_mem>)
 * // ...
 *
 * // Освобождаем буфер когда не нужен
 * generator.ReleaseBuffer(gpu_data);
 * @endcode
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-11
 */

#include <CL/cl.h>
#include <string>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <cstring>
#include <vector>
#include <complex>

namespace test_signal_generator {

// ════════════════════════════════════════════════════════════════════════════
// OpenCL Kernel Source — генерация синусоид
// ════════════════════════════════════════════════════════════════════════════

static const char* SINUSOID_KERNEL_SOURCE = R"CL(
/**
 * @brief Генерация комплексных синусоид на GPU
 *
 * Каждый work item генерирует одну точку (complex<float>).
 * Global work size = antenna_count × n_point
 *
 * @param output        Выходной буфер [antenna_count × n_point] float2
 * @param antenna_count Количество антенн
 * @param n_point       Точек на антенну
 * @param sample_rate   Частота дискретизации (Гц)
 * @param base_freq     Базовая частота для антенны 0 (Гц)
 * @param freq_step     Шаг частоты между антеннами (Гц)
 */
__kernel void generate_sinusoid(
    __global float2* output,
    const uint antenna_count,
    const uint n_point,
    const float sample_rate,
    const float base_freq,
    const float freq_step)
{
    const size_t gid = get_global_id(0);
    const size_t antenna_id = gid / n_point;
    const size_t sample_id = gid % n_point;

    // Проверка границ
    if (antenna_id >= antenna_count) return;

    // Частота для этой антенны: base_freq + antenna_id × freq_step
    const float freq = base_freq + (float)antenna_id * freq_step;

    // Время: sample_id / sample_rate
    const float t = (float)sample_id / sample_rate;

    // Фаза: 2π × freq × t
    const float phase = 2.0f * M_PI_F * freq * t;

    // Синусоида: e^(i×phase) = cos(phase) + i×sin(phase)
    output[gid] = (float2)(cos(phase), sin(phase));
}
)CL";

// ════════════════════════════════════════════════════════════════════════════
// Класс TestSignalGenerator
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class TestSignalGenerator
 * @brief Генератор тестовых синусоид на GPU
 *
 * НЕ владеет контекстом/очередью — использует внешние ресурсы.
 * Компилирует kernel один раз, переиспользует для генерации.
 */
class TestSignalGenerator {
public:
    /**
     * @brief Конструктор
     * @param ctx OpenCL контекст (НЕ владеет!)
     * @param queue Command queue (НЕ владеет!)
     * @param device Device ID (НЕ владеет!)
     */
    TestSignalGenerator(cl_context ctx, cl_command_queue queue, cl_device_id device)
        : ctx_(ctx), queue_(queue), device_(device) {

        if (!ctx_ || !queue_ || !device_) {
            throw std::invalid_argument("TestSignalGenerator: invalid OpenCL resources");
        }

        // Компилируем kernel при создании
        CompileKernel();

        std::cout << "[TestSignalGenerator] Initialized, kernel compiled\n";
    }

    /**
     * @brief Деструктор — освобождает только kernel/program (НЕ контекст!)
     */
    ~TestSignalGenerator() {
        if (kernel_) {
            clReleaseKernel(kernel_);
            kernel_ = nullptr;
        }
        if (program_) {
            clReleaseProgram(program_);
            program_ = nullptr;
        }
        std::cout << "[TestSignalGenerator] Destroyed\n";
    }

    // Запрет копирования
    TestSignalGenerator(const TestSignalGenerator&) = delete;
    TestSignalGenerator& operator=(const TestSignalGenerator&) = delete;

    /**
     * @brief Генерация синусоид на GPU
     *
     * @param antenna_count Количество антенн (лучей)
     * @param n_point       Точек на антенну
     * @param sample_rate   Частота дискретизации (Гц)
     * @param base_freq     Базовая частота для антенны 0 (Гц)
     * @param freq_step     Шаг частоты между антеннами (Гц)
     * @return cl_mem буфер с данными [antenna_count × n_point × complex<float>]
     *
     * ВАЖНО: Вызывающий код должен освободить буфер через ReleaseBuffer()!
     */
    cl_mem GenerateSinusoids(
        size_t antenna_count,
        size_t n_point,
        float sample_rate,
        float base_freq,
        float freq_step = 0.25f)
    {
        // Размер буфера
        size_t total_points = antenna_count * n_point;
        size_t buffer_size = total_points * sizeof(std::complex<float>);  // float2 = 8 bytes

        std::cout << "[TestSignalGenerator] Generating " << antenna_count
                  << " × " << n_point << " = " << total_points << " points ("
                  << (buffer_size / 1024.0 / 1024.0) << " MB)\n";
        std::cout << "  Frequencies: " << base_freq << " Hz to "
                  << (base_freq + (antenna_count - 1) * freq_step) << " Hz"
                  << " (step " << freq_step << " Hz)\n";

        // Создаём выходной буфер
        cl_int err;
        cl_mem output_buffer = clCreateBuffer(ctx_, CL_MEM_READ_WRITE,
                                               buffer_size, nullptr, &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("GenerateSinusoids: clCreateBuffer failed: " + std::to_string(err));
        }

        // Устанавливаем аргументы kernel
        uint32_t ant_count_u = static_cast<uint32_t>(antenna_count);
        uint32_t n_point_u = static_cast<uint32_t>(n_point);

        err  = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &output_buffer);
        err |= clSetKernelArg(kernel_, 1, sizeof(uint32_t), &ant_count_u);
        err |= clSetKernelArg(kernel_, 2, sizeof(uint32_t), &n_point_u);
        err |= clSetKernelArg(kernel_, 3, sizeof(float), &sample_rate);
        err |= clSetKernelArg(kernel_, 4, sizeof(float), &base_freq);
        err |= clSetKernelArg(kernel_, 5, sizeof(float), &freq_step);

        if (err != CL_SUCCESS) {
            clReleaseMemObject(output_buffer);
            throw std::runtime_error("GenerateSinusoids: clSetKernelArg failed: " + std::to_string(err));
        }

        // Запускаем kernel
        size_t global_work_size = total_points;
        size_t local_work_size = 256;  // Стандартный размер work-group

        // Выравниваем global size до кратного local size
        if (global_work_size % local_work_size != 0) {
            global_work_size = ((global_work_size / local_work_size) + 1) * local_work_size;
        }

        cl_event kernel_event = nullptr;
        err = clEnqueueNDRangeKernel(queue_, kernel_, 1, nullptr,
                                      &global_work_size, &local_work_size,
                                      0, nullptr, &kernel_event);
        if (err != CL_SUCCESS) {
            clReleaseMemObject(output_buffer);
            throw std::runtime_error("GenerateSinusoids: clEnqueueNDRangeKernel failed: " + std::to_string(err));
        }

        // Ждём завершения и получаем время выполнения
        clWaitForEvents(1, &kernel_event);

        cl_ulong start_time, end_time;
        clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START,
                                sizeof(cl_ulong), &start_time, nullptr);
        clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END,
                                sizeof(cl_ulong), &end_time, nullptr);

        double time_ms = (end_time - start_time) / 1e6;
        std::cout << "  Generation time: " << time_ms << " ms\n";

        clReleaseEvent(kernel_event);

        return output_buffer;
    }

    /**
     * @brief Освободить буфер, созданный GenerateSinusoids()
     * @param buffer Буфер для освобождения
     */
    void ReleaseBuffer(cl_mem buffer) {
        if (buffer) {
            clReleaseMemObject(buffer);
        }
    }

    /**
     * @brief Валидация данных — читаем первые N точек и сравниваем с CPU
     *
     * @param buffer      GPU буфер с данными
     * @param antenna_count Количество антенн
     * @param n_point     Точек на антенну
     * @param sample_rate Частота дискретизации
     * @param base_freq   Базовая частота
     * @param freq_step   Шаг частоты
     * @param check_points Сколько точек проверить (по умолчанию 10)
     * @return true если данные совпадают с CPU reference
     */
    bool ValidateBuffer(
        cl_mem buffer,
        size_t antenna_count,
        size_t n_point,
        float sample_rate,
        float base_freq,
        float freq_step,
        size_t check_points = 10)
    {
        // Читаем первые check_points точек для каждой антенны (или меньше)
        size_t points_to_check = std::min(check_points, n_point);
        size_t antennas_to_check = std::min(antenna_count, size_t(5));

        std::vector<std::complex<float>> gpu_data(antennas_to_check * points_to_check);

        bool all_ok = true;
        const float tolerance = 1e-5f;

        std::cout << "[TestSignalGenerator] Validating first " << points_to_check
                  << " points of " << antennas_to_check << " antennas...\n";

        for (size_t ant = 0; ant < antennas_to_check; ++ant) {
            // Читаем points_to_check точек для антенны ant
            size_t offset = ant * n_point * sizeof(std::complex<float>);
            size_t size = points_to_check * sizeof(std::complex<float>);

            cl_int err = clEnqueueReadBuffer(queue_, buffer, CL_TRUE,
                                              offset, size,
                                              gpu_data.data() + ant * points_to_check,
                                              0, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                std::cerr << "  Validation read failed for antenna " << ant << "\n";
                return false;
            }

            // Сравниваем с CPU reference
            float freq = base_freq + ant * freq_step;
            bool antenna_ok = true;

            for (size_t i = 0; i < points_to_check && antenna_ok; ++i) {
                float t = static_cast<float>(i) / sample_rate;
                float phase = 2.0f * M_PI * freq * t;
                std::complex<float> expected(std::cos(phase), std::sin(phase));
                std::complex<float> actual = gpu_data[ant * points_to_check + i];

                float diff_re = std::abs(actual.real() - expected.real());
                float diff_im = std::abs(actual.imag() - expected.imag());

                if (diff_re > tolerance || diff_im > tolerance) {
                    std::cerr << "  Antenna " << ant << " point " << i
                              << ": expected (" << expected.real() << ", " << expected.imag()
                              << "), got (" << actual.real() << ", " << actual.imag() << ")\n";
                    antenna_ok = false;
                    all_ok = false;
                }
            }

            if (antenna_ok) {
                std::cout << "  Antenna " << ant << " (freq=" << freq << " Hz): OK\n";
            }
        }

        return all_ok;
    }

private:
    /**
     * @brief Компиляция OpenCL kernel
     */
    void CompileKernel() {
        cl_int err;

        // Создаём program
        const char* source = SINUSOID_KERNEL_SOURCE;
        size_t source_len = strlen(source);

        program_ = clCreateProgramWithSource(ctx_, 1, &source, &source_len, &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("CompileKernel: clCreateProgramWithSource failed: " + std::to_string(err));
        }

        // Компилируем
        err = clBuildProgram(program_, 1, &device_, "-cl-std=CL1.2", nullptr, nullptr);
        if (err != CL_SUCCESS) {
            // Получаем лог ошибок
            size_t log_size;
            clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);

            std::string build_log(log_size, '\0');
            clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, log_size, &build_log[0], nullptr);

            clReleaseProgram(program_);
            program_ = nullptr;

            throw std::runtime_error("CompileKernel: clBuildProgram failed:\n" + build_log);
        }

        // Создаём kernel
        kernel_ = clCreateKernel(program_, "generate_sinusoid", &err);
        if (err != CL_SUCCESS) {
            clReleaseProgram(program_);
            program_ = nullptr;
            throw std::runtime_error("CompileKernel: clCreateKernel failed: " + std::to_string(err));
        }
    }

    cl_context ctx_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_program program_ = nullptr;
    cl_kernel kernel_ = nullptr;
};

} // namespace test_signal_generator
