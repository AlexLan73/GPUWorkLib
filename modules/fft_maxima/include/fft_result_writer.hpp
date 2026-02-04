#pragma once
// ════════════════════════════════════════════════════════════════════════════
// FFT Result Writer - вывод результатов FFT на экран и в файлы
// Отделён от основного класса расчётов для чистоты архитектуры
// ════════════════════════════════════════════════════════════════════════════

#include "interface/antenna_fft_params.h"
#include <string>
#include <vector>
#include <complex>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <ctime>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace antenna_fft {

// ════════════════════════════════════════════════════════════════════════════
// FFTResultWriter - класс для вывода результатов
// ════════════════════════════════════════════════════════════════════════════

class FFTResultWriter {
public:
    // ════════════════════════════════════════════════════════════════════════
    // Вывод результатов на экран
    // ════════════════════════════════════════════════════════════════════════

    static void PrintResults(const AntennaFFTResult& result) {
        std::cout << "\n═══════════════════════════════════════════════════════════\n";
        std::cout << "  AntennaFFTProcMax Results\n";
        std::cout << "═══════════════════════════════════════════════════════════\n";
        std::cout << "Task ID: " << result.task_id << "\n";
        std::cout << "Module: " << result.module_name << "\n";
        std::cout << "Total Beams: " << result.total_beams << "\n";
        std::cout << "nFFT: " << result.nFFT << "\n\n";

        for (size_t i = 0; i < result.results.size(); ++i) {
            const auto& beam = result.results[i];
            std::cout << "Beam " << i << ":\n";
            std::cout << "  Refined Frequency: " << std::fixed << std::setprecision(4)
                      << beam.refined_frequency << " Hz";
            if (!beam.max_values.empty()) {
                float refined_bin = static_cast<float>(beam.max_values[0].index_point) + beam.freq_offset;
                std::cout << " (bin " << refined_bin << ")";
            }
            std::cout << "\n";
            std::cout << "  Max Values Found: " << beam.max_values.size() << "\n";
            for (size_t j = 0; j < beam.max_values.size(); ++j) {
                const auto& max_val = beam.max_values[j];
                std::cout << "    [" << j << "] Index: " << max_val.index_point
                          << ", Amplitude: " << std::fixed << std::setprecision(2) << max_val.amplitude
                          << ", Phase: " << max_val.phase << " deg"
                          << ", Re: " << max_val.real
                          << ", Im: " << max_val.imag << "\n";
            }
            std::cout << "\n";
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // Форматирование статистики профилирования
    // ════════════════════════════════════════════════════════════════════════

    static std::string GetProfilingStats(const FFTProfilingResults& profiling) {
        std::ostringstream oss;
        oss << "\n═══════════════════════════════════════════════════════════\n";
        oss << "  Profiling Statistics\n";
        oss << "═══════════════════════════════════════════════════════════\n";
        oss << "Upload Time:        " << std::fixed << std::setprecision(3)
            << profiling.upload_time_ms << " ms\n";
        oss << "Pre-Callback Time:  " << profiling.pre_callback_time_ms << " ms\n";
        oss << "FFT Time:           " << profiling.fft_time_ms << " ms\n";
        oss << "Post-Callback Time: " << profiling.post_callback_time_ms << " ms\n";
        oss << "Reduction Time:     " << profiling.reduction_time_ms << " ms\n";
        oss << "Download Time:      " << profiling.download_time_ms << " ms\n";
        oss << "Total Time:         " << profiling.total_time_ms << " ms\n";
        return oss.str();
    }

    // ════════════════════════════════════════════════════════════════════════
    // Сохранение результатов в файлы (MD + JSON)
    // ════════════════════════════════════════════════════════════════════════

    static void SaveResultsToFile(
        const AntennaFFTResult& result,
        const std::string& filepath,
        const FFTProfilingResults& profiling,
        const AntennaFFTParams& params,
        cl_command_queue queue = nullptr,
        cl_mem post_callback_userdata = nullptr
    ) {
        std::string base_path = filepath;
        if (filepath.empty()) {
            base_path = "antenna_result.md";
        }
        if (base_path.find("/") != 0 && base_path.find(":\\") == std::string::npos) {
            base_path = "Reports/" + base_path;
        }

        // Базовое имя без расширения
        std::string base_no_ext = base_path;
        size_t dot_pos = base_no_ext.find_last_of('.');
        if (dot_pos != std::string::npos) {
            base_no_ext = base_no_ext.substr(0, dot_pos);
        }

        std::string md_path = base_no_ext + ".md";
        std::string json_path = base_no_ext + ".json";

        // ════════════════════════════════════════════════════════════════
        // Markdown файл
        // ════════════════════════════════════════════════════════════════
        std::ofstream md_file(md_path);
        if (!md_file.is_open()) {
            throw std::runtime_error("Failed to open file for writing: " + md_path);
        }

        // Timestamp
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::tm tm_buf;
        #ifdef _WIN32
        localtime_s(&tm_buf, &time_t);
        #else
        localtime_r(&time_t, &tm_buf);
        #endif
        char time_str[64];
        std::strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", &tm_buf);

        md_file << "# AntennaFFTProcMax Results\n\n";
        md_file << "**Generated:** " << time_str << "\n\n";
        md_file << "**Task ID:** " << result.task_id << "\n";
        md_file << "**Module:** " << result.module_name << "\n";
        md_file << "**Total Beams:** " << result.total_beams << "\n";
        md_file << "**nFFT:** " << result.nFFT << "\n\n";

        md_file << "## Profiling (GPU events)\n\n";
        md_file << "Upload Time:        " << std::fixed << std::setprecision(3) << profiling.upload_time_ms << " ms\n";
        md_file << "FFT Time:           " << std::fixed << std::setprecision(3) << profiling.fft_time_ms << " ms\n";
        md_file << "Post-Callback Time: " << std::fixed << std::setprecision(3) << profiling.post_callback_time_ms << " ms\n";
        md_file << "Reduction Time:     " << std::fixed << std::setprecision(3) << profiling.reduction_time_ms << " ms\n";
        md_file << "Total Time:         " << std::fixed << std::setprecision(3) << profiling.total_time_ms << " ms\n\n";

        md_file << "## Results by Beam\n\n";
        md_file << "| Beam | Peak | Index | Amplitude | Phase (deg) | Re | Im | Refined Freq (Hz) |\n";
        md_file << "|------|------|-------|-----------|-------------|----|----|-------------------|\n";

        for (size_t i = 0; i < result.results.size(); ++i) {
            const auto& beam_result = result.results[i];
            if (beam_result.max_values.empty()) {
                md_file << "| " << i << " | - | - | - | - | - | - | - |\n";
            } else {
                for (size_t j = 0; j < beam_result.max_values.size(); ++j) {
                    const auto& max_val = beam_result.max_values[j];
                    md_file << "| " << i << " | " << (j + 1) << " | " << max_val.index_point
                            << " | " << std::fixed << std::setprecision(2) << max_val.amplitude
                            << " | " << std::setprecision(2) << max_val.phase
                            << " | " << std::setprecision(2) << max_val.real
                            << " | " << std::setprecision(2) << max_val.imag;
                    if (j == 0) {
                        md_file << " | " << std::setprecision(4) << beam_result.refined_frequency;
                    } else {
                        md_file << " | -";
                    }
                    md_file << " |\n";
                }
            }
        }
        md_file.close();

        // ════════════════════════════════════════════════════════════════
        // JSON файл
        // ════════════════════════════════════════════════════════════════
        std::ofstream json_file(json_path);
        if (!json_file.is_open()) {
            throw std::runtime_error("Failed to open file for writing: " + json_path);
        }

        // Опционально: считать FFT комплексный вектор
        std::vector<std::complex<float>> fft_data;
        if (queue && post_callback_userdata) {
            size_t post_params_size = sizeof(cl_uint) * 4;
            size_t post_complex_size = params.beam_count * params.out_count_points_fft * sizeof(cl_float2);
            fft_data.resize(params.beam_count * params.out_count_points_fft);
            cl_int err = clEnqueueReadBuffer(
                queue,
                post_callback_userdata,
                CL_TRUE,
                post_params_size,
                post_complex_size,
                fft_data.data(),
                0, nullptr, nullptr
            );
            if (err != CL_SUCCESS) {
                fft_data.clear(); // Не удалось прочитать - пропускаем
            }
        }

        json_file << "{\n";
        json_file << "  \"task_id\": \"" << result.task_id << "\",\n";
        json_file << "  \"module_name\": \"" << result.module_name << "\",\n";
        json_file << "  \"total_beams\": " << result.total_beams << ",\n";
        json_file << "  \"nFFT\": " << result.nFFT << ",\n";
        json_file << "  \"profiling_ms\": {\n";
        json_file << "    \"upload\": " << std::fixed << std::setprecision(3) << profiling.upload_time_ms << ",\n";
        json_file << "    \"fft\": " << std::fixed << std::setprecision(3) << profiling.fft_time_ms << ",\n";
        json_file << "    \"post_callback\": " << std::fixed << std::setprecision(3) << profiling.post_callback_time_ms << ",\n";
        json_file << "    \"reduction\": " << std::fixed << std::setprecision(3) << profiling.reduction_time_ms << ",\n";
        json_file << "    \"total\": " << std::fixed << std::setprecision(3) << profiling.total_time_ms << "\n";
        json_file << "  },\n";
        json_file << "  \"results\": [\n";

        for (size_t i = 0; i < result.results.size(); ++i) {
            const auto& beam_result = result.results[i];
            json_file << "    {\n";
            json_file << "      \"beam_index\": " << i << ",\n";
            json_file << "      \"v_fft\": " << beam_result.v_fft << ",\n";
            json_file << "      \"freq_offset\": " << std::fixed << std::setprecision(6) << beam_result.freq_offset << ",\n";
            json_file << "      \"refined_frequency\": " << std::fixed << std::setprecision(4) << beam_result.refined_frequency << ",\n";
            json_file << "      \"max_values\": [\n";

            for (size_t j = 0; j < beam_result.max_values.size(); ++j) {
                const auto& max_val = beam_result.max_values[j];
                json_file << "        {\n";
                json_file << "          \"index_point\": " << max_val.index_point << ",\n";
                json_file << "          \"real\": " << std::fixed << std::setprecision(2) << max_val.real << ",\n";
                json_file << "          \"imag\": " << std::fixed << std::setprecision(2) << max_val.imag << ",\n";
                json_file << "          \"amplitude\": " << std::fixed << std::setprecision(2) << max_val.amplitude << ",\n";
                json_file << "          \"phase\": " << std::fixed << std::setprecision(2) << max_val.phase << "\n";
                json_file << "        }";
                if (j < beam_result.max_values.size() - 1) json_file << ",";
                json_file << "\n";
            }

            json_file << "      ]";

            // Опционально: FFT complex data
            if (!fft_data.empty()) {
                json_file << ",\n      \"fft_complex\": [\n";
                size_t beam_offset = i * params.out_count_points_fft;
                for (size_t k = 0; k < params.out_count_points_fft; ++k) {
                    size_t idx = beam_offset + k;
                    if (idx < fft_data.size()) {
                        json_file << "        [" << std::fixed << std::setprecision(6)
                                  << fft_data[idx].real() << ", " << fft_data[idx].imag() << "]";
                    } else {
                        json_file << "        [0.0, 0.0]";
                    }
                    if (k + 1 < params.out_count_points_fft) json_file << ",";
                    json_file << "\n";
                }
                json_file << "      ]\n";
            } else {
                json_file << "\n";
            }

            json_file << "    }";
            if (i < result.results.size() - 1) json_file << ",";
            json_file << "\n";
        }

        json_file << "  ]\n";
        json_file << "}\n";
        json_file.close();
    }

    // ════════════════════════════════════════════════════════════════════════
    // Вывод профилирования на экран
    // ════════════════════════════════════════════════════════════════════════

    static void PrintProfiling(const FFTProfilingResults& profiling) {
        std::cout << GetProfilingStats(profiling);
    }
};

} // namespace antenna_fft
