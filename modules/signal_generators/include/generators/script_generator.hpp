#pragma once

/**
 * @file script_generator.hpp
 * @brief ScriptGenerator - text DSL -> OpenCL kernel compiler
 *
 * Allows defining signal formulas in text format that get compiled
 * into GPU kernels at runtime.
 *
 * Format:
 * @code
 * [Params]
 * ANTENNAS = 256
 * POINTS = 10000
 *
 * [Defs]
 * // Variables that depend on antenna ID
 * var_A = 1.0 + (float)ID * 0.01
 * var_W = 0.1 + (float)ID * 0.0005
 * var_P = (float)ID * 3.14 / 180.0
 *
 * [Signal]
 * // Use 'res' for real output, 'res_re'+'res_im' for complex
 * res = var_A * sin(var_W * (float)T + var_P);
 * @endcode
 *
 * Built-in variables:
 * - ID  : antenna/beam index (0 to ANTENNAS-1)
 * - T   : sample index (0 to POINTS-1)
 * - M_PI_F : pi constant
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-13
 */

#include "interface/i_backend.hpp"
#include <CL/cl.h>
#include <string>
#include <vector>
#include <complex>
#include <cstdint>

namespace signal_gen {

/// Parsed script parameters from [Params] section
struct ScriptParams {
    uint32_t antennas = 1;     ///< Number of antennas/beams
    uint32_t points   = 1024;  ///< Number of samples per antenna
};

/// Result of parsing a script
struct ParsedScript {
    ScriptParams params;
    std::vector<std::string> defs;          ///< Lines from [Defs]
    std::vector<std::string> signal_lines;  ///< Lines from [Signal]
    bool has_res     = false;  ///< 'res' variable detected
    bool has_res_re  = false;  ///< 'res_re' variable detected
    bool has_res_im  = false;  ///< 'res_im' variable detected
};

/**
 * @class ScriptGenerator
 * @brief Compiles text DSL -> OpenCL kernel and executes on GPU
 *
 * Usage:
 * @code
 * ScriptGenerator gen(backend);
 * gen.LoadScript(R"(
 *     [Params]
 *     ANTENNAS = 8
 *     POINTS = 4096
 *     [Defs]
 *     var_W = 0.1 + (float)ID * 0.005
 *     [Signal]
 *     res = sin(var_W * (float)T);
 * )");
 *
 * cl_mem gpu_buf = gen.Generate();
 * // ... use gpu_buf ...
 * clReleaseMemObject(gpu_buf);
 * @endcode
 */
class ScriptGenerator {
public:
    explicit ScriptGenerator(drv_gpu_lib::IBackend* backend);
    ~ScriptGenerator();

    // No copy
    ScriptGenerator(const ScriptGenerator&) = delete;
    ScriptGenerator& operator=(const ScriptGenerator&) = delete;

    // Move
    ScriptGenerator(ScriptGenerator&& other) noexcept;
    ScriptGenerator& operator=(ScriptGenerator&& other) noexcept;

    /**
     * @brief Parse and compile script from string
     * @param script_text Script in [Params]/[Defs]/[Signal] format
     * @throws std::runtime_error on parse or compilation failure
     */
    void LoadScript(const std::string& script_text);

    /**
     * @brief Parse and compile script from file
     * @param file_path Path to .signal or .txt file
     * @throws std::runtime_error if file cannot be read
     */
    void LoadFile(const std::string& file_path);

    /**
     * @brief Generate signal on GPU
     * @return cl_mem buffer [antennas * points * sizeof(complex<float>)]
     * @note Caller must release via clReleaseMemObject()
     */
    cl_mem Generate();

    /**
     * @brief Generate and read back to CPU
     * @return Vector of complex samples [antennas * points]
     */
    std::vector<std::complex<float>> GenerateToCpu();

    /// Get parsed params
    uint32_t GetAntennas() const { return script_.params.antennas; }
    uint32_t GetPoints() const { return script_.params.points; }
    size_t   GetTotalSamples() const { return static_cast<size_t>(script_.params.antennas) * script_.params.points; }

    /// Get generated OpenCL kernel source (for debugging)
    const std::string& GetKernelSource() const { return kernel_source_; }

    /// Check if script is loaded and compiled
    bool IsReady() const { return program_ != nullptr; }

private:
    ParsedScript ParseScript(const std::string& text);
    std::string  GenerateKernelSource(const ParsedScript& script);
    std::string  PrepareExpression(const std::string& line);
    void CompileKernel(const std::string& source);
    void ReleaseGpuResources();

    static std::string Trim(const std::string& s);
    static std::string ToUpper(const std::string& s);

    drv_gpu_lib::IBackend* backend_ = nullptr;
    cl_context context_       = nullptr;
    cl_command_queue queue_    = nullptr;
    cl_device_id device_      = nullptr;
    cl_program program_       = nullptr;

    ParsedScript script_;
    std::string  kernel_source_;
};

} // namespace signal_gen
