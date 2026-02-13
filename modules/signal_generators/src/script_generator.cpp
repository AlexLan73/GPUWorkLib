/**
 * @file script_generator.cpp
 * @brief ScriptGenerator - text DSL -> OpenCL kernel compiler
 *
 * Parses [Params]/[Defs]/[Signal] format, generates OpenCL kernel
 * source, compiles and executes on GPU.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-13
 */

#include "generators/script_generator.hpp"
#include <stdexcept>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cctype>

namespace signal_gen {

// ============================================================================
// Constructor / Destructor
// ============================================================================

ScriptGenerator::ScriptGenerator(drv_gpu_lib::IBackend* backend)
    : backend_(backend)
{
    if (!backend_ || !backend_->IsInitialized()) {
        throw std::runtime_error("ScriptGenerator: backend is null or not initialized");
    }

    context_ = static_cast<cl_context>(backend_->GetNativeContext());
    queue_   = static_cast<cl_command_queue>(backend_->GetNativeQueue());
    device_  = static_cast<cl_device_id>(backend_->GetNativeDevice());
}

ScriptGenerator::~ScriptGenerator() {
    ReleaseGpuResources();
}

ScriptGenerator::ScriptGenerator(ScriptGenerator&& other) noexcept
    : backend_(other.backend_)
    , context_(other.context_)
    , queue_(other.queue_)
    , device_(other.device_)
    , program_(other.program_)
    , script_(std::move(other.script_))
    , kernel_source_(std::move(other.kernel_source_))
{
    other.program_ = nullptr;
}

ScriptGenerator& ScriptGenerator::operator=(ScriptGenerator&& other) noexcept {
    if (this != &other) {
        ReleaseGpuResources();
        backend_ = other.backend_;
        context_ = other.context_;
        queue_ = other.queue_;
        device_ = other.device_;
        program_ = other.program_;
        script_ = std::move(other.script_);
        kernel_source_ = std::move(other.kernel_source_);
        other.program_ = nullptr;
    }
    return *this;
}

// ============================================================================
// Public API
// ============================================================================

void ScriptGenerator::LoadScript(const std::string& script_text) {
    ReleaseGpuResources();

    script_ = ParseScript(script_text);

    if (script_.signal_lines.empty()) {
        throw std::runtime_error("ScriptGenerator: [Signal] section is empty or missing");
    }

    kernel_source_ = GenerateKernelSource(script_);
    CompileKernel(kernel_source_);
}

void ScriptGenerator::LoadFile(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("ScriptGenerator: cannot open file '" + file_path + "'");
    }

    std::ostringstream ss;
    ss << file.rdbuf();
    LoadScript(ss.str());
}

cl_mem ScriptGenerator::Generate() {
    if (!program_) {
        throw std::runtime_error("ScriptGenerator: no script loaded (call LoadScript first)");
    }

    uint32_t antennas = script_.params.antennas;
    uint32_t points   = script_.params.points;
    size_t total = static_cast<size_t>(antennas) * points;
    size_t buffer_size = total * sizeof(std::complex<float>);

    cl_int err;
    cl_mem output = clCreateBuffer(context_, CL_MEM_READ_WRITE, buffer_size, nullptr, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("ScriptGenerator::Generate: clCreateBuffer failed: " + std::to_string(err));
    }

    cl_kernel kernel = clCreateKernel(program_, "script_signal", &err);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(output);
        throw std::runtime_error("ScriptGenerator::Generate: clCreateKernel failed: " + std::to_string(err));
    }

    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 1, sizeof(uint32_t), &antennas);
    err |= clSetKernelArg(kernel, 2, sizeof(uint32_t), &points);

    if (err != CL_SUCCESS) {
        clReleaseKernel(kernel);
        clReleaseMemObject(output);
        throw std::runtime_error("ScriptGenerator::Generate: clSetKernelArg failed");
    }

    size_t local_size = 256;
    size_t global_size = ((total + local_size - 1) / local_size) * local_size;

    err = clEnqueueNDRangeKernel(queue_, kernel, 1, nullptr,
                                  &global_size, &local_size,
                                  0, nullptr, nullptr);
    clReleaseKernel(kernel);

    if (err != CL_SUCCESS) {
        clReleaseMemObject(output);
        throw std::runtime_error("ScriptGenerator::Generate: clEnqueueNDRangeKernel failed: " + std::to_string(err));
    }

    clFinish(queue_);
    return output;
}

std::vector<std::complex<float>> ScriptGenerator::GenerateToCpu() {
    cl_mem gpu_buf = Generate();

    size_t total = GetTotalSamples();
    std::vector<std::complex<float>> data(total);

    cl_int err = clEnqueueReadBuffer(queue_, gpu_buf, CL_TRUE, 0,
                                      total * sizeof(std::complex<float>),
                                      data.data(), 0, nullptr, nullptr);
    clReleaseMemObject(gpu_buf);

    if (err != CL_SUCCESS) {
        throw std::runtime_error("ScriptGenerator::GenerateToCpu: clEnqueueReadBuffer failed: " + std::to_string(err));
    }

    return data;
}

// ============================================================================
// Parser
// ============================================================================

ParsedScript ScriptGenerator::ParseScript(const std::string& text) {
    ParsedScript result;
    enum class Section { NONE, PARAMS, DEFS, SIGNAL };
    Section section = Section::NONE;

    std::istringstream iss(text);
    std::string line;

    while (std::getline(iss, line)) {
        std::string trimmed = Trim(line);

        // Skip empty lines and comments
        if (trimmed.empty())
            continue;
        if (trimmed.size() >= 2 && trimmed[0] == '/' && trimmed[1] == '/')
            continue;
        if (trimmed[0] == '#')
            continue;

        // Section headers (case-insensitive)
        std::string upper = ToUpper(trimmed);
        if (upper == "[PARAMS]")  { section = Section::PARAMS; continue; }
        if (upper == "[DEFS]")    { section = Section::DEFS;   continue; }
        if (upper == "[SIGNAL]")  { section = Section::SIGNAL; continue; }

        switch (section) {
            case Section::PARAMS: {
                auto eq = trimmed.find('=');
                if (eq != std::string::npos) {
                    std::string key = ToUpper(Trim(trimmed.substr(0, eq)));
                    std::string val = Trim(trimmed.substr(eq + 1));

                    if (key == "ANTENNAS" || key == "ANTENNA_COUNT" || key == "BEAMS")
                        result.params.antennas = static_cast<uint32_t>(std::stoul(val));
                    else if (key == "POINTS" || key == "LENGTH" || key == "SAMPLES")
                        result.params.points = static_cast<uint32_t>(std::stoul(val));
                }
                break;
            }
            case Section::DEFS: {
                result.defs.push_back(trimmed);
                break;
            }
            case Section::SIGNAL: {
                result.signal_lines.push_back(trimmed);

                // Detect output variable names
                std::string check = trimmed;
                // Remove type prefix if present for detection
                if (check.substr(0, 6) == "float ") check = check.substr(6);
                check = Trim(check);

                if (check.substr(0, 7) == "res_re " || check.substr(0, 7) == "res_re=")
                    result.has_res_re = true;
                else if (check.substr(0, 7) == "res_im " || check.substr(0, 7) == "res_im=")
                    result.has_res_im = true;
                else if (check.substr(0, 4) == "res " || check.substr(0, 4) == "res=")
                    result.has_res = true;
                break;
            }
            default:
                break;
        }
    }

    return result;
}

// ============================================================================
// Kernel Source Generator
// ============================================================================

std::string ScriptGenerator::GenerateKernelSource(const ParsedScript& script) {
    std::ostringstream k;

    k << "// Auto-generated by ScriptGenerator\n";
    k << "__kernel void script_signal(\n";
    k << "    __global float2* output,\n";
    k << "    const uint antennas,\n";
    k << "    const uint points)\n";
    k << "{\n";
    k << "    const uint gid = get_global_id(0);\n";
    k << "    const uint ID = gid / points;\n";
    k << "    const uint T  = gid % points;\n";
    k << "    if (ID >= antennas) return;\n";
    k << "\n";

    // [Defs] section
    if (!script.defs.empty()) {
        k << "    // --- [Defs] ---\n";
        for (const auto& def_line : script.defs) {
            k << "    " << PrepareExpression(def_line) << "\n";
        }
        k << "\n";
    }

    // [Signal] section
    k << "    // --- [Signal] ---\n";
    for (const auto& sig_line : script.signal_lines) {
        k << "    " << PrepareExpression(sig_line) << "\n";
    }
    k << "\n";

    // Output mapping
    if (script.has_res_re && script.has_res_im) {
        k << "    output[gid] = (float2)(res_re, res_im);\n";
    } else if (script.has_res) {
        k << "    output[gid] = (float2)(res, 0.0f);\n";
    } else if (script.has_res_re) {
        k << "    output[gid] = (float2)(res_re, 0.0f);\n";
    } else {
        // Fallback: try 'res' anyway, OpenCL compiler will catch errors
        k << "    output[gid] = (float2)(res, 0.0f);\n";
    }

    k << "}\n";

    return k.str();
}

std::string ScriptGenerator::PrepareExpression(const std::string& line) {
    std::string expr = Trim(line);

    // Remove inline comment (but be careful with // inside strings)
    auto comment_pos = expr.find("//");
    if (comment_pos != std::string::npos) {
        expr = Trim(expr.substr(0, comment_pos));
    }

    if (expr.empty()) return "";

    // Check if this is a variable assignment (has '=' but not '==' / '!=' / '<=' / '>=')
    bool is_assignment = false;
    size_t eq_pos = std::string::npos;

    for (size_t i = 0; i < expr.size(); ++i) {
        if (expr[i] == '=') {
            // Check it's not ==, !=, <=, >=
            bool is_compare = false;
            if (i + 1 < expr.size() && expr[i + 1] == '=') is_compare = true;  // ==
            if (i > 0 && (expr[i - 1] == '!' || expr[i - 1] == '<' || expr[i - 1] == '>'))
                is_compare = true;  // !=, <=, >=

            if (!is_compare) {
                eq_pos = i;
                is_assignment = true;
                break;
            }
        }
    }

    // If it's an assignment, check if it already has a type prefix
    if (is_assignment && eq_pos != std::string::npos) {
        std::string lhs = Trim(expr.substr(0, eq_pos));

        // Check if LHS already starts with a type keyword
        bool has_type = false;
        const char* types[] = { "float", "int", "uint", "double", "float2", "float4" };
        for (auto t : types) {
            size_t tlen = strlen(t);
            if (lhs.size() > tlen && lhs.substr(0, tlen) == t &&
                (lhs[tlen] == ' ' || lhs[tlen] == '\t')) {
                has_type = true;
                break;
            }
        }

        if (!has_type) {
            expr = "float " + expr;
        }
    }

    // Add semicolon if missing
    if (!expr.empty() && expr.back() != ';') {
        expr += ";";
    }

    return expr;
}

// ============================================================================
// Kernel Compilation
// ============================================================================

void ScriptGenerator::CompileKernel(const std::string& source) {
    cl_int err;
    const char* src = source.c_str();
    size_t src_len = source.size();

    program_ = clCreateProgramWithSource(context_, 1, &src, &src_len, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("ScriptGenerator: clCreateProgramWithSource failed: " + std::to_string(err));
    }

    // Build with single-precision constants and fast math
    err = clBuildProgram(program_, 1, &device_,
                         "-cl-fast-relaxed-math -cl-single-precision-constant",
                         nullptr, nullptr);

    if (err != CL_SUCCESS) {
        // Get build log for diagnostics
        size_t log_size = 0;
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size + 1, '\0');
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);

        clReleaseProgram(program_);
        program_ = nullptr;

        std::string error_msg = "ScriptGenerator: kernel compilation failed!\n\n";
        error_msg += "=== Generated kernel ===\n" + source + "\n";
        error_msg += "=== Build log ===\n" + std::string(log.data()) + "\n";

        throw std::runtime_error(error_msg);
    }
}

void ScriptGenerator::ReleaseGpuResources() {
    if (program_) { clReleaseProgram(program_); program_ = nullptr; }
}

// ============================================================================
// Utilities
// ============================================================================

std::string ScriptGenerator::Trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    size_t end = s.find_last_not_of(" \t\r\n");
    return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}

std::string ScriptGenerator::ToUpper(const std::string& s) {
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return result;
}

} // namespace signal_gen
