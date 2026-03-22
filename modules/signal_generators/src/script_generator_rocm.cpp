/**
 * @file script_generator_rocm.cpp
 * @brief ScriptGeneratorROCm — DSL → HIP kernel (hiprtc)
 *
 * Port of script_generator.cpp: OpenCL → hiprtc.
 * DSL parser is identical (CPU-only code), only kernel format and
 * compilation/execution path changed.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-22
 */

#if ENABLE_ROCM

#include "generators/script_generator_rocm.hpp"
#include "generators/script_generator.hpp"  // ParsedScript, ScriptParams (types only)

#include <hip/hiprtc.h>
#include <stdexcept>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <cstring>

namespace signal_gen {

// ============================================================================
// Constructor / Destructor / Move
// ============================================================================

ScriptGeneratorROCm::ScriptGeneratorROCm(drv_gpu_lib::IBackend* backend)
    : backend_(backend) {
  if (!backend_ || !backend_->IsInitialized())
    throw std::runtime_error("ScriptGeneratorROCm: backend is null or not initialized");
  stream_ = static_cast<hipStream_t>(backend_->GetNativeQueue());
}

ScriptGeneratorROCm::~ScriptGeneratorROCm() {
  ReleaseGpuResources();
}

ScriptGeneratorROCm::ScriptGeneratorROCm(ScriptGeneratorROCm&& other) noexcept
    : backend_(other.backend_), stream_(other.stream_)
    , module_(other.module_), kernel_fn_(other.kernel_fn_)
    , antennas_(other.antennas_), points_(other.points_)
    , kernel_source_(std::move(other.kernel_source_)) {
  other.module_ = nullptr; other.kernel_fn_ = nullptr;
}

ScriptGeneratorROCm& ScriptGeneratorROCm::operator=(ScriptGeneratorROCm&& other) noexcept {
  if (this != &other) {
    ReleaseGpuResources();
    backend_ = other.backend_; stream_ = other.stream_;
    module_ = other.module_; kernel_fn_ = other.kernel_fn_;
    antennas_ = other.antennas_; points_ = other.points_;
    kernel_source_ = std::move(other.kernel_source_);
    other.module_ = nullptr; other.kernel_fn_ = nullptr;
  }
  return *this;
}

uint32_t ScriptGeneratorROCm::GetAntennas() const { return antennas_; }
uint32_t ScriptGeneratorROCm::GetPoints() const { return points_; }
size_t ScriptGeneratorROCm::GetTotalSamples() const {
  return static_cast<size_t>(antennas_) * points_;
}

// ============================================================================
// Load / Compile
// ============================================================================

void ScriptGeneratorROCm::LoadScript(const std::string& script_text) {
  ReleaseGpuResources();
  auto script = ParseScript(script_text);
  antennas_ = script.params.antennas;
  points_ = script.params.points;
  kernel_source_ = GenerateHIPKernelSource(script);
  CompileKernel(kernel_source_);
}

void ScriptGeneratorROCm::LoadFile(const std::string& file_path) {
  std::ifstream file(file_path);
  if (!file.is_open())
    throw std::runtime_error("ScriptGeneratorROCm: cannot open '" + file_path + "'");
  std::ostringstream ss;
  ss << file.rdbuf();
  LoadScript(ss.str());
}

// ============================================================================
// Generate
// ============================================================================

void* ScriptGeneratorROCm::Generate() {
  if (!module_)
    throw std::runtime_error("ScriptGeneratorROCm: no script loaded");

  size_t total = GetTotalSamples();
  size_t buffer_size = total * sizeof(std::complex<float>);

  void* output = nullptr;
  hipError_t err = hipMalloc(&output, buffer_size);
  if (err != hipSuccess)
    throw std::runtime_error("ScriptGeneratorROCm: hipMalloc failed");

  unsigned int antennas = antennas_;
  unsigned int points = points_;
  void* args[] = { &output, &antennas, &points };

  unsigned int grid = static_cast<unsigned int>((total + 255) / 256);
  err = hipModuleLaunchKernel(kernel_fn_,
      grid, 1, 1, 256, 1, 1,
      0, stream_, args, nullptr);
  if (err != hipSuccess) {
    hipFree(output);
    throw std::runtime_error("ScriptGeneratorROCm: kernel launch failed");
  }
  hipStreamSynchronize(stream_);
  return output;
}

std::vector<std::complex<float>> ScriptGeneratorROCm::GenerateToCpu() {
  void* gpu_buf = Generate();
  size_t total = GetTotalSamples();
  std::vector<std::complex<float>> data(total);
  hipMemcpyDtoH(data.data(), gpu_buf, total * sizeof(std::complex<float>));
  hipFree(gpu_buf);
  return data;
}

// ============================================================================
// Parser (identical to OpenCL version — CPU-only logic)
// ============================================================================

ParsedScript ScriptGeneratorROCm::ParseScript(const std::string& text) {
  ParsedScript result;
  enum class Section { NONE, PARAMS, DEFS, SIGNAL };
  Section section = Section::NONE;

  std::istringstream iss(text);
  std::string line;

  while (std::getline(iss, line)) {
    std::string trimmed = Trim(line);
    if (trimmed.empty()) continue;
    if (trimmed.size() >= 2 && trimmed[0] == '/' && trimmed[1] == '/') continue;
    if (trimmed[0] == '#') continue;

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
      case Section::DEFS:
        result.defs.push_back(trimmed);
        break;
      case Section::SIGNAL: {
        result.signal_lines.push_back(trimmed);
        std::string check = trimmed;
        if (check.substr(0, 6) == "float ") check = check.substr(6);
        check = Trim(check);
        if (check.substr(0, 7) == "res_re " || check.substr(0, 7) == "res_re=") result.has_res_re = true;
        else if (check.substr(0, 7) == "res_im " || check.substr(0, 7) == "res_im=") result.has_res_im = true;
        else if (check.substr(0, 4) == "res " || check.substr(0, 4) == "res=") result.has_res = true;
        break;
      }
      default: break;
    }
  }
  return result;
}

// ============================================================================
// HIP Kernel Source Generation (differs from OpenCL: extern "C" __global__)
// ============================================================================

std::string ScriptGeneratorROCm::GenerateHIPKernelSource(const ParsedScript& script) {
  std::ostringstream k;
  k << "// Auto-generated by ScriptGeneratorROCm\n";
  k << "struct float2_t { float x; float y; };\n\n";
  k << "#ifndef M_PI_F\n#define M_PI_F 3.14159265358979323846f\n#endif\n\n";
  k << "extern \"C\" __global__ void script_signal(\n";
  k << "    float2_t* output,\n";
  k << "    const unsigned int antennas,\n";
  k << "    const unsigned int points)\n";
  k << "{\n";
  k << "    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;\n";
  k << "    const unsigned int ID = gid / points;\n";
  k << "    const unsigned int T  = gid % points;\n";
  k << "    if (ID >= antennas) return;\n\n";

  if (!script.defs.empty()) {
    k << "    // --- [Defs] ---\n";
    for (const auto& def : script.defs)
      k << "    " << PrepareExpression(def) << "\n";
    k << "\n";
  }

  k << "    // --- [Signal] ---\n";
  for (const auto& sig : script.signal_lines)
    k << "    " << PrepareExpression(sig) << "\n";
  k << "\n";

  // Output mapping: float2_t instead of OpenCL (float2)
  k << "    float2_t out;\n";
  if (script.has_res_re && script.has_res_im) {
    k << "    out.x = res_re; out.y = res_im;\n";
  } else if (script.has_res) {
    k << "    out.x = res; out.y = 0.0f;\n";
  } else if (script.has_res_re) {
    k << "    out.x = res_re; out.y = 0.0f;\n";
  } else {
    k << "    out.x = res; out.y = 0.0f;\n";
  }
  k << "    output[gid] = out;\n";
  k << "}\n";
  return k.str();
}

std::string ScriptGeneratorROCm::PrepareExpression(const std::string& line) {
  std::string expr = Trim(line);
  auto cp = expr.find("//");
  if (cp != std::string::npos) expr = Trim(expr.substr(0, cp));
  if (expr.empty()) return "";

  // Check for assignment
  bool is_assign = false;
  size_t eq_pos = std::string::npos;
  for (size_t i = 0; i < expr.size(); ++i) {
    if (expr[i] == '=') {
      bool cmp = false;
      if (i + 1 < expr.size() && expr[i+1] == '=') cmp = true;
      if (i > 0 && (expr[i-1]=='!'||expr[i-1]=='<'||expr[i-1]=='>')) cmp = true;
      if (!cmp) { eq_pos = i; is_assign = true; break; }
    }
  }

  if (is_assign && eq_pos != std::string::npos) {
    std::string lhs = Trim(expr.substr(0, eq_pos));
    bool has_type = false;
    const char* types[] = {"float","int","uint","double"};
    for (auto t : types) {
      size_t tl = strlen(t);
      if (lhs.size() > tl && lhs.substr(0,tl)==t && (lhs[tl]==' '||lhs[tl]=='\t'))
      { has_type = true; break; }
    }
    if (!has_type) expr = "float " + expr;
  }

  if (!expr.empty() && expr.back() != ';') expr += ";";
  return expr;
}

// ============================================================================
// Kernel Compilation (hiprtc)
// ============================================================================

void ScriptGeneratorROCm::CompileKernel(const std::string& source) {
  hiprtcProgram prog;
  hiprtcResult rtc = hiprtcCreateProgram(&prog, source.c_str(),
      "script_signal.hip", 0, nullptr, nullptr);
  if (rtc != HIPRTC_SUCCESS)
    throw std::runtime_error("ScriptGeneratorROCm: hiprtcCreateProgram failed");

  const char* opts[] = {"-O3", "-std=c++17"};
  rtc = hiprtcCompileProgram(prog, 2, opts);
  if (rtc != HIPRTC_SUCCESS) {
    size_t log_size = 0;
    hiprtcGetProgramLogSize(prog, &log_size);
    std::string log(log_size, '\0');
    hiprtcGetProgramLog(prog, &log[0]);
    hiprtcDestroyProgram(&prog);
    throw std::runtime_error(
        "ScriptGeneratorROCm: compilation failed!\n=== Source ===\n" +
        source + "\n=== Log ===\n" + log);
  }

  size_t code_size = 0;
  hiprtcGetCodeSize(prog, &code_size);
  std::vector<char> code(code_size);
  hiprtcGetCode(prog, code.data());
  hiprtcDestroyProgram(&prog);

  hipError_t err = hipModuleLoadData(&module_, code.data());
  if (err != hipSuccess)
    throw std::runtime_error("ScriptGeneratorROCm: hipModuleLoadData failed");

  err = hipModuleGetFunction(&kernel_fn_, module_, "script_signal");
  if (err != hipSuccess) {
    hipModuleUnload(module_); module_ = nullptr;
    throw std::runtime_error("ScriptGeneratorROCm: hipModuleGetFunction failed");
  }
}

void ScriptGeneratorROCm::ReleaseGpuResources() {
  if (module_) { hipModuleUnload(module_); module_ = nullptr; kernel_fn_ = nullptr; }
}

// ============================================================================
// Utilities
// ============================================================================

std::string ScriptGeneratorROCm::Trim(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  size_t b = s.find_last_not_of(" \t\r\n");
  return (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
}

std::string ScriptGeneratorROCm::ToUpper(const std::string& s) {
  std::string r = s;
  std::transform(r.begin(), r.end(), r.begin(),
      [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return r;
}

}  // namespace signal_gen

#endif  // ENABLE_ROCM
