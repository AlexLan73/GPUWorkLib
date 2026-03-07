/**
 * @file lch_farrow.cpp
 * @brief LchFarrow - fractional delay processor (Lagrange 48x5) on GPU
 *
 * Algorithm (DelayedFormSignal_Kernel_CORRECT):
 *   read_pos = n - delay_samples
 *   center = floor(read_pos), frac = read_pos - center
 *   row = (uint)(frac * 48) % 48
 *   output[n] = sum(L[row][k] * input[center - 1 + k], k=0..4)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-18
 */

#include "lch_farrow.hpp"
#include "kernels/kernel_loader.hpp"

#include <stdexcept>
#include <cmath>
#include <chrono>
#include <cstring>
#include <fstream>
#include <sstream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace lch_farrow {

// ════════════════════════════════════════════════════════════════════════════
// Built-in Lagrange matrix 48x5 (from lagrange_matrix_48x5.json)
// ════════════════════════════════════════════════════════════════════════════

static const float kBuiltinLagrangeMatrix[48 * 5] = {
  // Row 0  (mu = 0/48)
   0.0f,     1.0f,     0.0f,     0.0f,     0.0f,
  // Row 1  (mu = 1/48)
  -0.0052f,  1.0417f, -0.0417f,  0.0052f,  0.0f,
  // Row 2
  -0.01f,    1.08f,   -0.08f,    0.01f,    0.0f,
  // Row 3
  -0.0143f,  1.1143f, -0.1143f,  0.0143f,  0.0f,
  // Row 4
  -0.018f,   1.144f,  -0.144f,   0.018f,   0.0f,
  // Row 5
  -0.0208f,  1.1667f, -0.1667f,  0.0208f,  0.0f,
  // Row 6
  -0.0228f,  1.1827f, -0.1827f,  0.0228f,  0.0f,
  // Row 7
  -0.0239f,  1.1914f, -0.1914f,  0.0239f,  0.0f,
  // Row 8
  -0.024f,   1.2f,    -0.2f,     0.024f,   0.0f,
  // Row 9
  -0.0231f,  1.1914f, -0.1914f,  0.0231f,  0.0f,
  // Row 10
  -0.0208f,  1.1667f, -0.1667f,  0.0208f,  0.0f,
  // Row 11
  -0.0169f,  1.1198f, -0.1198f,  0.0169f,  0.0f,
  // Row 12
  -0.0111f,  1.0432f, -0.0432f,  0.0111f,  0.0f,
  // Row 13
   0.0026f,  0.9323f,  0.0677f, -0.0026f,  0.0f,
  // Row 14
   0.0229f,  0.7812f,  0.2188f, -0.0229f,  0.0f,
  // Row 15
   0.0507f,  0.5823f,  0.4177f, -0.0507f,  0.0f,
  // Row 16
   0.0859f,  0.3281f,  0.6719f, -0.0859f,  0.0f,
  // Row 17
   0.1276f,  0.0104f,  0.9896f, -0.1276f,  0.0f,
  // Row 18
   0.175f,  -0.3802f,  1.3802f, -0.175f,   0.0f,
  // Row 19
   0.2274f, -0.8385f,  1.8385f, -0.2274f,  0.0f,
  // Row 20
   0.2839f, -1.3567f,  2.3567f, -0.2839f,  0.0f,
  // Row 21
   0.3438f, -1.9375f,  2.9375f, -0.3438f,  0.0f,
  // Row 22
   0.4063f, -2.5846f,  3.5846f, -0.4063f,  0.0f,
  // Row 23
   0.4705f, -3.2917f,  4.2917f, -0.4705f,  0.0f,
  // Row 24
   0.5355f, -4.0521f,  5.0521f, -0.5355f,  0.0f,
  // Row 25
   0.6f,    -4.8594f,  5.8594f, -0.6f,     0.0f,
  // Row 26
   0.6628f, -5.7083f,  6.7083f, -0.6628f,  0.0f,
  // Row 27
   0.7227f, -6.592f,   7.592f,  -0.7227f,  0.0f,
  // Row 28
   0.7786f, -7.5052f,  8.5052f, -0.7786f,  0.0f,
  // Row 29
   0.8293f, -8.4411f,  9.4411f, -0.8293f,  0.0f,
  // Row 30
   0.8734f, -9.3937f, 10.3937f, -0.8734f,  0.0f,
  // Row 31
   0.9102f,-10.3567f, 11.3567f, -0.9102f,  0.0f,
  // Row 32
   0.9384f,-11.3229f, 12.3229f, -0.9384f,  0.0f,
  // Row 33
   0.957f, -12.2857f, 13.2857f, -0.957f,   0.0f,
  // Row 34
   0.9648f,-13.2386f, 14.2386f, -0.9648f,  0.0f,
  // Row 35
   0.9609f,-14.1748f, 15.1748f, -0.9609f,  0.0f,
  // Row 36
   0.9446f,-14.9877f, 16.0f,    -0.9446f,  0.0f,
  // Row 37
   0.9141f,-15.6684f, 16.75f,   -0.9141f,  0.0f,
  // Row 38
   0.8684f,-16.2096f, 17.4219f, -0.8684f,  0.0f,
  // Row 39
   0.8066f,-16.6039f, 18.0078f, -0.8066f,  0.0f,
  // Row 40
   0.7275f,-16.8438f, 18.5f,    -0.7275f,  0.0f,
  // Row 41
   0.6299f,-16.9219f, 19.0f,    -0.6299f,  0.0f,
  // Row 42
   0.5126f,-16.8301f, 19.3984f, -0.5126f,  0.0f,
  // Row 43
   0.3745f,-16.5605f, 19.6875f, -0.3745f,  0.0f,
  // Row 44
   0.2143f,-16.0955f, 19.875f,  -0.2143f,  0.0f,
  // Row 45
   0.0307f,-15.4175f, 20.0f,    -0.0307f,  0.0f,
  // Row 46
  -0.1816f,-14.5086f, 20.0234f,  0.1816f,  0.0f,
  // Row 47
  -0.4347f,-13.3521f, 20.0547f,  0.4347f,  0.0f
};

// OpenCL kernel loaded from kernels/lch_farrow_delay.cl via LoadKernelFile

// ════════════════════════════════════════════════════════════════════════════
// Constructor / Destructor
// ════════════════════════════════════════════════════════════════════════════

LchFarrow::LchFarrow(drv_gpu_lib::IBackend* backend)
    : backend_(backend) {

  if (!backend_ || !backend_->IsInitialized()) {
    throw std::runtime_error(
        "LchFarrow: backend is null or not initialized");
  }

  context_ = static_cast<cl_context>(backend_->GetNativeContext());
  queue_   = static_cast<cl_command_queue>(backend_->GetNativeQueue());
  device_  = static_cast<cl_device_id>(backend_->GetNativeDevice());

  // Load built-in matrix
  lagrange_matrix_.assign(kBuiltinLagrangeMatrix,
                          kBuiltinLagrangeMatrix + 48 * 5);
  matrix_loaded_ = true;

  CompileKernel();
  UploadMatrix();
}

LchFarrow::~LchFarrow() {
  ReleaseGpuResources();
}

LchFarrow::LchFarrow(LchFarrow&& other) noexcept
    : backend_(other.backend_)
    , delay_us_(std::move(other.delay_us_))
    , sample_rate_(other.sample_rate_)
    , noise_amplitude_(other.noise_amplitude_)
    , norm_val_(other.norm_val_)
    , noise_seed_(other.noise_seed_)
    , lagrange_matrix_(std::move(other.lagrange_matrix_))
    , matrix_loaded_(other.matrix_loaded_)
    , context_(other.context_)
    , queue_(other.queue_)
    , device_(other.device_)
    , program_(other.program_)
    , kernel_(other.kernel_)
    , matrix_buf_(other.matrix_buf_)
    , delay_buf_(other.delay_buf_)
    , delay_buf_size_(other.delay_buf_size_) {
  other.program_ = nullptr;
  other.kernel_ = nullptr;
  other.matrix_buf_ = nullptr;
  other.delay_buf_ = nullptr;
  other.delay_buf_size_ = 0;
}

LchFarrow& LchFarrow::operator=(LchFarrow&& other) noexcept {
  if (this != &other) {
    ReleaseGpuResources();
    backend_ = other.backend_;
    delay_us_ = std::move(other.delay_us_);
    sample_rate_ = other.sample_rate_;
    noise_amplitude_ = other.noise_amplitude_;
    norm_val_ = other.norm_val_;
    noise_seed_ = other.noise_seed_;
    lagrange_matrix_ = std::move(other.lagrange_matrix_);
    matrix_loaded_ = other.matrix_loaded_;
    context_ = other.context_;
    queue_ = other.queue_;
    device_ = other.device_;
    program_ = other.program_;
    kernel_ = other.kernel_;
    matrix_buf_ = other.matrix_buf_;
    delay_buf_ = other.delay_buf_;
    delay_buf_size_ = other.delay_buf_size_;
    other.program_ = nullptr;
    other.kernel_ = nullptr;
    other.matrix_buf_ = nullptr;
    other.delay_buf_ = nullptr;
    other.delay_buf_size_ = 0;
  }
  return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// Setters
// ════════════════════════════════════════════════════════════════════════════

void LchFarrow::SetDelays(const std::vector<float>& delay_us) {
  delay_us_ = delay_us;
}

void LchFarrow::SetSampleRate(float sample_rate) {
  sample_rate_ = sample_rate;
}

void LchFarrow::SetNoise(float noise_amplitude, float norm_val,
                          uint32_t noise_seed) {
  noise_amplitude_ = noise_amplitude;
  norm_val_ = norm_val;
  noise_seed_ = noise_seed;
}

void LchFarrow::LoadMatrix(const std::string& json_path) {
  std::ifstream file(json_path);
  if (!file.is_open()) {
    throw std::runtime_error(
        "LchFarrow::LoadMatrix: cannot open " + json_path);
  }

  std::string content((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());

  lagrange_matrix_.clear();

  auto data_pos = content.find("\"data\"");
  if (data_pos == std::string::npos) {
    throw std::runtime_error(
        "LchFarrow::LoadMatrix: no 'data' key in JSON");
  }

  auto arr_start = content.find('[', data_pos);
  if (arr_start == std::string::npos) {
    throw std::runtime_error("LchFarrow::LoadMatrix: malformed JSON");
  }

  std::string nums;
  for (size_t i = arr_start; i < content.size(); ++i) {
    char c = content[i];
    if (c == '-' || c == '.' || (c >= '0' && c <= '9') ||
        c == 'e' || c == 'E' || c == '+') {
      nums += c;
    } else if (!nums.empty()) {
      lagrange_matrix_.push_back(std::stof(nums));
      nums.clear();
    }
  }
  if (!nums.empty()) {
    lagrange_matrix_.push_back(std::stof(nums));
  }

  if (lagrange_matrix_.size() != 48 * 5) {
    throw std::runtime_error(
        "LchFarrow::LoadMatrix: expected 240 values, got "
        + std::to_string(lagrange_matrix_.size()));
  }

  matrix_loaded_ = true;
  UploadMatrix();
}

// ════════════════════════════════════════════════════════════════════════════
// GPU Processing
// ════════════════════════════════════════════════════════════════════════════

// Helper: сохранить cl_event в prof_events или освободить (ноль overhead в production)
// Вызывать ПОСЛЕ того как event использован как wait в следующей операции
static void CollectOrRelease(cl_event ev, const char* name,
    lch_farrow::ProfEvents* prof_events)
{
  if (!ev) return;
  if (prof_events) {
    prof_events->push_back({name, ev});
  } else {
    clReleaseEvent(ev);
  }
}

drv_gpu_lib::InputData<cl_mem>
LchFarrow::Process(cl_mem input_buf, uint32_t antennas, uint32_t points,
                   ProfEvents* prof_events) {
  if (antennas == 0 || points == 0) {
    throw std::runtime_error("LchFarrow::Process: antennas or points is 0");
  }

  if (delay_us_.empty()) {
    delay_us_.assign(antennas, 0.0f);
  }

  if (delay_us_.size() != antennas) {
    throw std::invalid_argument(
        "LchFarrow: delay_us.size()=" + std::to_string(delay_us_.size())
        + " != antennas=" + std::to_string(antennas));
  }

  size_t total_points = static_cast<size_t>(antennas) * points;
  size_t buffer_size = total_points * sizeof(std::complex<float>);

  cl_int err;

  // Output buffer
  cl_mem output_buf = clCreateBuffer(
      context_, CL_MEM_READ_WRITE, buffer_size, nullptr, &err);
  if (err != CL_SUCCESS) {
    throw std::runtime_error(
        "LchFarrow: clCreateBuffer(output) failed: " + std::to_string(err));
  }

  // Persistent delay_buf_: resize only when needed
  size_t delay_size = delay_us_.size() * sizeof(float);
  if (delay_buf_size_ < delay_size) {
    if (delay_buf_) clReleaseMemObject(delay_buf_);
    delay_buf_ = clCreateBuffer(
        context_, CL_MEM_READ_ONLY, delay_size, nullptr, &err);
    if (err != CL_SUCCESS) {
      delay_buf_ = nullptr;
      delay_buf_size_ = 0;
      clReleaseMemObject(output_buf);
      throw std::runtime_error(
          "LchFarrow: clCreateBuffer(delay) failed: " + std::to_string(err));
    }
    delay_buf_size_ = delay_size;
  }

  cl_event upload_event = nullptr;
  err = clEnqueueWriteBuffer(
      queue_, delay_buf_, CL_FALSE, 0,
      delay_size, delay_us_.data(),
      0, nullptr, &upload_event);
  if (err != CL_SUCCESS) {
    clReleaseMemObject(output_buf);
    throw std::runtime_error(
        "LchFarrow: clEnqueueWriteBuffer(delay) failed: " + std::to_string(err));
  }

  // Set arguments (kernel_ created once in CompileKernel)
  uint32_t ant = antennas;
  uint32_t pts = points;
  float sr = sample_rate_;
  float an = noise_amplitude_;
  float nv = norm_val_;

  uint32_t ns = noise_seed_;
  if (ns == 0 && an > 0.0f) {
    ns = static_cast<uint32_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()
        & 0xFFFFFFFF);
  }

  err  = clSetKernelArg(kernel_, 0, sizeof(cl_mem),   &input_buf);
  err |= clSetKernelArg(kernel_, 1, sizeof(cl_mem),   &output_buf);
  err |= clSetKernelArg(kernel_, 2, sizeof(cl_mem),   &matrix_buf_);
  err |= clSetKernelArg(kernel_, 3, sizeof(cl_mem),   &delay_buf_);
  err |= clSetKernelArg(kernel_, 4, sizeof(uint32_t), &ant);
  err |= clSetKernelArg(kernel_, 5, sizeof(uint32_t), &pts);
  err |= clSetKernelArg(kernel_, 6, sizeof(float),    &sr);
  err |= clSetKernelArg(kernel_, 7, sizeof(float),    &an);
  err |= clSetKernelArg(kernel_, 8, sizeof(float),    &nv);
  err |= clSetKernelArg(kernel_, 9, sizeof(uint32_t), &ns);

  if (err != CL_SUCCESS) {
    clReleaseMemObject(output_buf);
    throw std::runtime_error("LchFarrow: clSetKernelArg failed");
  }

  // 2D NDRange: dim0=samples (ceil to local), dim1=antennas
  size_t local_2d[2] = { 256, 1 };
  size_t global_2d[2] = {
      ((static_cast<size_t>(points) + 255) / 256) * 256,
      static_cast<size_t>(antennas)
  };

  cl_event kernel_event = nullptr;
  err = clEnqueueNDRangeKernel(
      queue_, kernel_, 2, nullptr,
      global_2d, local_2d, 1, &upload_event, &kernel_event);

  if (err != CL_SUCCESS) {
    if (upload_event) clReleaseEvent(upload_event);
    if (kernel_event) clReleaseEvent(kernel_event);
    clReleaseMemObject(output_buf);
    throw std::runtime_error(
        "LchFarrow: enqueue failed: " + std::to_string(err));
  }

  // upload_event уже использован как wait — собрать/освободить
  CollectOrRelease(upload_event, "Upload_delay", prof_events);
  CollectOrRelease(kernel_event, "Kernel", prof_events);

  clFinish(queue_);

  drv_gpu_lib::InputData<cl_mem> result;
  result.antenna_count = antennas;
  result.n_point       = points;
  result.data          = output_buf;
  result.gpu_memory_bytes = buffer_size;
  result.sample_rate   = sample_rate_;
  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// CPU Processing (reference)
// ════════════════════════════════════════════════════════════════════════════

std::vector<std::vector<std::complex<float>>>
LchFarrow::ProcessCpu(
    const std::vector<std::vector<std::complex<float>>>& input,
    uint32_t antennas, uint32_t points) {

  if (delay_us_.empty()) {
    delay_us_.assign(antennas, 0.0f);
  }

  std::vector<std::vector<std::complex<float>>> result(antennas);

  for (uint32_t a = 0; a < antennas; ++a) {
    result[a].resize(points, {0.0f, 0.0f});

    float delay_samples = delay_us_[a] * 1e-6f * sample_rate_;
    float L[5];

    for (uint32_t n = 0; n < points; ++n) {
      float read_pos = static_cast<float>(n) - delay_samples;
      if (read_pos < 0.0f) continue;

      int center = static_cast<int>(std::floor(read_pos));
      float frac = read_pos - static_cast<float>(center);
      uint32_t row = (static_cast<uint32_t>(frac * 48.0f)) % 48u;

      for (int k = 0; k < 5; ++k)
        L[k] = lagrange_matrix_[row * 5 + k];

      std::complex<float> val(0.0f, 0.0f);
      for (int k = 0; k < 5; ++k) {
        int idx = center - 1 + k;
        if (idx >= 0 && idx < static_cast<int>(points)) {
          val += L[k] * input[a][idx];
        }
      }
      result[a][n] = val;
    }
  }

  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// GPU internals
// ════════════════════════════════════════════════════════════════════════════

void LchFarrow::CompileKernel() {
  cl_int err;
  std::string kernel_source = LoadKernelFile("lch_farrow_delay.cl");
  const char* source_ptr = kernel_source.c_str();
  size_t source_len = kernel_source.size();

  program_ = clCreateProgramWithSource(
      context_, 1, &source_ptr, &source_len, &err);
  if (err != CL_SUCCESS) {
    throw std::runtime_error(
        "LchFarrow: clCreateProgramWithSource failed");
  }

  err = clBuildProgram(
      program_, 1, &device_, "-cl-fast-relaxed-math", nullptr, nullptr);
  if (err != CL_SUCCESS) {
    size_t log_size;
    clGetProgramBuildInfo(
        program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    std::vector<char> log(log_size);
    clGetProgramBuildInfo(
        program_, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(),
        nullptr);
    clReleaseProgram(program_);
    program_ = nullptr;
    throw std::runtime_error(
        "LchFarrow kernel build failed:\n" + std::string(log.data()));
  }

  // Create kernel once — reuse in Process()
  kernel_ = clCreateKernel(program_, "lch_farrow_delay", &err);
  if (err != CL_SUCCESS) {
    clReleaseProgram(program_);
    program_ = nullptr;
    throw std::runtime_error(
        "LchFarrow: clCreateKernel failed: " + std::to_string(err));
  }
}

void LchFarrow::UploadMatrix() {
  if (matrix_buf_) {
    clReleaseMemObject(matrix_buf_);
    matrix_buf_ = nullptr;
  }

  cl_int err;
  matrix_buf_ = clCreateBuffer(
      context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      lagrange_matrix_.size() * sizeof(float),
      lagrange_matrix_.data(), &err);
  if (err != CL_SUCCESS) {
    throw std::runtime_error(
        "LchFarrow: upload matrix failed: " + std::to_string(err));
  }
}

void LchFarrow::ReleaseGpuResources() {
  if (kernel_) {
    clReleaseKernel(kernel_);
    kernel_ = nullptr;
  }
  if (program_) {
    clReleaseProgram(program_);
    program_ = nullptr;
  }
  if (matrix_buf_) {
    clReleaseMemObject(matrix_buf_);
    matrix_buf_ = nullptr;
  }
  if (delay_buf_) {
    clReleaseMemObject(delay_buf_);
    delay_buf_ = nullptr;
    delay_buf_size_ = 0;
  }
}

} // namespace lch_farrow
