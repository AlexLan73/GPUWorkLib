/**
 * @file delayed_form_signal_generator.cpp
 * @brief Реализация DelayedFormSignalGenerator — Farrow 48×5 дробная задержка
 *
 * Алгоритм:
 *   1. FormSignalGenerator генерирует чистый сигнал (noise=0)
 *   2. Kernel apply_fractional_delay: целый сдвиг + 5-точечная Lagrange
 *      интерполяция + шум (Philox + Box-Muller)
 *
 * Матрица 48×5 встроена в код (из lagrange_matrix_48x5.json).
 * Загрузка из JSON поддерживается через LoadMatrix().
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-17
 */

#include "generators/delayed_form_signal_generator.hpp"
#include "kernel_loader.hpp"
#include <stdexcept>
#include <cmath>
#include <chrono>
#include <cstring>
#include <fstream>
#include <sstream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace signal_gen {
namespace {

// Сохранить cl_event для профилирования или освободить (production path).
// Ключевое правило: вызывать ПОСЛЕ того как event использован как wait-dependency.
void CollectOrRelease(cl_event ev, const char* name,
                      DelayedFormSignalGenerator::ProfEvents* prof_events) {
  if (!ev) return;
  if (prof_events) prof_events->push_back({name, ev});
  else clReleaseEvent(ev);
}

}  // namespace

// ════════════════════════════════════════════════════════════════════════════
// Встроенная матрица Lagrange 48×5 (из lagrange_matrix_48x5.json)
// ════════════════════════════════════════════════════════════════════════════

static const float kBuiltinLagrangeMatrix[48 * 5] = {
  // Row 0  (μ = 0/48)
   0.0f,     1.0f,     0.0f,     0.0f,     0.0f,
  // Row 1  (μ = 1/48)
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

// ════════════════════════════════════════════════════════════════════════════
// Конструктор / Деструктор
// ════════════════════════════════════════════════════════════════════════════

DelayedFormSignalGenerator::DelayedFormSignalGenerator(
    drv_gpu_lib::IBackend* backend)
    : backend_(backend)
    , signal_gen_(backend) {

  if (!backend_ || !backend_->IsInitialized()) {
    throw std::runtime_error(
        "DelayedFormSignalGenerator: backend is null or not initialized");
  }

  context_ = static_cast<cl_context>(backend_->GetNativeContext());
  queue_   = static_cast<cl_command_queue>(backend_->GetNativeQueue());
  device_  = static_cast<cl_device_id>(backend_->GetNativeDevice());

  // Загрузить встроенную матрицу
  lagrange_matrix_.assign(kBuiltinLagrangeMatrix,
                          kBuiltinLagrangeMatrix + 48 * 5);
  matrix_loaded_ = true;

  CompileDelayKernel();
  UploadMatrix();
}

DelayedFormSignalGenerator::~DelayedFormSignalGenerator() {
  ReleaseGpuResources();
}

DelayedFormSignalGenerator::DelayedFormSignalGenerator(
    DelayedFormSignalGenerator&& other) noexcept
    : backend_(other.backend_)
    , params_(other.params_)
    , delay_us_(std::move(other.delay_us_))
    , lagrange_matrix_(std::move(other.lagrange_matrix_))
    , matrix_loaded_(other.matrix_loaded_)
    , context_(other.context_)
    , queue_(other.queue_)
    , device_(other.device_)
    , delay_program_(other.delay_program_)
    , matrix_buf_(other.matrix_buf_)
    , signal_gen_(std::move(other.signal_gen_)) {
  other.delay_program_ = nullptr;
  other.matrix_buf_ = nullptr;
}

DelayedFormSignalGenerator& DelayedFormSignalGenerator::operator=(
    DelayedFormSignalGenerator&& other) noexcept {
  if (this != &other) {
    ReleaseGpuResources();
    backend_ = other.backend_;
    params_ = other.params_;
    delay_us_ = std::move(other.delay_us_);
    lagrange_matrix_ = std::move(other.lagrange_matrix_);
    matrix_loaded_ = other.matrix_loaded_;
    context_ = other.context_;
    queue_ = other.queue_;
    device_ = other.device_;
    delay_program_ = other.delay_program_;
    matrix_buf_ = other.matrix_buf_;
    signal_gen_ = std::move(other.signal_gen_);
    other.delay_program_ = nullptr;
    other.matrix_buf_ = nullptr;
  }
  return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// SetParams / SetDelays / LoadMatrix
// ════════════════════════════════════════════════════════════════════════════

void DelayedFormSignalGenerator::SetParams(const FormParams& params) {
  params_ = params;
}

void DelayedFormSignalGenerator::SetParamsFromString(
    const std::string& params_str) {
  params_ = FormParams::ParseFromString(params_str);
}

void DelayedFormSignalGenerator::SetDelays(const std::vector<float>& delay_us) {
  delay_us_ = delay_us;
}

void DelayedFormSignalGenerator::LoadMatrix(const std::string& json_path) {
  std::ifstream file(json_path);
  if (!file.is_open()) {
    throw std::runtime_error(
        "DelayedFormSignalGenerator::LoadMatrix: cannot open " + json_path);
  }

  std::string content((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());

  // Простой парсер JSON: ищем "data": [...] и извлекаем числа
  lagrange_matrix_.clear();

  auto data_pos = content.find("\"data\"");
  if (data_pos == std::string::npos) {
    throw std::runtime_error(
        "DelayedFormSignalGenerator::LoadMatrix: no 'data' key in JSON");
  }

  // Находим начало массива [[
  auto arr_start = content.find('[', data_pos);
  if (arr_start == std::string::npos) {
    throw std::runtime_error(
        "DelayedFormSignalGenerator::LoadMatrix: malformed JSON");
  }

  // Извлекаем все числа из массива
  std::string nums;
  for (size_t i = arr_start; i < content.size(); ++i) {
    char c = content[i];
    if (c == '-' || c == '.' || (c >= '0' && c <= '9') || c == 'e' || c == 'E' || c == '+') {
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
        "DelayedFormSignalGenerator::LoadMatrix: expected 240 values, got "
        + std::to_string(lagrange_matrix_.size()));
  }

  matrix_loaded_ = true;
  UploadMatrix();
}

// ════════════════════════════════════════════════════════════════════════════
// GPU генерация: сигнал + задержка + шум
// ════════════════════════════════════════════════════════════════════════════

drv_gpu_lib::InputData<cl_mem>
DelayedFormSignalGenerator::GenerateInputData() {
  return GenerateInputData(nullptr);
}

drv_gpu_lib::InputData<cl_mem>
DelayedFormSignalGenerator::GenerateInputData(ProfEvents* prof_events) {
  if (params_.antennas == 0 || params_.points == 0) {
    throw std::runtime_error(
        "DelayedFormSignalGenerator::GenerateInputData: antennas or points is 0");
  }

  // Проверка delay_us
  if (delay_us_.empty()) {
    // Если задержки не заданы — заполняем нулями
    delay_us_.assign(params_.antennas, 0.0f);
  }

  if (delay_us_.size() != params_.antennas) {
    throw std::invalid_argument(
        "DelayedFormSignalGenerator: delay_us.size()="
        + std::to_string(delay_us_.size())
        + " != antennas=" + std::to_string(params_.antennas));
  }

  // ── Шаг 1: Генерация чистого сигнала (noise=0) ──
  // prof_events пробрасывается: FormSignalGenerator добавит "Kernel" (FormSignal stage)
  FormParams clean_params = params_;
  clean_params.noise_amplitude = 0.0;
  clean_params.tau_base = 0.0;
  clean_params.tau_step = 0.0;
  clean_params.tau_min = 0.0;
  clean_params.tau_max = 0.0;
  signal_gen_.SetParams(clean_params);

  auto clean_signal = signal_gen_.GenerateInputData(prof_events);
  cl_mem input_buf = clean_signal.data;

  // ── Шаг 2: Применение задержки + шум ──
  size_t total_points = GetTotalSamples();
  size_t buffer_size = total_points * sizeof(std::complex<float>);

  cl_int err;

  // Создать выходной буфер
  cl_mem output_buf = clCreateBuffer(
      context_, CL_MEM_READ_WRITE, buffer_size, nullptr, &err);
  if (err != CL_SUCCESS) {
    clReleaseMemObject(input_buf);
    throw std::runtime_error(
        "DelayedFormSignalGenerator: clCreateBuffer(output) failed: "
        + std::to_string(err));
  }

  // Загрузить delay_us на GPU (CL_MEM_COPY_HOST_PTR — синхронный, event не нужен)
  cl_mem delay_buf = clCreateBuffer(
      context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      delay_us_.size() * sizeof(float),
      const_cast<float*>(delay_us_.data()), &err);
  if (err != CL_SUCCESS) {
    clReleaseMemObject(input_buf);
    clReleaseMemObject(output_buf);
    throw std::runtime_error(
        "DelayedFormSignalGenerator: clCreateBuffer(delay) failed: "
        + std::to_string(err));
  }

  // Создать kernel
  cl_kernel k = clCreateKernel(delay_program_, "apply_fractional_delay", &err);
  if (err != CL_SUCCESS) {
    clReleaseMemObject(input_buf);
    clReleaseMemObject(output_buf);
    clReleaseMemObject(delay_buf);
    throw std::runtime_error(
        "DelayedFormSignalGenerator: clCreateKernel failed: "
        + std::to_string(err));
  }

  // Параметры
  uint32_t ant = params_.antennas;
  uint32_t pts = params_.points;
  float sr = static_cast<float>(params_.fs);
  float an = static_cast<float>(params_.noise_amplitude);
  float norm_val = static_cast<float>(params_.norm);

  uint32_t noise_seed = params_.noise_seed;
  if (noise_seed == 0 && an > 0.0f) {
    noise_seed = static_cast<uint32_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()
        & 0xFFFFFFFF);
  }

  // Установить аргументы
  err  = clSetKernelArg(k, 0, sizeof(cl_mem),   &input_buf);
  err |= clSetKernelArg(k, 1, sizeof(cl_mem),   &output_buf);
  err |= clSetKernelArg(k, 2, sizeof(cl_mem),   &matrix_buf_);
  err |= clSetKernelArg(k, 3, sizeof(cl_mem),   &delay_buf);
  err |= clSetKernelArg(k, 4, sizeof(uint32_t), &ant);
  err |= clSetKernelArg(k, 5, sizeof(uint32_t), &pts);
  err |= clSetKernelArg(k, 6, sizeof(float),    &sr);
  err |= clSetKernelArg(k, 7, sizeof(float),    &an);
  err |= clSetKernelArg(k, 8, sizeof(float),    &norm_val);
  err |= clSetKernelArg(k, 9, sizeof(uint32_t), &noise_seed);

  if (err != CL_SUCCESS) {
    clReleaseKernel(k);
    clReleaseMemObject(input_buf);
    clReleaseMemObject(output_buf);
    clReleaseMemObject(delay_buf);
    throw std::runtime_error(
        "DelayedFormSignalGenerator: clSetKernelArg failed");
  }

  // 2D grid: dim0 = samples, dim1 = antennas (eliminates div/mod in kernel)
  size_t local_size[2]  = { 256, 1 };
  size_t global_size[2] = {
      ((static_cast<size_t>(params_.points) + 255) / 256) * 256,
      static_cast<size_t>(params_.antennas)
  };

  cl_event ev_delay = nullptr;
  err = clEnqueueNDRangeKernel(
      queue_, k, 2, nullptr,
      global_size, local_size, 0, nullptr, prof_events ? &ev_delay : nullptr);

  clReleaseKernel(k);
  clReleaseMemObject(input_buf);
  clReleaseMemObject(delay_buf);

  if (err != CL_SUCCESS) {
    clReleaseMemObject(output_buf);
    throw std::runtime_error(
        "DelayedFormSignalGenerator: enqueue failed: "
        + std::to_string(err));
  }

  CollectOrRelease(ev_delay, "FarrowDelay", prof_events);

  clFinish(queue_);

  drv_gpu_lib::InputData<cl_mem> result;
  result.antenna_count = params_.antennas;
  result.n_point       = params_.points;
  result.data          = output_buf;
  result.gpu_memory_bytes = buffer_size;
  result.sample_rate   = static_cast<float>(params_.fs);
  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// CPU генерация
// ════════════════════════════════════════════════════════════════════════════

std::vector<std::vector<std::complex<float>>>
DelayedFormSignalGenerator::GenerateToCpu() {
  auto input = GenerateInputData();
  cl_mem gpu_buf = input.data;

  size_t total = GetTotalSamples();
  std::vector<std::complex<float>> flat(total);

  cl_int err = clEnqueueReadBuffer(
      queue_, gpu_buf, CL_TRUE, 0,
      total * sizeof(std::complex<float>),
      flat.data(), 0, nullptr, nullptr);
  clReleaseMemObject(gpu_buf);

  if (err != CL_SUCCESS) {
    throw std::runtime_error(
        "DelayedFormSignalGenerator::GenerateToCpu: read failed");
  }

  std::vector<std::vector<std::complex<float>>> result(params_.antennas);
  for (uint32_t a = 0; a < params_.antennas; ++a) {
    size_t offset = static_cast<size_t>(a) * params_.points;
    result[a].assign(
        flat.begin() + offset,
        flat.begin() + offset + params_.points);
  }

  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// GPU internals
// ════════════════════════════════════════════════════════════════════════════

void DelayedFormSignalGenerator::CompileDelayKernel() {
  // Load kernel from .cl files: prng.cl + delayed_form_signal.cl
  std::string source = LoadKernelWithPrng("delayed_form_signal.cl");
  const char* src_ptr = source.c_str();
  size_t source_len = source.size();

  cl_int err;
  delay_program_ = clCreateProgramWithSource(
      context_, 1, &src_ptr, &source_len, &err);
  if (err != CL_SUCCESS) {
    throw std::runtime_error(
        "DelayedFormSignalGenerator: clCreateProgramWithSource failed");
  }

  err = clBuildProgram(
      delay_program_, 1, &device_, "-cl-fast-relaxed-math", nullptr, nullptr);
  if (err != CL_SUCCESS) {
    size_t log_size;
    clGetProgramBuildInfo(
        delay_program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    std::vector<char> log(log_size);
    clGetProgramBuildInfo(
        delay_program_, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(),
        nullptr);
    clReleaseProgram(delay_program_);
    delay_program_ = nullptr;
    throw std::runtime_error(
        "DelayedFormSignalGenerator kernel build failed:\n"
        + std::string(log.data()));
  }
}

void DelayedFormSignalGenerator::UploadMatrix() {
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
        "DelayedFormSignalGenerator: upload matrix failed: "
        + std::to_string(err));
  }
}

void DelayedFormSignalGenerator::ReleaseGpuResources() {
  if (delay_program_) {
    clReleaseProgram(delay_program_);
    delay_program_ = nullptr;
  }
  if (matrix_buf_) {
    clReleaseMemObject(matrix_buf_);
    matrix_buf_ = nullptr;
  }
}

} // namespace signal_gen
