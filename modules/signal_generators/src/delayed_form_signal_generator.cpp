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
#include "prof_utils.hpp"
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

// ════════════════════════════════════════════════════════════════════════════
// Встроенная матрица Lagrange 48×5 (из lagrange_matrix_48x5.json)
// ════════════════════════════════════════════════════════════════════════════

// 4-точечная кубическая Lagrange-интерполяция (cubic Farrow filter)
// Узлы: x ∈ {-1, 0, 1, 2} → сэмплы center-1, center, center+1, center+2
// Строка k: μ = k/48, k = 0..47
// Формулы:
//   L₀(μ) = -μ(μ-1)(μ-2)/6
//   L₁(μ) = (μ+1)(μ-1)(μ-2)/2
//   L₂(μ) = -(μ+1)μ(μ-2)/2
//   L₃(μ) = (μ+1)μ(μ-1)/6
//   L₄(μ) = 0  (5-й tap не используется)
// Генерация: tools/compute_lagrange_matrix.py
static const float kBuiltinLagrangeMatrix[48 * 5] = {
  -0.00000000f,  1.00000000f,  0.00000000f, -0.00000000f,  0.00000000f,  // Row  0  (μ = 0/48)
  -0.00672894f,  0.98915383f,  0.02104583f, -0.00347072f,  0.00000000f,  // Row  1  (μ = 1/48)
  -0.01303289f,  0.97746672f,  0.04249855f, -0.00693239f,  0.00000000f,  // Row  2  (μ = 2/48)
  -0.01892090f,  0.96496582f,  0.06433105f, -0.01037598f,  0.00000000f,  // Row  3  (μ = 3/48)
  -0.02440201f,  0.95167824f,  0.08651620f, -0.01379244f,  0.00000000f,  // Row  4  (μ = 4/48)
  -0.02948526f,  0.93763111f,  0.10902687f, -0.01717273f,  0.00000000f,  // Row  5  (μ = 5/48)
  -0.03417969f,  0.92285156f,  0.13183594f, -0.02050781f,  0.00000000f,  // Row  6  (μ = 6/48)
  -0.03849435f,  0.90736672f,  0.15491627f, -0.02378864f,  0.00000000f,  // Row  7  (μ = 7/48)
  -0.04243827f,  0.89120370f,  0.17824074f, -0.02700617f,  0.00000000f,  // Row  8  (μ = 8/48)
  -0.04602051f,  0.87438965f,  0.20178223f, -0.03015137f,  0.00000000f,  // Row  9  (μ = 9/48)
  -0.04925010f,  0.85695168f,  0.22551360f, -0.03321518f,  0.00000000f,  // Row 10  (μ = 10/48)
  -0.05213608f,  0.83891692f,  0.24940773f, -0.03618857f,  0.00000000f,  // Row 11  (μ = 11/48)
  -0.05468750f,  0.82031250f,  0.27343750f, -0.03906250f,  0.00000000f,  // Row 12  (μ = 12/48)
  -0.05691340f,  0.80116555f,  0.29757577f, -0.04182792f,  0.00000000f,  // Row 13  (μ = 13/48)
  -0.05882282f,  0.78150318f,  0.32179543f, -0.04447579f,  0.00000000f,  // Row 14  (μ = 14/48)
  -0.06042480f,  0.76135254f,  0.34606934f, -0.04699707f,  0.00000000f,  // Row 15  (μ = 15/48)
  -0.06172840f,  0.74074074f,  0.37037037f, -0.04938272f,  0.00000000f,  // Row 16  (μ = 16/48)
  -0.06274263f,  0.71969491f,  0.39467140f, -0.05162369f,  0.00000000f,  // Row 17  (μ = 17/48)
  -0.06347656f,  0.69824219f,  0.41894531f, -0.05371094f,  0.00000000f,  // Row 18  (μ = 18/48)
  -0.06393922f,  0.67640969f,  0.44316497f, -0.05563543f,  0.00000000f,  // Row 19  (μ = 19/48)
  -0.06413966f,  0.65422454f,  0.46730324f, -0.05738812f,  0.00000000f,  // Row 20  (μ = 20/48)
  -0.06408691f,  0.63171387f,  0.49133301f, -0.05895996f,  0.00000000f,  // Row 21  (μ = 21/48)
  -0.06379003f,  0.60890480f,  0.51522714f, -0.06034192f,  0.00000000f,  // Row 22  (μ = 22/48)
  -0.06325804f,  0.58582447f,  0.53895851f, -0.06152494f,  0.00000000f,  // Row 23  (μ = 23/48)
  -0.06250000f,  0.56250000f,  0.56250000f, -0.06250000f,  0.00000000f,  // Row 24  (μ = 24/48 = 0.5)
  -0.06152494f,  0.53895851f,  0.58582447f, -0.06325804f,  0.00000000f,  // Row 25  (μ = 25/48)
  -0.06034192f,  0.51522714f,  0.60890480f, -0.06379003f,  0.00000000f,  // Row 26  (μ = 26/48)
  -0.05895996f,  0.49133301f,  0.63171387f, -0.06408691f,  0.00000000f,  // Row 27  (μ = 27/48)
  -0.05738812f,  0.46730324f,  0.65422454f, -0.06413966f,  0.00000000f,  // Row 28  (μ = 28/48)
  -0.05563543f,  0.44316497f,  0.67640969f, -0.06393922f,  0.00000000f,  // Row 29  (μ = 29/48)
  -0.05371094f,  0.41894531f,  0.69824219f, -0.06347656f,  0.00000000f,  // Row 30  (μ = 30/48)
  -0.05162369f,  0.39467140f,  0.71969491f, -0.06274263f,  0.00000000f,  // Row 31  (μ = 31/48)
  -0.04938272f,  0.37037037f,  0.74074074f, -0.06172840f,  0.00000000f,  // Row 32  (μ = 32/48)
  -0.04699707f,  0.34606934f,  0.76135254f, -0.06042480f,  0.00000000f,  // Row 33  (μ = 33/48)
  -0.04447579f,  0.32179543f,  0.78150318f, -0.05882282f,  0.00000000f,  // Row 34  (μ = 34/48)
  -0.04182792f,  0.29757577f,  0.80116555f, -0.05691340f,  0.00000000f,  // Row 35  (μ = 35/48)
  -0.03906250f,  0.27343750f,  0.82031250f, -0.05468750f,  0.00000000f,  // Row 36  (μ = 36/48)
  -0.03618857f,  0.24940773f,  0.83891692f, -0.05213608f,  0.00000000f,  // Row 37  (μ = 37/48)
  -0.03321518f,  0.22551360f,  0.85695168f, -0.04925010f,  0.00000000f,  // Row 38  (μ = 38/48)
  -0.03015137f,  0.20178223f,  0.87438965f, -0.04602051f,  0.00000000f,  // Row 39  (μ = 39/48)
  -0.02700617f,  0.17824074f,  0.89120370f, -0.04243827f,  0.00000000f,  // Row 40  (μ = 40/48)
  -0.02378864f,  0.15491627f,  0.90736672f, -0.03849435f,  0.00000000f,  // Row 41  (μ = 41/48)
  -0.02050781f,  0.13183594f,  0.92285156f, -0.03417969f,  0.00000000f,  // Row 42  (μ = 42/48)
  -0.01717273f,  0.10902687f,  0.93763111f, -0.02948526f,  0.00000000f,  // Row 43  (μ = 43/48)
  -0.01379244f,  0.08651620f,  0.95167824f, -0.02440201f,  0.00000000f,  // Row 44  (μ = 44/48)
  -0.01037598f,  0.06433105f,  0.96496582f, -0.01892090f,  0.00000000f,  // Row 45  (μ = 45/48)
  -0.00693239f,  0.04249855f,  0.97746672f, -0.01303289f,  0.00000000f,  // Row 46  (μ = 46/48)
  -0.00347072f,  0.02104583f,  0.98915383f, -0.00672894f,  0.00000000f,  // Row 47  (μ = 47/48)
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
