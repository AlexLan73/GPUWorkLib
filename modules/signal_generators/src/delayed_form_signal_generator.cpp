/**
 * @file delayed_form_signal_generator.cpp
 * @brief Реализация DelayedFormSignalGenerator — обёртка FormSignalGenerator + LchFarrow
 *
 * Алгоритм:
 *   1. FormSignalGenerator генерирует чистый сигнал (noise=0, tau=0)
 *   2. LchFarrow::Process() применяет Lagrange 48×5 дробную задержку + шум
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-17
 */

#include "generators/delayed_form_signal_generator.hpp"
#include <stdexcept>

namespace signal_gen {

// ════════════════════════════════════════════════════════════════════════════
// Конструктор
// ════════════════════════════════════════════════════════════════════════════

DelayedFormSignalGenerator::DelayedFormSignalGenerator(
    drv_gpu_lib::IBackend* backend)
    : backend_(backend)
    , signal_gen_(backend)
    , lch_farrow_(backend) {

  if (!backend_ || !backend_->IsInitialized()) {
    throw std::runtime_error(
        "DelayedFormSignalGenerator: backend is null or not initialized");
  }
}

// ════════════════════════════════════════════════════════════════════════════
// SetParams / SetDelays / LoadMatrix
// ════════════════════════════════════════════════════════════════════════════

void DelayedFormSignalGenerator::SetParams(const FormParams& params) {
  params_ = params;
  lch_farrow_.SetSampleRate(static_cast<float>(params_.fs));
  lch_farrow_.SetNoise(
      static_cast<float>(params_.noise_amplitude),
      static_cast<float>(params_.norm),
      params_.noise_seed);
}

void DelayedFormSignalGenerator::SetParamsFromString(
    const std::string& params_str) {
  SetParams(FormParams::ParseFromString(params_str));
}

void DelayedFormSignalGenerator::SetDelays(const std::vector<float>& delay_us) {
  lch_farrow_.SetDelays(delay_us);
}

void DelayedFormSignalGenerator::LoadMatrix(const std::string& json_path) {
  lch_farrow_.LoadMatrix(json_path);
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

  // Если задержки не заданы — заполняем нулями
  if (lch_farrow_.GetDelays().empty()) {
    lch_farrow_.SetDelays(std::vector<float>(params_.antennas, 0.0f));
  }

  if (lch_farrow_.GetDelays().size() != params_.antennas) {
    throw std::invalid_argument(
        "DelayedFormSignalGenerator: delay_us.size()="
        + std::to_string(lch_farrow_.GetDelays().size())
        + " != antennas=" + std::to_string(params_.antennas));
  }

  // ── Шаг 1: Генерация чистого сигнала (noise=0, tau=0) ──
  FormParams clean_params = params_;
  clean_params.noise_amplitude = 0.0;
  clean_params.tau_base = 0.0;
  clean_params.tau_step = 0.0;
  clean_params.tau_min = 0.0;
  clean_params.tau_max = 0.0;
  signal_gen_.SetParams(clean_params);

  auto clean_signal = signal_gen_.GenerateInputData(prof_events);
  cl_mem input_buf = clean_signal.data;

  // ── Шаг 2: Применение задержки + шум через LchFarrow ──
  auto result = lch_farrow_.Process(
      input_buf, params_.antennas, params_.points, prof_events);

  clReleaseMemObject(input_buf);

  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// CPU генерация
// ════════════════════════════════════════════════════════════════════════════

std::vector<std::vector<std::complex<float>>>
DelayedFormSignalGenerator::GenerateToCpu() {
  auto input = GenerateInputData();
  cl_mem gpu_buf = input.data;

  cl_command_queue queue =
      static_cast<cl_command_queue>(backend_->GetNativeQueue());

  size_t total = GetTotalSamples();
  std::vector<std::complex<float>> flat(total);

  cl_int err = clEnqueueReadBuffer(
      queue, gpu_buf, CL_TRUE, 0,
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

} // namespace signal_gen
