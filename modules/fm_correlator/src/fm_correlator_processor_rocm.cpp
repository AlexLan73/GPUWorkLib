/**
 * @file fm_correlator_processor_rocm.cpp
 * @brief ROCm backend implementation for FM Correlator
 *
 * CONTENTS:
 *   PART 1: Constructor, Destructor
 *   PART 2: SetParams — compile kernels, allocate buffers, create FFT plans
 *   PART 3: PrepareReference — H2D + shifts + C2C FFT
 *   PART 4: Process — H2D(inp) + R2C + multiply + C2R + extract + D2H
 *   PART 5: RunTestPattern — generate_test_inputs on GPU + correlation
 *   PART 6: ProcessWithBatching — BatchManager for large S
 *   PART 7: CompileKernels — hiprtc compilation
 *   PART 8: Buffer/Plan management + cleanup
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-03
 */

#if ENABLE_ROCM

#include "fm_correlator_processor_rocm.hpp"
#include "kernels/fm_kernels_rocm.hpp"
#include "services/console_output.hpp"
#include "services/batch_manager.hpp"

#include <stdexcept>
#include <string>
#include <algorithm>
#include <cstring>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// PART 1: Constructor, Destructor
// ════════════════════════════════════════════════════════════════════════════

FMCorrelatorProcessorROCm::FMCorrelatorProcessorROCm(IBackend* backend)
    : backend_(backend) {
  if (!backend_ || !backend_->IsInitialized()) {
    throw std::runtime_error(
        "FMCorrelatorProcessorROCm: backend is null or not initialized");
  }

  hipError_t err;
  err = hipStreamCreate(&stream0_);
  if (err != hipSuccess)
    throw std::runtime_error("FMCorrelator: hipStreamCreate(0) failed: " +
                             std::string(hipGetErrorString(err)));
  err = hipStreamCreate(&stream1_);
  if (err != hipSuccess)
    throw std::runtime_error("FMCorrelator: hipStreamCreate(1) failed: " +
                             std::string(hipGetErrorString(err)));
}

FMCorrelatorProcessorROCm::~FMCorrelatorProcessorROCm() {
  ReleaseAll();
}

// ════════════════════════════════════════════════════════════════════════════
// PART 2: SetParams
// ════════════════════════════════════════════════════════════════════════════

void FMCorrelatorProcessorROCm::SetParams(const FMCorrelatorParams& params) {
  params_ = params;
  ref_prepared_ = false;

  if (!kernels_compiled_) {
    CompileKernels();
  }

  FreeBuffers();
  DestroyPlans();

  AllocateBuffers();
  CreatePlans();
}

// ════════════════════════════════════════════════════════════════════════════
// PART 3: PrepareReference
// ════════════════════════════════════════════════════════════════════════════

void FMCorrelatorProcessorROCm::PrepareReference(
    const std::vector<float>& ref) {
  const size_t N = params_.fft_size;
  const int K = params_.num_shifts;

  if (ref.size() != N) {
    throw std::runtime_error("PrepareReference: ref size " +
        std::to_string(ref.size()) + " != fft_size " + std::to_string(N));
  }

  // H2D: upload ref to d_ref_float_
  hipError_t err = hipMemcpyHtoDAsync(
      d_ref_float_, const_cast<float*>(ref.data()),
      N * sizeof(float), stream0_);
  if (err != hipSuccess)
    throw std::runtime_error("PrepareReference: H2D failed: " +
                             std::string(hipGetErrorString(err)));

  // Kernel: apply_cyclic_shifts
  int N_int = static_cast<int>(N);
  int K_int = K;
  void* args_shift[] = { &d_ref_float_, &d_ref_complex_, &N_int, &K_int };
  unsigned int grid_x = (static_cast<unsigned int>(N) + kBlockSize - 1) / kBlockSize;

  err = hipModuleLaunchKernel(
      fn_apply_shifts_,
      grid_x, static_cast<unsigned int>(K), 1,
      kBlockSize, 1, 1,
      0, stream0_,
      args_shift, nullptr);
  if (err != hipSuccess)
    throw std::runtime_error("PrepareReference: apply_cyclic_shifts failed: " +
                             std::string(hipGetErrorString(err)));

  // C2C FFT Forward on ref_complex -> ref_fft (in-place)
  hipfftResult fft_err = hipfftExecC2C(
      plan_ref_,
      static_cast<hipfftComplex*>(d_ref_complex_),
      static_cast<hipfftComplex*>(d_ref_complex_),
      HIPFFT_FORWARD);
  if (fft_err != HIPFFT_SUCCESS)
    throw std::runtime_error("PrepareReference: hipfftExecC2C failed: " +
                             std::to_string(fft_err));

  (void)hipStreamSynchronize(stream0_);
  ref_prepared_ = true;
}

// ════════════════════════════════════════════════════════════════════════════
// PART 4: Process
// ════════════════════════════════════════════════════════════════════════════

FMCorrelatorResult FMCorrelatorProcessorROCm::Process(
    const std::vector<float>& inp) {
  if (!ref_prepared_) {
    throw std::runtime_error("Process: reference not prepared");
  }

  const size_t N = params_.fft_size;
  const int S = params_.num_signals;

  if (inp.size() != static_cast<size_t>(S) * N) {
    throw std::runtime_error("Process: inp size " +
        std::to_string(inp.size()) + " != S*N " +
        std::to_string(static_cast<size_t>(S) * N));
  }

  // H2D: upload inp to d_inp_float_ on stream1
  hipError_t err = hipMemcpyHtoDAsync(
      d_inp_float_, const_cast<float*>(inp.data()),
      S * N * sizeof(float), stream1_);
  if (err != hipSuccess)
    throw std::runtime_error("Process: H2D inp failed: " +
                             std::string(hipGetErrorString(err)));

  // R2C FFT on inp on stream1
  hipfftResult fft_err = hipfftExecR2C(
      plan_inp_, d_inp_float_,
      static_cast<hipfftComplex*>(d_inp_fft_));
  if (fft_err != HIPFFT_SUCCESS)
    throw std::runtime_error("Process: hipfftExecR2C failed: " +
                             std::to_string(fft_err));

  // Sync both streams before correlation
  (void)hipStreamSynchronize(stream0_);
  (void)hipStreamSynchronize(stream1_);

  return RunCorrelationPipeline();
}

// ════════════════════════════════════════════════════════════════════════════
// RunCorrelationPipeline — shared by Process and RunTestPattern
// ════════════════════════════════════════════════════════════════════════════

FMCorrelatorResult FMCorrelatorProcessorROCm::RunCorrelationPipeline() {
  int N = static_cast<int>(params_.fft_size);
  int K = params_.num_shifts;
  int S = params_.num_signals;
  int n_kg = params_.num_output_points;
  int half_N = N / 2 + 1;

  hipError_t err;

  // multiply_conj_fused on stream0
  int ref_stride = N;
  void* args_mul[] = { &d_ref_complex_, &d_inp_fft_, &d_corr_fft_,
                        &half_N, &ref_stride, &K, &S };
  unsigned int grid_mul_x = (static_cast<unsigned int>(half_N) + kBlockSize - 1) / kBlockSize;

  err = hipModuleLaunchKernel(
      fn_multiply_conj_,
      grid_mul_x, static_cast<unsigned int>(K), static_cast<unsigned int>(S),
      kBlockSize, 1, 1,
      0, stream0_,
      args_mul, nullptr);
  if (err != hipSuccess)
    throw std::runtime_error("RunCorrelationPipeline: multiply_conj_fused failed: " +
                             std::string(hipGetErrorString(err)));

  // C2R IFFT on stream0
  hipfftResult fft_err = hipfftExecC2R(
      plan_corr_,
      static_cast<hipfftComplex*>(d_corr_fft_),
      d_corr_time_);
  if (fft_err != HIPFFT_SUCCESS)
    throw std::runtime_error("RunCorrelationPipeline: hipfftExecC2R failed: " +
                             std::to_string(fft_err));

  // extract_magnitudes_real on stream0
  void* args_ext[] = { &d_corr_time_, &d_peaks_,
                        &N, &n_kg, &K, &S };
  unsigned int grid_ext_x = (static_cast<unsigned int>(n_kg) + kBlockSize - 1) / kBlockSize;

  err = hipModuleLaunchKernel(
      fn_extract_mag_,
      grid_ext_x, static_cast<unsigned int>(K), static_cast<unsigned int>(S),
      kBlockSize, 1, 1,
      0, stream0_,
      args_ext, nullptr);
  if (err != hipSuccess)
    throw std::runtime_error("RunCorrelationPipeline: extract_magnitudes failed: " +
                             std::string(hipGetErrorString(err)));

  // D2H: peaks
  size_t peaks_size = static_cast<size_t>(S) * K * n_kg;
  FMCorrelatorResult result;
  result.peaks.resize(peaks_size);
  result.num_signals = S;
  result.num_shifts = K;
  result.num_output_points = n_kg;

  err = hipMemcpyDtoHAsync(
      result.peaks.data(), d_peaks_,
      peaks_size * sizeof(float), stream0_);
  if (err != hipSuccess)
    throw std::runtime_error("RunCorrelationPipeline: D2H peaks failed: " +
                             std::string(hipGetErrorString(err)));

  (void)hipStreamSynchronize(stream0_);
  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// PART 5: RunTestPattern
// ════════════════════════════════════════════════════════════════════════════

FMCorrelatorResult FMCorrelatorProcessorROCm::RunTestPattern(int shift_step) {
  if (!ref_prepared_) {
    throw std::runtime_error("RunTestPattern: reference not prepared");
  }

  int N = static_cast<int>(params_.fft_size);
  int S = params_.num_signals;

  // generate_test_inputs on stream1
  void* args_gen[] = { &d_ref_float_, &d_inp_float_,
                        &N, &S, &shift_step };
  unsigned int grid_gen_x = (static_cast<unsigned int>(N) + kBlockSize - 1) / kBlockSize;

  hipError_t err = hipModuleLaunchKernel(
      fn_gen_test_inputs_,
      grid_gen_x, static_cast<unsigned int>(S), 1,
      kBlockSize, 1, 1,
      0, stream1_,
      args_gen, nullptr);
  if (err != hipSuccess)
    throw std::runtime_error("RunTestPattern: generate_test_inputs failed: " +
                             std::string(hipGetErrorString(err)));

  // R2C FFT on stream1
  hipfftResult fft_err = hipfftExecR2C(
      plan_inp_, d_inp_float_,
      static_cast<hipfftComplex*>(d_inp_fft_));
  if (fft_err != HIPFFT_SUCCESS)
    throw std::runtime_error("RunTestPattern: hipfftExecR2C failed: " +
                             std::to_string(fft_err));

  // Sync both streams
  (void)hipStreamSynchronize(stream0_);
  (void)hipStreamSynchronize(stream1_);

  return RunCorrelationPipeline();
}

// ════════════════════════════════════════════════════════════════════════════
// PART 6: ProcessWithBatching
// ════════════════════════════════════════════════════════════════════════════

FMCorrelatorResult FMCorrelatorProcessorROCm::ProcessWithBatching(
    const std::vector<float>& inp, int total_signals) {
  const size_t N = params_.fft_size;
  const int K = params_.num_shifts;
  const int n_kg = params_.num_output_points;

  size_t per_signal_memory =
      N * sizeof(float)                               // d_inp_float row
    + (N / 2 + 1) * sizeof(hipfftComplex)             // d_inp_fft row
    + K * (N / 2 + 1) * sizeof(hipfftComplex)         // d_corr_fft
    + K * N * sizeof(float)                            // d_corr_time
    + K * n_kg * sizeof(float);                        // d_peaks

  size_t external_mem = K * N * sizeof(hipfftComplex)  // ref_fft on GPU
                      + N * sizeof(float);             // ref_float on GPU

  size_t batch_sz = BatchManager::CalculateOptimalBatchSize(
      backend_, total_signals, per_signal_memory, 0.7, external_mem);
  auto batches = BatchManager::CreateBatches(total_signals, batch_sz, 3, true);

  FMCorrelatorResult combined;
  combined.num_signals = total_signals;
  combined.num_shifts = K;
  combined.num_output_points = n_kg;
  combined.peaks.reserve(static_cast<size_t>(total_signals) * K * n_kg);

  for (auto& b : batches) {
    int batch_S = static_cast<int>(b.count);

    // Recreate plans if batch size changed
    if (batch_S != current_batch_S_) {
      FMCorrelatorParams batch_params = params_;
      batch_params.num_signals = batch_S;

      DestroyPlans();
      FreeBuffers();
      params_.num_signals = batch_S;
      AllocateBuffers();
      CreatePlans();
      current_batch_S_ = batch_S;
    }

    // Slice input
    size_t offset = b.start * N;
    std::vector<float> batch_inp(
        inp.begin() + static_cast<ptrdiff_t>(offset),
        inp.begin() + static_cast<ptrdiff_t>(offset + b.count * N));

    auto batch_result = Process(batch_inp);
    combined.peaks.insert(combined.peaks.end(),
                          batch_result.peaks.begin(),
                          batch_result.peaks.end());
  }

  // Restore original params
  params_.num_signals = total_signals;
  current_batch_S_ = 0;

  return combined;
}

// ════════════════════════════════════════════════════════════════════════════
// PART 7: CompileKernels
// ════════════════════════════════════════════════════════════════════════════

void FMCorrelatorProcessorROCm::CompileKernels() {
  if (kernels_compiled_) return;

  auto& con = ConsoleOutput::GetInstance();
  const char* source = kernels::GetFMCorrelatorKernelSource();

  hiprtcProgram prog;
  hiprtcResult rtcResult = hiprtcCreateProgram(
      &prog, source, "fm_correlator_kernels.hip", 0, nullptr, nullptr);
  if (rtcResult != HIPRTC_SUCCESS) {
    throw std::runtime_error(
        "FMCorrelator CompileKernels: hiprtcCreateProgram failed: " +
        std::string(hiprtcGetErrorString(rtcResult)));
  }

  const char* options[] = { "-O3" };
  rtcResult = hiprtcCompileProgram(prog, 1, options);
  if (rtcResult != HIPRTC_SUCCESS) {
    size_t logSize = 0;
    hiprtcGetProgramLogSize(prog, &logSize);
    std::string log(logSize, '\0');
    hiprtcGetProgramLog(prog, &log[0]);
    con.PrintError(0, "FM_Corr[ROCm]", "Kernel compile log:\n" + log);
    hiprtcDestroyProgram(&prog);
    throw std::runtime_error("FMCorrelator CompileKernels: compilation failed");
  }

  size_t codeSize = 0;
  hiprtcGetCodeSize(prog, &codeSize);
  std::vector<char> code(codeSize);
  hiprtcGetCode(prog, code.data());
  hiprtcDestroyProgram(&prog);

  hipError_t hipErr = hipModuleLoadData(&module_, code.data());
  if (hipErr != hipSuccess) {
    throw std::runtime_error(
        "FMCorrelator CompileKernels: hipModuleLoadData failed: " +
        std::string(hipGetErrorString(hipErr)));
  }

  auto getFunc = [&](hipFunction_t* fn, const char* name) {
    hipErr = hipModuleGetFunction(fn, module_, name);
    if (hipErr != hipSuccess) {
      (void)hipModuleUnload(module_);
      module_ = nullptr;
      throw std::runtime_error(
          std::string("FMCorrelator CompileKernels: hipModuleGetFunction(") +
          name + ") failed: " + hipGetErrorString(hipErr));
    }
  };

  getFunc(&fn_apply_shifts_,    "apply_cyclic_shifts");
  getFunc(&fn_multiply_conj_,   "multiply_conj_fused");
  getFunc(&fn_extract_mag_,     "extract_magnitudes_real");
  getFunc(&fn_gen_test_inputs_, "generate_test_inputs");

  kernels_compiled_ = true;
  con.Print(0, "FM_Corr[ROCm]",
      "HIP kernels compiled (4 kernels: shifts, multiply, extract, gen_test)");
}

// ════════════════════════════════════════════════════════════════════════════
// PART 8: Buffer/Plan management
// ════════════════════════════════════════════════════════════════════════════

void FMCorrelatorProcessorROCm::AllocateBuffers() {
  const size_t N = params_.fft_size;
  const int K = params_.num_shifts;
  const int S = params_.num_signals;
  const int n_kg = params_.num_output_points;
  const size_t half_N = N / 2 + 1;

  hipError_t err;

  auto alloc = [&](void** ptr, size_t bytes, const char* name) {
    err = hipMalloc(ptr, bytes);
    if (err != hipSuccess)
      throw std::runtime_error(
          std::string("FMCorrelator AllocateBuffers(") + name + "): " +
          hipGetErrorString(err));
  };

  alloc(reinterpret_cast<void**>(&d_ref_float_),   N * sizeof(float),                       "ref_float");
  alloc(&d_ref_complex_,                            K * N * sizeof(hipfftComplex),            "ref_complex");
  alloc(reinterpret_cast<void**>(&d_inp_float_),    S * N * sizeof(float),                   "inp_float");
  alloc(&d_inp_fft_,                                S * half_N * sizeof(hipfftComplex),       "inp_fft");
  alloc(&d_corr_fft_,                               static_cast<size_t>(S) * K * half_N * sizeof(hipfftComplex), "corr_fft");
  alloc(reinterpret_cast<void**>(&d_corr_time_),    static_cast<size_t>(S) * K * N * sizeof(float),              "corr_time");
  alloc(reinterpret_cast<void**>(&d_peaks_),         static_cast<size_t>(S) * K * n_kg * sizeof(float),          "peaks");

  buffers_allocated_ = true;
  current_batch_S_ = S;
}

void FMCorrelatorProcessorROCm::FreeBuffers() {
  if (!buffers_allocated_) return;

  auto safeFree = [](void*& ptr) {
    if (ptr) { (void)hipFree(ptr); ptr = nullptr; }
  };
  auto safeFreeF = [](float*& ptr) {
    if (ptr) { (void)hipFree(ptr); ptr = nullptr; }
  };

  safeFreeF(d_ref_float_);
  safeFree(d_ref_complex_);
  safeFreeF(d_inp_float_);
  safeFree(d_inp_fft_);
  safeFree(d_corr_fft_);
  safeFreeF(d_corr_time_);
  safeFreeF(d_peaks_);

  buffers_allocated_ = false;
}

void FMCorrelatorProcessorROCm::CreatePlans() {
  const int N = static_cast<int>(params_.fft_size);
  const int K = params_.num_shifts;
  const int S = params_.num_signals;

  hipfftResult res;
  int n_arr[1] = { N };

  // plan_ref: C2C Forward, batch=K
  res = hipfftPlanMany(&plan_ref_, 1, n_arr,
      nullptr, 1, N,
      nullptr, 1, N,
      HIPFFT_C2C, K);
  if (res != HIPFFT_SUCCESS)
    throw std::runtime_error("CreatePlans: plan_ref failed: " + std::to_string(res));
  (void)hipfftSetStream(plan_ref_, stream0_);

  // plan_inp: R2C Forward, batch=S
  res = hipfftPlanMany(&plan_inp_, 1, n_arr,
      nullptr, 1, N,
      nullptr, 1, N / 2 + 1,
      HIPFFT_R2C, S);
  if (res != HIPFFT_SUCCESS)
    throw std::runtime_error("CreatePlans: plan_inp failed: " + std::to_string(res));
  (void)hipfftSetStream(plan_inp_, stream1_);

  // plan_corr: C2R Inverse, batch=S*K
  res = hipfftPlanMany(&plan_corr_, 1, n_arr,
      nullptr, 1, N / 2 + 1,
      nullptr, 1, N,
      HIPFFT_C2R, S * K);
  if (res != HIPFFT_SUCCESS)
    throw std::runtime_error("CreatePlans: plan_corr failed: " + std::to_string(res));
  (void)hipfftSetStream(plan_corr_, stream0_);

  plans_created_ = true;
}

void FMCorrelatorProcessorROCm::DestroyPlans() {
  if (!plans_created_) return;
  if (plan_ref_)  { (void)hipfftDestroy(plan_ref_);  plan_ref_ = 0; }
  if (plan_inp_)  { (void)hipfftDestroy(plan_inp_);  plan_inp_ = 0; }
  if (plan_corr_) { (void)hipfftDestroy(plan_corr_); plan_corr_ = 0; }
  plans_created_ = false;
}

void FMCorrelatorProcessorROCm::ReleaseAll() {
  DestroyPlans();
  FreeBuffers();

  if (module_) {
    (void)hipModuleUnload(module_);
    module_ = nullptr;
    fn_apply_shifts_ = nullptr;
    fn_multiply_conj_ = nullptr;
    fn_extract_mag_ = nullptr;
    fn_gen_test_inputs_ = nullptr;
  }
  kernels_compiled_ = false;

  if (stream0_) { (void)hipStreamDestroy(stream0_); stream0_ = nullptr; }
  if (stream1_) { (void)hipStreamDestroy(stream1_); stream1_ = nullptr; }
}

}  // namespace drv_gpu_lib

#endif  // ENABLE_ROCM
