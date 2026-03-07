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
#include "services/kernel_cache_service.hpp"
#include "backends/rocm/rocm_backend.hpp"

#include <stdexcept>
#include <string>
#include <algorithm>
#include <cstring>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// PART 1: Constructor, Destructor
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Конструктор: проверяет backend, создаёт два HIP-потока.
 *
 * Два потока нужны для параллельной обработки:
 *   stream0: ref-путь (H2D ref → apply_shifts → C2C FFT) и финальная корреляция
 *   stream1: inp-путь (H2D inp → R2C FFT) — выполняется одновременно со stream0
 * Синхронизируются через hipStreamSynchronize перед multiply_conj_fused.
 */
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

/**
 * @brief Применить новые параметры FFT/сигналов.
 *
 * Сбрасывает ref_prepared_ — после смены параметров размеры буферов изменились,
 * старые спектры эталона недействительны → требуется повторный PrepareReference().
 * Порядок: compile (lazy) → FreeBuffers → DestroyPlans → AllocateBuffers → CreatePlans.
 */
void FMCorrelatorProcessorROCm::SetParams(const FMCorrelatorParams& params) {
  params_ = params;
  // Эталон становится невалидным — размер N мог измениться
  ref_prepared_ = false;

  if (!kernels_compiled_) {
    CompileKernels();
  }

  // Пересоздаём буферы и планы под новые N/K/S: старые размеры несовместимы
  FreeBuffers();
  DestroyPlans();

  AllocateBuffers();
  CreatePlans();
}

// ════════════════════════════════════════════════════════════════════════════
// PART 3: PrepareReference
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Подготовить эталонный сигнал: загрузить, создать K циклических сдвигов,
 *        вычислить C2C FFT для каждого сдвига.
 *
 * После этого d_ref_complex_ содержит [K × N] float2 — спектры K сдвигов.
 * Эти спектры используются многократно в Process() без повторной загрузки.
 *
 * Почему C2C, а не R2C для эталона: сдвиги формируют комплексный массив
 * float2 (мнимая часть = 0). R2C принимает только вещественный float* —
 * пришлось бы иметь отдельный буфер. Проще один C2C на готовый float2.
 *
 * Синхронизируем stream0_ в конце, чтобы ref_prepared_ = true означало
 * «данные действительно готовы на GPU».
 *
 * @param ref Эталонный сигнал размером fft_size (значения ±1.0f)
 */
void FMCorrelatorProcessorROCm::PrepareReference(
    const std::vector<float>& ref) {
  const size_t N = params_.fft_size;
  const int K = params_.num_shifts;

  if (ref.size() != N) {
    throw std::runtime_error("PrepareReference: ref size " +
        std::to_string(ref.size()) + " != fft_size " + std::to_string(N));
  }

  // H2D: загружаем вещественный ref в d_ref_float_ (временный, только для shifts)
  hipError_t err = hipMemcpyHtoDAsync(
      d_ref_float_, const_cast<float*>(ref.data()),
      N * sizeof(float), stream0_);
  if (err != hipSuccess)
    throw std::runtime_error("PrepareReference: H2D failed: " +
                             std::string(hipGetErrorString(err)));

  // Kernel: apply_cyclic_shifts — заполняет d_ref_complex_[K×N] float2
  // out[k][i] = {ref[(i+k)%N], 0.0f} — K параллельных циклических сдвигов
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

  // C2C FFT Forward, in-place: d_ref_complex_ → d_ref_complex_ (теперь спектры)
  // plan_ref_ привязан к stream0_ через hipfftSetStream в CreatePlans()
  hipfftResult fft_err = hipfftExecC2C(
      plan_ref_,
      static_cast<hipfftComplex*>(d_ref_complex_),
      static_cast<hipfftComplex*>(d_ref_complex_),
      HIPFFT_FORWARD);
  if (fft_err != HIPFFT_SUCCESS)
    throw std::runtime_error("PrepareReference: hipfftExecC2C failed: " +
                             std::to_string(fft_err));

  // Ждём завершения stream0_ — ref готов, Process() может использовать спектры
  (void)hipStreamSynchronize(stream0_);
  ref_prepared_ = true;
}

// ════════════════════════════════════════════════════════════════════════════
// PART 4: Process
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Полный пайплайн корреляции для S входных сигналов.
 *
 * Два параллельных потока:
 *   stream0: был занят PrepareReference() (уже завершён и синхронизирован)
 *   stream1: H2D(inp) → R2C FFT — запускаем здесь
 *
 * Потоки синхронизируются перед RunCorrelationPipeline(): нужно гарантировать,
 * что и ref-спектры (stream0), и inp-спектры (stream1) готовы к multiply.
 *
 * @param inp Flat [S × N] float, row-major (signal 0 first)
 * @return FMCorrelatorResult с peaks [S × K × n_kg]
 */
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

  // H2D на stream1 — параллельно с любыми вычислениями на stream0
  hipError_t err = hipMemcpyHtoDAsync(
      d_inp_float_, const_cast<float*>(inp.data()),
      S * N * sizeof(float), stream1_);
  if (err != hipSuccess)
    throw std::runtime_error("Process: H2D inp failed: " +
                             std::string(hipGetErrorString(err)));

  // R2C FFT на stream1: результат [S × (N/2+1)] float2 в d_inp_fft_
  // plan_inp_ привязан к stream1_ через hipfftSetStream в CreatePlans()
  hipfftResult fft_err = hipfftExecR2C(
      plan_inp_, d_inp_float_,
      static_cast<hipfftComplex*>(d_inp_fft_));
  if (fft_err != HIPFFT_SUCCESS)
    throw std::runtime_error("Process: hipfftExecR2C failed: " +
                             std::to_string(fft_err));

  // Синхронизируем ОБА потока: multiply_conj_fused читает и d_ref_complex_, и d_inp_fft_
  (void)hipStreamSynchronize(stream0_);
  (void)hipStreamSynchronize(stream1_);

  return RunCorrelationPipeline();
}

// ════════════════════════════════════════════════════════════════════════════
// ProcessFromPtr — H2D из raw pointer без CPU-копии (для ProcessWithBatching)
// ════════════════════════════════════════════════════════════════════════════

FMCorrelatorResult FMCorrelatorProcessorROCm::ProcessFromPtr(
    const float* data, int num_signals) {
  if (!ref_prepared_) {
    throw std::runtime_error("ProcessFromPtr: reference not prepared");
  }

  const size_t N = params_.fft_size;

  // H2D напрямую из указателя — без промежуточной std::vector копии
  hipError_t err = hipMemcpyHtoDAsync(
      d_inp_float_, const_cast<float*>(data),
      static_cast<size_t>(num_signals) * N * sizeof(float), stream1_);
  if (err != hipSuccess)
    throw std::runtime_error("ProcessFromPtr: H2D inp failed: " +
                             std::string(hipGetErrorString(err)));

  hipfftResult fft_err = hipfftExecR2C(
      plan_inp_, d_inp_float_,
      static_cast<hipfftComplex*>(d_inp_fft_));
  if (fft_err != HIPFFT_SUCCESS)
    throw std::runtime_error("ProcessFromPtr: hipfftExecR2C failed: " +
                             std::to_string(fft_err));

  (void)hipStreamSynchronize(stream0_);
  (void)hipStreamSynchronize(stream1_);

  return RunCorrelationPipeline();
}

// ════════════════════════════════════════════════════════════════════════════
// RunCorrelationPipeline — shared by Process and RunTestPattern
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Финальная часть пайплайна: multiply → IFFT → extract → D2H.
 *
 * Вызывается из Process() и RunTestPattern() после того как оба потока
 * завершили и d_ref_complex_, и d_inp_fft_ готовы.
 *
 * Шаги (все на stream0):
 * 1. multiply_conj_fused: corr_fft[s,k,i] = conj(ref_fft[k,i]) * inp_fft[s,i]
 * 2. C2R IFFT (plan_corr): batch S*K, выход d_corr_time_ [S×K×N] float
 * 3. extract_magnitudes_real: |corr_time[j]| / N → d_peaks_ [S×K×n_kg]
 * 4. D2H: d_peaks_ → host, sync stream0
 *
 * @return FMCorrelatorResult с заполненным peaks вектором
 */
FMCorrelatorResult FMCorrelatorProcessorROCm::RunCorrelationPipeline() {
  int N = static_cast<int>(params_.fft_size);
  int K = params_.num_shifts;
  int S = params_.num_signals;
  int n_kg = params_.num_output_points;
  // R2C даёт N/2+1 точек (hermitian symmetry) — ровно столько нужно для C2R
  int half_N = N / 2 + 1;

  hipError_t err;

  // Step 1: multiply_conj_fused на stream0
  // ref_stride=N (не half_N!): d_ref_complex_ хранит полные C2C спектры [K×N] float2,
  // а не R2C-усечённые. Stride между сдвигами = N комплексных элементов.
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

  // Step 2: C2R IFFT на stream0, batch S*K
  // Вход [S×K×(N/2+1)] float2 → выход [S×K×N] float
  // plan_corr_ привязан к stream0_ через hipfftSetStream в CreatePlans()
  hipfftResult fft_err = hipfftExecC2R(
      plan_corr_,
      static_cast<hipfftComplex*>(d_corr_fft_),
      d_corr_time_);
  if (fft_err != HIPFFT_SUCCESS)
    throw std::runtime_error("RunCorrelationPipeline: hipfftExecC2R failed: " +
                             std::to_string(fft_err));

  // Step 3: extract_magnitudes_real на stream0
  // |corr_time[j]| * inv_N → peaks (только первые n_kg точек — дальнейшие бесполезны)
  // inv_N передаём параметром: FMUL вместо FDIV в каждом потоке
  float inv_N = 1.0f / static_cast<float>(N);
  void* args_ext[] = { &d_corr_time_, &d_peaks_,
                        &N, &n_kg, &K, &S, &inv_N };
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

  // Step 4: D2H peaks → host, затем sync для гарантии завершения копирования
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

/**
 * @brief Тестовый паттерн: генерация входных сигналов прямо на GPU.
 *
 * Входной сигнал s = circshift(ref, s*shift_step): сигнал s — это ref,
 * сдвинутый на s*shift_step позиций. Ожидаемый пик сигнала s, сдвига k
 * находится в позиции (s*shift_step - k + N) % N.
 *
 * Зачем: тест без H2D для входных данных — полностью на GPU, быстро.
 * d_ref_float_ должен быть актуальным (заполняется в PrepareReference).
 *
 * @param shift_step Шаг сдвига между сигналами (в сэмплах)
 */
FMCorrelatorResult FMCorrelatorProcessorROCm::RunTestPattern(int shift_step) {
  if (!ref_prepared_) {
    throw std::runtime_error("RunTestPattern: reference not prepared");
  }

  int N = static_cast<int>(params_.fft_size);
  int S = params_.num_signals;

  // generate_test_inputs на stream1: inp[s*N+i] = ref[(i+s*shift_step) % N]
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

/**
 * @brief Авто-батчинг для случая когда total_signals > допустимого S.
 *
 * BatchManager::CalculateOptimalBatchSize считает сколько сигналов влезает
 * в 70% свободной GPU-памяти (external_mem вычтен — он всегда занят ref_fft).
 * Для каждого батча пересоздаём буферы/планы если batch_S изменился.
 *
 * ВАЖНО: ref_prepared_ остаётся true и эталон на GPU не трогается — только
 * буферы под inp/corr пересоздаются под batch_S.
 *
 * @param inp         Flat [total_signals × N] float
 * @param total_signals Реальное число сигналов (может быть > params_.num_signals)
 * @return Объединённый результат [total_signals × K × n_kg]
 */
FMCorrelatorResult FMCorrelatorProcessorROCm::ProcessWithBatching(
    const std::vector<float>& inp, int total_signals) {
  const size_t N = params_.fft_size;
  const int K = params_.num_shifts;
  const int n_kg = params_.num_output_points;

  // Память на один сигнал: строки inp_float + inp_fft + (K строк corr_fft/time/peaks)
  size_t per_signal_memory =
      N * sizeof(float)                               // d_inp_float row
    + (N / 2 + 1) * sizeof(hipfftComplex)             // d_inp_fft row
    + K * (N / 2 + 1) * sizeof(hipfftComplex)         // d_corr_fft
    + K * N * sizeof(float)                            // d_corr_time
    + K * n_kg * sizeof(float);                        // d_peaks

  // ref-буферы постоянно заняты — вычитаем из бюджета при расчёте батча
  size_t external_mem = K * N * sizeof(hipfftComplex)  // ref_fft на GPU
                      + N * sizeof(float);             // ref_float на GPU

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

    // Пересоздаём буферы/планы только при смене размера батча (частый случай:
    // все батчи одного размера, кроме последнего — пересоздание 1-2 раза)
    if (batch_S != current_batch_S_) {
      DestroyPlans();
      FreeBuffers();
      params_.num_signals = batch_S;
      AllocateBuffers();
      CreatePlans();
      current_batch_S_ = batch_S;
    }

    // Передаём указатель напрямую — без CPU-аллокации и копии std::vector
    size_t offset = b.start * N;
    auto batch_result = ProcessFromPtr(inp.data() + offset, batch_S);
    combined.peaks.insert(combined.peaks.end(),
                          batch_result.peaks.begin(),
                          batch_result.peaks.end());
  }

  // Восстанавливаем оригинальные params: следующий вызов Process()/SetParams()
  // должен видеть total_signals, а не последний batch_S
  params_.num_signals = total_signals;
  current_batch_S_ = 0;

  return combined;
}

// ════════════════════════════════════════════════════════════════════════════
// PART 7: CompileKernels
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Компилирует 4 HIP кернела через hiprtc (lazy, один раз).
 *
 * Почему hiprtc, а не заранее скомпилированные .hsaco файлы:
 * - нет зависимости от конкретной архитектуры GPU при сборке
 * - компиляция происходит на целевом устройстве → оптимальный код
 * - все 4 кернела в одном строковом литерале (GetFMCorrelatorKernelSource)
 *
 * Процесс:
 * 1. hiprtcCreateProgram — загрузить исходник
 * 2. hiprtcCompileProgram с -O3 — компилировать
 * 3. hiprtcGetCode — извлечь бинарный код (.hsaco)
 * 4. hipModuleLoadData — загрузить модуль в память GPU
 * 5. hipModuleGetFunction × 4 — получить хэндлы кернелов по именам
 *
 * При ошибке компиляции выводим полный лог (ошибки, предупреждения hiprtc).
 *
 * @note kernels_compiled_ = true только при успехе всех 5 шагов.
 */
void FMCorrelatorProcessorROCm::CompileKernels() {
  if (kernels_compiled_) return;

  auto& con = ConsoleOutput::GetInstance();

  // Инициализация кеша (lazy, один раз)
  if (!kernel_cache_) {
    kernel_cache_ = std::make_unique<drv_gpu_lib::KernelCacheService>(
        "modules/fm_correlator/kernels", drv_gpu_lib::BackendType::ROCm);
  }

  static constexpr const char* kKernelName = "fm_correlator_kernels";

  // Лямбда загрузки модуля + функций из бинаря
  auto loadModuleAndFunctions = [&](const void* data, size_t size) {
    hipError_t hipErr = hipModuleLoadData(&module_, data);
    if (hipErr != hipSuccess)
      throw std::runtime_error(
          "FMCorrelator CompileKernels: hipModuleLoadData failed: " +
          std::string(hipGetErrorString(hipErr)));

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
  };

  // Шаг 1: попробовать загрузить из дискового кеша (~1-5 мс вместо ~100-200 мс)
  if (kernel_cache_) {
    try {
      auto entry = kernel_cache_->Load(kKernelName);
      if (entry.has_binary()) {
        loadModuleAndFunctions(entry.binary.data(), entry.binary.size());
        kernels_compiled_ = true;
        con.Print(0, "FM_Corr[ROCm]",
            "HIP kernels loaded from cache (4 kernels)");
        return;
      }
    } catch (...) {
      // Cache miss или повреждение — компилируем из исходника
    }
  }

  // Шаг 2: компиляция из исходника через hiprtc
  const char* source = kernels::GetFMCorrelatorKernelSource();

  hiprtcProgram prog;
  hiprtcResult rtcResult = hiprtcCreateProgram(
      &prog, source, "fm_correlator_kernels.hip", 0, nullptr, nullptr);
  if (rtcResult != HIPRTC_SUCCESS) {
    throw std::runtime_error(
        "FMCorrelator CompileKernels: hiprtcCreateProgram failed: " +
        std::string(hiprtcGetErrorString(rtcResult)));
  }

  // Получить целевую архитектуру GPU для --offload-arch (ISA-оптимизации)
  std::string arch_name;
  try {
    auto* rocm_backend = static_cast<drv_gpu_lib::ROCmBackend*>(backend_);
    arch_name = rocm_backend->GetCore().GetArchName();
  } catch (...) {
    arch_name = "";
  }

  // WARP_SIZE: RDNA (gfx10xx/11xx/12xx) = 32, CDNA/Vega (gfx9xx) = 64
  int warp_size = 32;
  if (arch_name.find("gfx9") == 0) {
    warp_size = 64;
  }

  std::string warp_define = "-DWARP_SIZE=" + std::to_string(warp_size);
  std::string arch_flag = arch_name.empty() ? "" : ("--offload-arch=" + arch_name);

  std::vector<const char*> opts = { "-O3", warp_define.c_str() };
  if (!arch_flag.empty()) {
    opts.push_back(arch_flag.c_str());
  }

  rtcResult = hiprtcCompileProgram(prog,
      static_cast<int>(opts.size()), opts.data());
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

  loadModuleAndFunctions(code.data(), code.size());

  // Шаг 3: сохранить бинарь в кеш для следующих запусков
  if (kernel_cache_) {
    try {
      std::vector<uint8_t> binary(code.begin(), code.end());
      kernel_cache_->Save(kKernelName, std::string(source),
                          binary, arch_name, "fm_correlator");
    } catch (...) {
      // Не критично: кеш не удалось сохранить — следующий запуск перекомпилирует
    }
  }

  kernels_compiled_ = true;
  con.Print(0, "FM_Corr[ROCm]",
      "HIP kernels compiled (4 kernels, arch=" + arch_name + ")");
}

// ════════════════════════════════════════════════════════════════════════════
// PART 8: Buffer/Plan management
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Выделить GPU-буферы под текущие параметры (N, K, S, n_kg).
 *
 * half_N = N/2+1: R2C FFT возвращает hermitian-усечённый спектр (N/2+1 точек),
 * поэтому d_inp_fft_ и d_corr_fft_ используют именно этот размер.
 * d_ref_complex_ хранит полные C2C спектры [K×N] — stride=N, не half_N.
 *
 * Порядок выделения не критичен, но при ошибке throw прерывает — частично
 * выделенные буферы будут освобождены в деструкторе через ReleaseAll().
 */
void FMCorrelatorProcessorROCm::AllocateBuffers() {
  const size_t N = params_.fft_size;
  const int K = params_.num_shifts;
  const int S = params_.num_signals;
  const int n_kg = params_.num_output_points;
  // R2C output size: N/2+1 (hermitian symmetry saves ~50% памяти vs C2C)
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
  alloc(&d_ref_complex_,                            K * N * sizeof(hipfftComplex),            "ref_complex");   // C2C: полный спектр K×N
  alloc(reinterpret_cast<void**>(&d_inp_float_),    S * N * sizeof(float),                   "inp_float");
  alloc(&d_inp_fft_,                                S * half_N * sizeof(hipfftComplex),       "inp_fft");        // R2C: usечённый S×(N/2+1)
  alloc(&d_corr_fft_,                               static_cast<size_t>(S) * K * half_N * sizeof(hipfftComplex), "corr_fft");
  alloc(reinterpret_cast<void**>(&d_corr_time_),    static_cast<size_t>(S) * K * N * sizeof(float),              "corr_time");
  alloc(reinterpret_cast<void**>(&d_peaks_),        static_cast<size_t>(S) * K * n_kg * sizeof(float),           "peaks");

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

/**
 * @brief Создать три hipFFT-плана и привязать их к потокам.
 *
 * Планы создаются при SetParams() и хранятся постоянно — не пересоздавать
 * в каждом Process()! Создание плана дорого (~миллисекунды).
 *
 * hipfftSetStream ОБЯЗАТЕЛЕН: без него план выполняется на default stream
 * (нулевой поток), что блокирует всё и нарушает параллелизм stream0/stream1.
 *
 * Параметры hipfftPlanMany (inembed=nullptr → contiguous layout):
 *   plan_ref:  C2C Forward,  batch=K,   stream0
 *   plan_inp:  R2C Forward,  batch=S,   stream1
 *   plan_corr: C2R Inverse,  batch=S*K, stream0
 */
void FMCorrelatorProcessorROCm::CreatePlans() {
  const int N = static_cast<int>(params_.fft_size);
  const int K = params_.num_shifts;
  const int S = params_.num_signals;

  hipfftResult res;
  int n_arr[1] = { N };

  // plan_ref: K параллельных C2C FFT для сдвигов эталона (stream0)
  res = hipfftPlanMany(&plan_ref_, 1, n_arr,
      nullptr, 1, N,
      nullptr, 1, N,
      HIPFFT_C2C, K);
  if (res != HIPFFT_SUCCESS)
    throw std::runtime_error("CreatePlans: plan_ref failed: " + std::to_string(res));
  (void)hipfftSetStream(plan_ref_, stream0_);

  // plan_inp: S параллельных R2C FFT для входных сигналов (stream1)
  // Выходной stride = N/2+1 (hermitian), не N
  res = hipfftPlanMany(&plan_inp_, 1, n_arr,
      nullptr, 1, N,
      nullptr, 1, N / 2 + 1,
      HIPFFT_R2C, S);
  if (res != HIPFFT_SUCCESS)
    throw std::runtime_error("CreatePlans: plan_inp failed: " + std::to_string(res));
  (void)hipfftSetStream(plan_inp_, stream1_);

  // plan_corr: S*K параллельных C2R IFFT для всех пар (сигнал, сдвиг) (stream0)
  // Входной stride = N/2+1, выходной = N
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

/**
 * @brief Полное освобождение всех GPU-ресурсов.
 *
 * Порядок важен: сначала планы и буферы (зависят от модуля),
 * потом модуль (зависит от потоков), потом потоки.
 * Нулим fn_* указатели после выгрузки модуля — они больше невалидны.
 */
void FMCorrelatorProcessorROCm::ReleaseAll() {
  DestroyPlans();
  FreeBuffers();

  if (module_) {
    (void)hipModuleUnload(module_);
    module_ = nullptr;
    // После выгрузки модуля хэндлы кернелов невалидны — обнуляем явно
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
