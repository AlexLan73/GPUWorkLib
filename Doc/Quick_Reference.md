# GPUWorkLib — Quick Reference

> Краткий справочник публичных API всех модулей.
> Порядок: по приоритету использования (от обработки до инфраструктуры).
>
> **Date**: 2026-03-05 | **Full Reference**: [Full_Reference.md](Full_Reference.md) | **Index**: [INDEX.md](INDEX.md)

---

## Содержание

| # | Модуль | Описание | Полная документация |
|---|--------|----------|---------------------|
| 1 | [FFT Processor](#1-fft-processor) | БПФ (Complex / MagPhase) | [Doc/Modules/fft_processor/](Modules/fft_processor/) |
| 2 | [Statistics](#2-statistics-rocm) | Welford mean/var, медиана | [Doc/Modules/statistics/](Modules/statistics/) |
| 3 | [Vector Algebra](#3-vector-algebra-rocm) | Cholesky инверсия матриц | [Doc/Modules/vector_algebra/](Modules/vector_algebra/) |
| 4 | [FFT Maxima](#4-fft-maxima) | Поиск спектральных пиков | [Doc/Modules/fft_maxima/](Modules/fft_maxima/) |
| 5 | [Filters](#5-filters) | FIR / IIR фильтрация (OpenCL + ROCm, протестированы) | [Doc/Modules/filters/](Modules/filters/) |
| 6 | [Signal Generators](#6-signal-generators) | CW, LFM, Noise, Form | [Doc/Modules/signal_generators/](Modules/signal_generators/) |
| 7 | [LCH Farrow](#7-lch-farrow) | Дробная задержка | [Doc/Modules/lch_farrow/](Modules/lch_farrow/) |
| 8 | [Heterodyne](#8-heterodyne) | LFM Dechirp pipeline | [Doc/Modules/heterodyne/](Modules/heterodyne/) |
| 9 | [FM Correlator](#9-fm-correlator) | FM-корреляция с M-sequence (ROCm, протестирован) | [Doc/Modules/fm_correlator/](Modules/fm_correlator/) |
| 10 | [DrvGPU](#10-drvgpu) | Ядро: backend, память, сервисы | [Doc/DrvGPU/](DrvGPU/) |

---

## 1. FFT Processor

**Заголовок**: `modules/fft_processor/include/fft_processor.hpp`
**Backend**: OpenCL (`clFFT`) / ROCm (`hipFFT`)

```cpp
// Создание
FFTProcessor proc(backend);          // backend — IBackend* из DrvGPU

// Params
FFTProcessorParams params;
params.nfft      = 1024;             // размер FFT (степень 2)
params.n_beams   = 8;                // количество антенн/каналов
params.sample_rate = 12e6f;         // частота дискретизации (Hz)
params.window    = WindowType::HANN; // оконная функция

// CPU → Complex output
auto results = proc.ProcessComplex(data, params);
// results[i].beam_id, results[i].spectrum — вектор complex<float>

// CPU → Magnitude+Phase
auto results = proc.ProcessMagPhase(data, params);
// results[i].beam_id, results[i].magnitude[], results[i].phase[]

// GPU → Complex (данные уже на GPU)
auto results = proc.ProcessComplex(gpu_buf, params, gpu_bytes);

// GPU → Magnitude+Phase
auto results = proc.ProcessMagPhase(gpu_buf, params, gpu_bytes);

// Профилирование
FFTProfilingData pd = proc.GetProfilingData();
uint32_t n = proc.GetNFFT();
```

---

## 2. Statistics *(OpenCL / ROCm)*

**Заголовок**: `modules/statistics/include/statistics_processor.hpp`
**Backend**: OpenCL (CPU reference) / ROCm GPU (`ENABLE_ROCM=1`)

```cpp
// Создание
StatisticsProcessor stat(backend);   // backend — OpenCL или ROCm IBackend*

// Params
StatisticsParams params;
params.n_beams  = 8;                 // количество антенн
params.n_points = 1024;             // точек на антенну

// Комплексное среднее
auto mean = stat.ComputeMean(data, params);
// mean[i].beam_id, mean[i].mean — complex<float>

// Медиана амплитуд
auto med = stat.ComputeMedian(data, params);
// med[i].beam_id, med[i].median_magnitude — float

// Полная статистика (one-pass Welford)
auto res = stat.ComputeStatistics(data, params);
// res[i].mean, res[i].variance, res[i].std_dev, res[i].mean_magnitude

// GPU данные (void* — HIP device ptr)
auto res = stat.ComputeStatistics(gpu_ptr, params);
```

---

## 3. Vector Algebra *(OpenCL / ROCm)*

**Заголовок**: `modules/vector_algebra/include/cholesky_inverter_rocm.hpp`
**Backend**: OpenCL (CPU reference) / ROCm GPU (`ENABLE_ROCM=1`)

```cpp
// Создание
CholeskyInverterROCm inv(backend);
CholeskyInverterROCm inv(backend, SymmetrizeMode::GpuKernel); // по умолчанию

// Режимы симметризации
inv.SetSymmetrizeMode(SymmetrizeMode::GpuKernel);  // GPU kernel (быстро)
inv.SetSymmetrizeMode(SymmetrizeMode::Roundtrip);  // CPU roundtrip (совместимость)

// Инверсия одной матрицы n×n
InputData<vector<complex<float>>> input{data};     // row-major, n*n элементов
CholeskyResult res = inv.Invert(input, n);

// Инверсия batch из k матриц n×n
CholeskyResult res = inv.InvertBatch(input, n);    // data.size() == k*n*n

// Извлечение результата
auto vec = res.AsVector();                          // vector<complex<float>>, n*n
auto mat = res.matrix();                            // vector<vector<complex<float>>>
void* hip_ptr = res.AsHipPtr();                    // raw HIP device ptr

// Отключить проверку info для benchmark (без sync)
inv.SetCheckInfo(false);
```

---

## 4. FFT Maxima

**Заголовок**: `modules/fft_maxima/include/interface/i_spectrum_processor.hpp`
**Backend**: OpenCL / ROCm (через фабрику)

```cpp
// Создание через фабрику
auto proc = SpectrumProcessorFactory::Create(BackendType::OPENCL, backend);

// Параметры
SpectrumParams params;
params.nfft        = 1024;
params.n_antennas  = 8;
params.sample_rate = 12e6f;
proc->Initialize(params);

// Один пик на антенну (CPU вход)
auto results = proc->ProcessFromCPU(data);
// results[i].antenna_id, results[i].interpolated.freq_offset, .magnitude

// Один пик (GPU вход)
auto results = proc->ProcessFromGPU(gpu_ptr, n_antennas, nfft, gpu_bytes);

// Все максимумы (FFT + поиск за один вызов)
AllMaximaResult all = proc->FindAllMaximaFromGPUPipeline(
    gpu_ptr, n_antennas, nfft, gpu_bytes,
    OutputDestination::CPU, search_start, search_end);
// all.beams[i].antenna_id, all.beams[i].maxima[j].magnitude, .freq_offset

// Только поиск (готовые FFT данные на GPU)
AllMaximaResult all = proc->FindAllMaxima(
    fft_gpu_ptr, beam_count, nFFT, sample_rate);
```

---

## 5. Filters

**Заголовок**: `modules/filters/include/filters/fir_filter.hpp`
**Заголовок**: `modules/filters/include/filters/iir_filter.hpp`
**Backend**: OpenCL / ROCm

```cpp
// --- FIR фильтр ---
FirFilter fir(backend);
fir.SetCoefficients(coeffs);         // vector<float> коэффициенты
fir.LoadConfig("config.json");       // или из JSON

// GPU обработка
auto out_gpu = fir.Process(in_gpu_buf, channels, points);

// CPU reference
auto out_cpu = fir.ProcessCpu(data, channels, points);

uint32_t taps = fir.GetNumTaps();    // количество коэффициентов

// --- IIR фильтр (biquad sections) ---
IirFilter iir(backend);

// Секция Direct Form II: b0 x[n] + b1 x[n-1] + b2 x[n-2] - a1 y[n-1] - a2 y[n-2]
iir.SetBiquadSections({{b0,b1,b2,a1,a2}, ...});
iir.LoadConfig("config.json");

auto out_gpu = iir.Process(in_gpu_buf, channels, points);
auto out_cpu = iir.ProcessCpu(data, channels, points);

uint32_t secs = iir.GetNumSections();
```

---

## 6. Signal Generators

**Заголовок**: `modules/signal_generators/include/i_signal_generator.hpp`
**Backend**: OpenCL / ROCm

```cpp
// --- CW (непрерывная волна) ---
CwParams cw;
cw.f0        = 100.0;               // частота (Hz)
cw.amplitude = 1.0;
cw.phase     = 0.0;                  // начальная фаза (rad)
auto gen = SignalGeneratorFactory::CreateCw(backend, cw);

// --- LFM (линейная частотная модуляция) ---
LfmParams lfm;
lfm.f_start  = 100.0;               // начальная частота (Hz)
lfm.f_end    = 500.0;               // конечная частота (Hz)
lfm.amplitude = 1.0;
auto gen = SignalGeneratorFactory::CreateLfm(backend, lfm);

// --- Шум ---
NoiseParams noise;
noise.type   = NoiseType::GAUSSIAN;
noise.power  = 1.0;
noise.seed   = 42;
auto gen = SignalGeneratorFactory::CreateNoise(backend, noise);

// --- Генерация ---
SystemSampling sys;
sys.sample_rate = 12e6f;
sys.n_points    = 4096;

cl_mem buf = gen->GenerateToGpu(sys, n_beams);   // → GPU буфер
gen->GenerateToCpu(sys, out_ptr, out_size);       // → CPU буфер

// Универсальный Create по запросу
SignalRequest req{ SignalKind::CW, sys, cw_params };
auto gen = SignalGeneratorFactory::Create(backend, req);
```

---

## 7. LCH Farrow

**Заголовок**: `modules/lch_farrow/include/lch_farrow.hpp`
**Backend**: OpenCL / ROCm

```cpp
// Создание
LchFarrow farrow(backend);

// Конфигурация
farrow.SetSampleRate(12e6f);                        // частота дискретизации (Hz)
farrow.SetDelays({0.0f, 1.5f, 3.0f, 4.5f, 6.0f}); // задержки в мкс (per-antenna)
farrow.SetNoise(0.01f);                              // шум (опционально)
farrow.LoadMatrix("matrix.json");                    // 48×5 матрица Lagrange

// GPU обработка
auto out = farrow.Process(in_gpu_buf, n_antennas, n_points);

// CPU reference
auto out_cpu = farrow.ProcessCpu(data, n_antennas, n_points);

// Параметры
const auto& delays = farrow.GetDelays();
float fs = farrow.GetSampleRate();
```

---

## 8. Heterodyne

**Заголовок**: `modules/heterodyne/include/heterodyne_dechirp.hpp`
**Backend**: OpenCL / ROCm

```cpp
// Создание
HeterodyneDechirp hetero(backend);

// Параметры LFM сигнала
HeterodyneParams p;
p.f_start      = 0.0f;             // начальная частота LFM (Hz)
p.f_end        = 1e6f;             // конечная частота (Hz)
p.sample_rate  = 12e6f;           // частота дискретизации
p.num_samples  = 4000;            // точек на антенну
p.num_antennas = 5;

hetero.SetParams(p);

// Обработка CPU данных
HeterodyneResult res = hetero.Process(rx_data);    // vector<complex<float>>

// Обработка GPU данных
HeterodyneResult res = hetero.ProcessExternal(gpu_ptr, p);

// Результаты
res.success;                                         // bool
res.antennas[i].antenna_idx;
res.antennas[i].f_beat_hz;                          // beat frequency (Hz)
res.antennas[i].range_m;                            // дальность (м)
res.antennas[i].peak_snr_db;                        // SNR (dB)

// Полезные вычисления
float bw = p.GetBandwidth();
float dur = p.GetDuration();
float chirp_rate = p.GetChirpRate();
float range = HeterodyneResult::CalcRange(f_beat, sample_rate, n_samples, bw);
```

---

## 9. FM Correlator

**Заголовок**: `modules/fm_correlator/include/fm_correlator.hpp`
**Backend**: ROCm only (`ENABLE_ROCM=1`)

```cpp
// Создание
FMCorrelator corr(backend);   // backend — ROCm IBackend*

// Параметры
FMCorrelatorParams p;
p.fft_size          = 32768;        // размер FFT
p.num_shifts        = 32;           // количество сдвигов K
p.num_signals       = 5;            // количество сигналов S
p.num_output_points = 2000;         // точек в выводе n_kg
p.lfsr_seed         = 0x12345678;   // seed LFSR для M-seq
corr.SetParams(p);

// Генерация M-sequence
auto mseq = corr.GenerateMSequence();        // seed из params
auto mseq = corr.GenerateMSequence(seed);    // custom seed

// Подготовить эталон (FFT на GPU)
corr.PrepareReference();                      // из M-seq (seed из params)
corr.PrepareReference(ref_vec);              // из внешнего вектора

// Корреляция
FMCorrelatorResult res = corr.Process(inp);  // inp — vector<float>, S сигналов

// Тестовый паттерн (без H2D)
FMCorrelatorResult res = corr.RunTestPattern(shift_step);

// Доступ к результатам [S × K × n_kg]
float v = res.at(signal, shift, point);      // accessor по индексам
// res.peaks — flat vector<float>, row-major
```

---

## 10. DrvGPU

**Заголовок**: `DrvGPU/include/drv_gpu.hpp`

### Одиночный GPU

```cpp
// Инициализация
DrvGPU gpu(BackendType::OPENCL, 0);  // индекс GPU
DrvGPU gpu(BackendType::ROCM, 0);

// Информация
std::string name = gpu.GetDeviceName();
GPUDeviceInfo info = gpu.GetDeviceInfo();
int idx = gpu.GetDeviceIndex();

// Доступ к backend (для модулей)
IBackend* backend = &gpu.GetBackend();

// Синхронизация
gpu.Synchronize();
gpu.Flush();
```

### Multi-GPU

```cpp
GPUManager mgr;
mgr.InitializeAll(BackendType::OPENCL);         // все доступные GPU
mgr.InitializeSpecific(BackendType::OPENCL, {0, 1});

size_t n = mgr.GetGPUCount();
DrvGPU& g0 = mgr.GetGPU(0);
DrvGPU& gNext = mgr.GetNextGPU();              // Round-Robin
```

### GPUProfiler

```cpp
GPUProfiler profiler;
profiler.SetGPUInfo(mgr.GetGPUReportInfo(0));  // ⚠️ ОБЯЗАТЕЛЬНО перед Start!
profiler.Start();

// ... работа ...

profiler.Stop("my_stage");
profiler.PrintReport();                         // вывод в консоль
profiler.ExportMarkdown("report.md");          // файл
profiler.ExportJSON("report.json");            // JSON
```

### ConsoleOutput

```cpp
auto& con = gpu.GetBackend().GetConsoleOutput(); // или из GPUManager
con.Print(gpu_id, "ModuleName", "сообщение");   // 3-arg (мультиGPU-безопасный)
```

---

## 📊 Профилирование (все модули)

Каждый модуль имеет набор **benchmark классов** в `tests/{module}_benchmark.hpp`:

```cpp
// Профилирование: только через GPUProfiler!
GPUProfiler profiler;
profiler.SetGPUInfo(mgr.GetGPUReportInfo(gpu_id));  // ⚠️ Обязательно!
profiler.Start();

// ... операция ...

profiler.Stop("my_kernel");

// Вывод (запрещено вручную GetStats() + con.Print)
profiler.PrintReport();              // консоль
profiler.ExportMarkdown("out.md");   // файл
profiler.ExportJSON("out.json");     // JSON
```

**Запрещено**: `GetStats()` → цикл → `con.Print()` или `std::cout`.
**Разрешено**: только `PrintReport()` / `ExportMarkdown()` / `ExportJSON()`.

---

*See also: [Full_Reference.md](Full_Reference.md) · [INDEX.md](INDEX.md) · [Architecture/](Architecture/)*
*Last updated: 2026-03-05 | Maintained by: Кодо*
