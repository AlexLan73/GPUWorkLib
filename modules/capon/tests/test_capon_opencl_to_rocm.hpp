#pragma once

/**
 * @file test_capon_opencl_to_rocm.hpp
 * @brief Данные от OpenCL — расчёт Кейпона на ROCm (Zero Copy Interop)
 *
 * ══════════════════════════════════════════════════════════════════════════
 * ДЛЯ ЧЕГО ЭТОТ ТЕСТ
 * ══════════════════════════════════════════════════════════════════════════
 *
 * В реальных DSP-системах данные часто уже находятся в OpenCL (например,
 * вышли из предыдущего этапа обработки). Нужно передать их в ROCm для
 * расчёта алгоритма Кейпона. Есть два способа:
 *
 *   МЕДЛЕННЫЙ — через CPU (не делай так!):
 *     cl_mem (VRAM) → RAM (CPU) → hipMalloc (VRAM)   ← 2 ненужных DMA!
 *
 *   БЫСТРЫЙ — Zero Copy:
 *     cl_mem (VRAM) ──────────────────────────────► hip_ptr (тот же VRAM)
 *     (передаём только GPU-адрес, данные не перемещаются)
 *
 * ══════════════════════════════════════════════════════════════════════════
 * КАК РАБОТАЕТ ZERO COPY
 * ══════════════════════════════════════════════════════════════════════════
 *
 * OpenCL и HIP/ROCm на одном AMD GPU разделяют одну физическую видеопамять.
 * ZeroCopyBridge «рассказывает» HIP, по какому адресу в VRAM лежат данные.
 *
 * Метод 1 — AMD GPU VA (предпочтительный, 0 overhead):
 *   Расширение CL_MEM_AMD_GPU_VA даёт нам GPU Virtual Address (VA) буфера.
 *   HIP принимает этот адрес напрямую — никаких системных вызовов.
 *
 *   cl_mem ──CL_MEM_AMD_GPU_VA──► void* gpu_va ──► hip_ptr  (~нс)
 *
 * Метод 2 — DMA-BUF (Linux kernel, ~мкс):
 *   OpenCL экспортирует cl_mem как Linux dma-buf file descriptor.
 *   HIP импортирует fd через hipImportExternalMemory.
 *
 *   cl_mem ──dma-buf fd──► hipExternalMemory_t ──► hip_ptr  (~мкс)
 *
 * ══════════════════════════════════════════════════════════════════════════
 * PIPELINE ТЕСТА (пошагово)
 * ══════════════════════════════════════════════════════════════════════════
 *
 *   [CPU]  std::vector<cx> signal, steering    — генерируем тестовые данные
 *     │
 *     │  (1 DMA transfer: CPU RAM → GPU VRAM)
 *     ▼
 *   [OpenCL]  cl_mem Y, cl_mem U              — данные в VRAM через OpenCL
 *     │
 *     │  cl.Synchronize() ← ОБЯЗАТЕЛЬНО! Иначе ROCm начнёт читать
 *     │                      до завершения записи OpenCL
 *     │
 *     │  ZeroCopyBridge::ImportFromOpenCl()   — только указатель, 0 копий
 *     ▼
 *   [ROCm]  void* hip_Y, void* hip_U          — те же данные, вид со стороны HIP
 *     │
 *     │  CaponProcessor::ComputeRelief(hip_Y, hip_U, params)
 *     │    ├── CovarianceMatrix:  R = (1/N)*Y*Y^H + μI   (rocBLAS CGEMM)
 *     │    ├── CaponInvert:       R⁻¹  (rocSOLVER POTRF+POTRI)
 *     │    ├── ComputeWeights:    W = R⁻¹*U              (rocBLAS CGEMM)
 *     │    └── CaponRelief:       z[m] = 1/Re(u^H*W[:,m])  (HIP kernel)
 *     ▼
 *   [CPU]  CaponReliefResult — M вещественных значений пространственного спектра
 *
 * ══════════════════════════════════════════════════════════════════════════
 * ТЕСТЫ
 * ══════════════════════════════════════════════════════════════════════════
 *
 *   01. detect_interop       — определить метод Zero Copy, вывести возможности GPU
 *   02. signal_from_opencl   — загрузить Y и U через OpenCL, Кейпон на ROCm
 *   03. results_match_ref    — Zero Copy путь = прямой CPU путь (одинаковый рельеф)
 *   04. beamform_from_opencl — адаптивное ДО [M×N] с входом через OpenCL
 *
 * ══════════════════════════════════════════════════════════════════════════
 * ЗАВИСИМОСТИ
 * ══════════════════════════════════════════════════════════════════════════
 *
 *   - OpenCLBackend  (DrvGPU/backends/opencl/)
 *   - ROCmBackend    (DrvGPU/backends/rocm/)
 *   - ZeroCopyBridge (DrvGPU/backends/rocm/zero_copy_bridge.hpp)
 *   - CaponProcessor (modules/capon/include/capon_processor.hpp)
 *
 * @note  ТОЛЬКО Linux + AMD GPU с поддержкой DMA-BUF или AMD GPU VA.
 *        На других системах все тесты пропускаются (SKIP).
 * @note  OpenCL и ROCm должны использовать одно и то же физическое GPU (device 0).
 *
 * @see   DrvGPU/tests/test_zero_copy.hpp    — базовые тесты ZeroCopyBridge
 * @see   modules/capon/tests/test_capon_rocm.hpp  — базовые тесты CaponProcessor
 * @see   modules/capon/tests/README.md       — описание модуля capon
 *
 * @author Кодо (AI Assistant)
 * @date   2026-03-23
 */

#if ENABLE_ROCM

// ── Алгоритм Кейпона ─────────────────────────────────────────────────────
#include "capon_processor.hpp"

// ── GPU инфраструктура ────────────────────────────────────────────────────
#include "backends/opencl/opencl_backend.hpp"
#include "backends/opencl/opencl_export.hpp"
#include "backends/rocm/rocm_backend.hpp"
#include "backends/rocm/zero_copy_bridge.hpp"
#include "services/console_output.hpp"

// ── OpenCL и HIP ──────────────────────────────────────────────────────────
#include <CL/cl.h>
#include <hip/hip_runtime.h>

// ── Стандартные ───────────────────────────────────────────────────────────
#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace test_capon_opencl_to_rocm {

using cx = std::complex<float>;
using namespace drv_gpu_lib;

// ════════════════════════════════════════════════════════════════════════════
// Вывод на консоль (мультиGPU-безопасный через ConsoleOutput)
// ════════════════════════════════════════════════════════════════════════════

inline void TestPrint(const std::string& msg) {
  ConsoleOutput::GetInstance().Print(0, "Capon[OCL→ROCm]", msg);
}

// ════════════════════════════════════════════════════════════════════════════
// Backend-синглтоны — инициализируются один раз, используются всеми тестами
//
// Почему статические: каждая инициализация занимает ~100мс (загрузка
// OpenCL/ROCm контекстов). Создаём один раз для всего набора тестов.
// ════════════════════════════════════════════════════════════════════════════

static OpenCLBackend& GetClBackend() {
  static OpenCLBackend cl;
  static bool inited = false;
  if (!inited) {
    cl.Initialize(0);  // device 0 = тот же GPU что и ROCm
    inited = true;
  }
  return cl;
}

static ROCmBackend& GetRocmBackend() {
  static ROCmBackend rocm;
  static bool inited = false;
  if (!inited) {
    rocm.Initialize(0);  // device 0 = тот же GPU что и OpenCL
    inited = true;
  }
  return rocm;
}

// ════════════════════════════════════════════════════════════════════════════
// Вспомогательные функции генерации тестовых данных
// ════════════════════════════════════════════════════════════════════════════

/**
 * ULA steering matrix: u[p,m] = exp(j*2π*p*0.5*sin(θ_m))
 * Хранение: column-major, [p + m*P]  ← C++ column-major стандарт для BLAS
 */
static std::vector<cx> MakeSteeringMatrix(
    uint32_t n_channels, uint32_t n_directions,
    float theta_min_rad, float theta_max_rad) {
  std::vector<cx> U(static_cast<size_t>(n_channels) * n_directions);
  for (uint32_t m = 0; m < n_directions; ++m) {
    float theta = (n_directions > 1)
        ? theta_min_rad + (theta_max_rad - theta_min_rad) * m / (n_directions - 1)
        : theta_min_rad;
    float d_sin = std::sin(theta) * 0.5f;  // d/λ = 0.5 (полуволновой интервал)
    for (uint32_t p = 0; p < n_channels; ++p) {
      float phase = 2.0f * static_cast<float>(M_PI) * p * d_sin;
      U[m * n_channels + p] = cx(std::cos(phase), std::sin(phase));
    }
  }
  return U;
}

/**
 * Комплексный гауссов шум CN(0, sigma²) через LCG + Box-Muller.
 * Воспроизводимый (детерминированный seed) — результаты не зависят от ОС.
 */
static std::vector<cx> MakeNoise(size_t count, float sigma = 1.0f,
                                 uint32_t seed = 42) {
  std::vector<cx> noise(count);
  uint32_t state = seed;
  auto rng = [&]() -> float {
    state = state * 1664525u + 1013904223u;          // LCG
    return static_cast<float>(state) / 4294967295.0f; // [0, 1)
  };
  for (size_t i = 0; i < count; ++i) {
    float u1 = rng(); if (u1 < 1e-10f) u1 = 1e-10f;  // защита log(0)
    float u2 = rng();
    float mag = sigma * std::sqrt(-2.0f * std::log(u1));  // Box-Muller
    float phi = 2.0f * static_cast<float>(M_PI) * u2;
    noise[i]  = cx(mag * std::cos(phi), mag * std::sin(phi));
  }
  return noise;
}

// ════════════════════════════════════════════════════════════════════════════
// Общая проверка доступности Zero Copy — вызывается в начале каждого теста
// ════════════════════════════════════════════════════════════════════════════

/**
 * @return true если ZeroCopy доступен на устройстве device 0.
 *         false + вывод SKIP если не поддерживается.
 */
static bool CheckZeroCopyAvailable(const char* test_name) {
  auto& cl = GetClBackend();
  cl_device_id cl_device = static_cast<cl_device_id>(cl.GetNativeDevice());
  auto method = DetectBestZeroCopyMethod(cl_device);

  if (method != ZeroCopyMethod::DMA_BUF && method != ZeroCopyMethod::AMD_GPU_VA) {
    char buf[128];
    std::snprintf(buf, sizeof(buf),
        "[%s] SKIP — ZeroCopy не поддерживается (method=%s)",
        test_name, ZeroCopyMethodToString(method));
    TestPrint(buf);
    return false;
  }
  return true;
}

// ════════════════════════════════════════════════════════════════════════════
// Test 01: detect_interop
//
// Задача: выяснить, поддерживает ли наш GPU Zero Copy между OpenCL и ROCm.
//         Вывести возможности устройства для диагностики.
//
// Этот тест никогда не падает — он только сообщает, что доступно.
// Остальные тесты пропустятся (SKIP) если метод == NONE.
// ════════════════════════════════════════════════════════════════════════════

inline void test_01_detect_interop() {
  TestPrint("[01] detect_interop — проверить возможности Zero Copy");

  auto& cl = GetClBackend();
  cl_device_id cl_device = static_cast<cl_device_id>(cl.GetNativeDevice());

  bool has_dma_buf = SupportsDmaBufExport(cl_device);
  bool has_amd_va  = SupportsAmdGpuVA(cl_device);
  auto method      = DetectBestZeroCopyMethod(cl_device);

  char buf[256];
  std::snprintf(buf, sizeof(buf),
      "  Capabilities: DMA-BUF=%s  AMD-GPU-VA=%s",
      has_dma_buf ? "YES" : "NO",
      has_amd_va  ? "YES" : "NO");
  TestPrint(buf);

  std::snprintf(buf, sizeof(buf),
      "  Selected method: %s", ZeroCopyMethodToString(method));
  TestPrint(buf);

  if (method == ZeroCopyMethod::NONE) {
    TestPrint("[01] SKIP — аппаратный Zero Copy недоступен");
  } else {
    TestPrint("[01] PASS — Zero Copy готов к использованию");
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Test 02: signal_from_opencl
//
// Задача: полный путь «OpenCL upload → Zero Copy → Capon на ROCm».
//
//   1. Генерируем матрицу сигнала Y [P×N] и управляющие векторы U [P×M] на CPU
//   2. Загружаем обе матрицы в VRAM через OpenCL (clEnqueueWriteBuffer)
//   3. Синхронизируем OpenCL (clFinish) — ОБЯЗАТЕЛЬНО перед ZeroCopy
//   4. ZeroCopyBridge создаёт hip_ptr на те же данные в VRAM (0 копий)
//   5. CaponProcessor считает рельеф через HIP-указатели
//   6. Проверяем: все z[m] > 0 и конечные
// ════════════════════════════════════════════════════════════════════════════

inline void test_02_signal_from_opencl() {
  TestPrint("[02] signal_from_opencl — сигнал через cl_mem → Кейпон на ROCm");

  if (!CheckZeroCopyAvailable("02")) return;

  auto& cl   = GetClBackend();
  auto& rocm = GetRocmBackend();
  cl_device_id cl_device = static_cast<cl_device_id>(cl.GetNativeDevice());

  // Размерность задачи
  const uint32_t P = 8;   // P антенн (каналов)
  const uint32_t N = 64;  // N временных отсчётов
  const uint32_t M = 16;  // M направлений сканирования

  // ── Шаг 1: Генерация тестовых данных на CPU ─────────────────────────────
  // Y [P×N] — матрица сигнала (чистый шум, column-major: индекс [p,n] = n*P+p)
  auto signal_cpu   = MakeNoise(P * N, 1.0f, 42u);
  // U [P×M] — управляющие векторы ULA (column-major: индекс [p,m] = m*P+p)
  auto steering_cpu = MakeSteeringMatrix(P, M,
      -static_cast<float>(M_PI) / 3.0f,   // -60°
       static_cast<float>(M_PI) / 3.0f);  // +60°

  const size_t bytes_Y = signal_cpu.size()   * sizeof(cx); // P*N * 8 байт
  const size_t bytes_U = steering_cpu.size() * sizeof(cx); // P*M * 8 байт

  // ── Шаг 2: Выделить OpenCL буферы в VRAM ────────────────────────────────
  // cl.Allocate(bytes) вызывает clCreateBuffer(CL_MEM_READ_WRITE, bytes)
  // Возвращает void* = cl_mem (нужен static_cast обратно)
  void* cl_Y = cl.Allocate(bytes_Y);
  void* cl_U = cl.Allocate(bytes_U);

  // ── Шаг 3: Загрузить данные CPU → GPU (один DMA-трансфер) ───────────────
  // Это единственная операция копирования в этом pipeline.
  // clEnqueueWriteBuffer → данные попадают в VRAM.
  cl.MemcpyHostToDevice(cl_Y, signal_cpu.data(),   bytes_Y);
  cl.MemcpyHostToDevice(cl_U, steering_cpu.data(), bytes_U);

  // ── Шаг 4: clFinish — синхронизация OpenCL ──────────────────────────────
  // КРИТИЧНО: ZeroCopyBridge не синхронизирует OpenCL автоматически.
  // Без Synchronize() ROCm может начать читать данные до завершения записи!
  cl.Synchronize();  // clFinish(cl_command_queue)

  bool ok = false;
  try {
    // ── Шаг 5: Zero Copy — cl_mem → HIP pointer ───────────────────────────
    //
    // ZeroCopyBridge::ImportFromOpenCl() автоматически выбирает лучший метод:
    //   - AMD GPU VA (если GPU поддерживает CL_MEM_AMD_GPU_VA) — 0 overhead
    //   - DMA-BUF   (dma_buf fd → hipImportExternalMemory)     — ~мкс
    //
    // После ImportFromOpenCl():
    //   bridge.GetHipPtr() указывает на ТЕ ЖЕ байты в VRAM что и cl_Y/cl_U.
    //   Никакого копирования данных не происходит.
    //
    // ВАЖНО: bridge должен оставаться живым пока HIP использует указатель!
    //        Деструктор bridge освобождает handle при выходе из блока.
    ZeroCopyBridge bridge_Y;
    ZeroCopyBridge bridge_U;

    bridge_Y.ImportFromOpenCl(static_cast<cl_mem>(cl_Y), bytes_Y, cl_device);
    bridge_U.ImportFromOpenCl(static_cast<cl_mem>(cl_U), bytes_U, cl_device);

    // Проверить, что импорт прошёл успешно
    assert(bridge_Y.IsActive() && bridge_Y.GetHipPtr() != nullptr);
    assert(bridge_U.IsActive() && bridge_U.GetHipPtr() != nullptr);

    {
      char buf[256];
      std::snprintf(buf, sizeof(buf),
          "  ZeroCopy: method=%s  hip_Y=0x%zx  hip_U=0x%zx",
          ZeroCopyMethodToString(bridge_Y.GetMethod()),
          reinterpret_cast<size_t>(bridge_Y.GetHipPtr()),
          reinterpret_cast<size_t>(bridge_U.GetHipPtr()));
      TestPrint(buf);
    }

    // ── Шаг 6: Расчёт Кейпона на ROCm ────────────────────────────────────
    //
    // CaponProcessor::ComputeRelief(void* gpu_Y, void* gpu_U, params)
    // принимает HIP device pointer-ы. Внутри:
    //   ① D2D copy: gpu_Y → internal kSignal buffer (hipMemcpyDeviceToDevice)
    //   ② D2D copy: gpu_U → internal kSteering buffer
    //   ③ CovarianceMatrixOp:  R = (1/N)*Y*Y^H + μI  (rocBLAS CGEMM)
    //   ④ CaponInvertOp:       R^{-1}  (rocSOLVER POTRF + POTRI)
    //   ⑤ ComputeWeightsOp:   W = R^{-1}*U  (rocBLAS CGEMM)
    //   ⑥ CaponReliefOp:      z[m] = 1/Re(u_m^H * W[:,m])  (HIP kernel)
    //   ⑦ Download:           z → CPU  (hipMemcpy D2H)
    capon::CaponParams params;
    params.n_channels   = P;
    params.n_samples    = N;
    params.n_directions = M;
    params.mu           = 0.01f;  // диагональная нагрузка (регуляризация)

    capon::CaponProcessor processor(&rocm);
    auto result = processor.ComputeRelief(
        bridge_Y.GetHipPtr(),   // void* gpu_signal   — Y в VRAM
        bridge_U.GetHipPtr(),   // void* gpu_steering — U в VRAM
        params);

    // ── Шаг 7: Проверка результата ────────────────────────────────────────
    assert(result.relief.size() == M);

    // Физическое требование MVDR: все z[m] должны быть > 0 и конечными.
    // (z[m] = 1/позитивная_форма → всегда > 0 при корректных данных)
    for (uint32_t m = 0; m < M; ++m) {
      assert(std::isfinite(result.relief[m]));
      assert(result.relief[m] > 0.0f);
    }

    float z_min = *std::min_element(result.relief.begin(), result.relief.end());
    float z_max = *std::max_element(result.relief.begin(), result.relief.end());
    {
      char buf[128];
      std::snprintf(buf, sizeof(buf),
          "  Relief[M=%u]: min=%.4g  max=%.4g  (all > 0, finite ✓)", M, z_min, z_max);
      TestPrint(buf);
    }

    ok = true;
    // bridge_Y и bridge_U автоматически освобождают HIP handles здесь

  } catch (const std::exception& e) {
    TestPrint(std::string("  EXCEPTION: ") + e.what());
  }

  // Освободить cl_mem ПОСЛЕ уничтожения bridge (LIFO порядок важен!)
  cl.Free(cl_Y);
  cl.Free(cl_U);

  assert(ok);
  TestPrint("[02] PASS");
}

// ════════════════════════════════════════════════════════════════════════════
// Test 03: results_match_ref
//
// Задача: доказать математическую прозрачность Zero Copy.
//
// ГАРАНТИЯ: Zero Copy должен давать ИДЕНТИЧНЫЙ результат прямой загрузке
//           через CPU. Пользователь использует ту же физику — тот же ответ.
//
// Сравнение:
//   Путь A (reference): CPU vector → CaponProcessor::ComputeRelief(vector,...)
//   Путь B (opencl):    CPU vector → cl_mem → ZeroCopy → ComputeRelief(void*,...)
//
// Оба вычисления используют одни и те же данные и один и тот же GPU.
// Результаты должны совпасть с точностью до float32 погрешности (~1e-5).
// ════════════════════════════════════════════════════════════════════════════

inline void test_03_results_match_ref() {
  TestPrint("[03] results_match_ref — сравнить: OpenCL-путь == CPU-путь");

  if (!CheckZeroCopyAvailable("03")) return;

  auto& cl   = GetClBackend();
  auto& rocm = GetRocmBackend();
  cl_device_id cl_device = static_cast<cl_device_id>(cl.GetNativeDevice());

  const uint32_t P = 8;
  const uint32_t N = 64;
  const uint32_t M = 16;

  // Одни и те же данные для обоих путей (одинаковый seed!)
  auto signal_cpu   = MakeNoise(P * N, 1.0f, 77u);
  auto steering_cpu = MakeSteeringMatrix(P, M,
      -static_cast<float>(M_PI) / 3.0f,
       static_cast<float>(M_PI) / 3.0f);

  const size_t bytes_Y = signal_cpu.size()   * sizeof(cx);
  const size_t bytes_U = steering_cpu.size() * sizeof(cx);

  capon::CaponParams params{P, N, M, 0.01f};

  // ── Путь A: прямой вызов с CPU данными (reference) ──────────────────────
  capon::CaponProcessor proc_ref(&rocm);
  capon::CaponReliefResult relief_ref = proc_ref.ComputeRelief(
      signal_cpu, steering_cpu, params);

  // ── Путь B: через OpenCL cl_mem + ZeroCopy ──────────────────────────────
  capon::CaponReliefResult relief_ocl;
  bool ok = false;

  try {
    // Загрузка через OpenCL (тот же единственный DMA-трансфер как в тесте 02)
    void* cl_Y = cl.Allocate(bytes_Y);
    void* cl_U = cl.Allocate(bytes_U);
    cl.MemcpyHostToDevice(cl_Y, signal_cpu.data(),   bytes_Y);
    cl.MemcpyHostToDevice(cl_U, steering_cpu.data(), bytes_U);
    cl.Synchronize();  // clFinish — обязательно перед ZeroCopy!

    ZeroCopyBridge bridge_Y, bridge_U;
    bridge_Y.ImportFromOpenCl(static_cast<cl_mem>(cl_Y), bytes_Y, cl_device);
    bridge_U.ImportFromOpenCl(static_cast<cl_mem>(cl_U), bytes_U, cl_device);

    capon::CaponProcessor proc_ocl(&rocm);
    relief_ocl = proc_ocl.ComputeRelief(
        bridge_Y.GetHipPtr(),
        bridge_U.GetHipPtr(),
        params);

    cl.Free(cl_Y);
    cl.Free(cl_U);
    ok = true;
  } catch (const std::exception& e) {
    TestPrint(std::string("  EXCEPTION: ") + e.what());
  }

  assert(ok);
  assert(relief_ocl.relief.size() == M);

  // ── Сравнение результатов ─────────────────────────────────────────────
  // Оба пути: одинаковые данные → один GPU → одинаковый алгоритм.
  // Максимальное расхождение должно быть ~0 (float32 точность).
  // Допуск 1e-4: учитываем порядок float32-операций внутри CaponProcessor.
  float max_diff  = 0.0f;
  float max_reldiff = 0.0f;
  for (uint32_t m = 0; m < M; ++m) {
    float diff    = std::fabs(relief_ref.relief[m] - relief_ocl.relief[m]);
    float reldiff = diff / (std::fabs(relief_ref.relief[m]) + 1e-30f);
    if (diff    > max_diff)    max_diff    = diff;
    if (reldiff > max_reldiff) max_reldiff = reldiff;
  }

  {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "  max |z_ref - z_opencl| = %.3e   max_rel = %.3e   (допуск < 1e-4)",
        max_diff, max_reldiff);
    TestPrint(buf);
  }

  // Если допуск нарушен — Zero Copy что-то испортил в данных!
  assert(max_diff < 1e-4f &&
         "Zero Copy изменил данные: результаты не совпадают с прямым путём");
  TestPrint("[03] PASS — Zero Copy прозрачен: результаты идентичны");
}

// ════════════════════════════════════════════════════════════════════════════
// Test 04: beamform_from_opencl
//
// Задача: проверить AdaptiveBeamform с входом через OpenCL cl_mem.
//
// AdaptiveBeamform вычисляет Y_out = (R^{-1}*U)^H * Y — адаптивные веса
// применяются к входному сигналу. Выход: матрица [M × N] комплексных отсчётов,
// где строка m — выход адаптивного луча для направления m.
// ════════════════════════════════════════════════════════════════════════════

inline void test_04_beamform_from_opencl() {
  TestPrint("[04] beamform_from_opencl — адаптивное ДО с входом через cl_mem");

  if (!CheckZeroCopyAvailable("04")) return;

  auto& cl   = GetClBackend();
  auto& rocm = GetRocmBackend();
  cl_device_id cl_device = static_cast<cl_device_id>(cl.GetNativeDevice());

  const uint32_t P = 4;  // антенны
  const uint32_t N = 32; // отсчёты
  const uint32_t M = 6;  // направления

  auto signal_cpu   = MakeNoise(P * N, 1.0f, 13u);
  auto steering_cpu = MakeSteeringMatrix(P, M,
      -static_cast<float>(M_PI) / 6.0f,   // -30°
       static_cast<float>(M_PI) / 6.0f);  // +30°

  const size_t bytes_Y = signal_cpu.size()   * sizeof(cx);
  const size_t bytes_U = steering_cpu.size() * sizeof(cx);

  bool ok = false;
  try {
    // OpenCL upload + Synchronize + ZeroCopy (тот же паттерн что в тесте 02)
    void* cl_Y = cl.Allocate(bytes_Y);
    void* cl_U = cl.Allocate(bytes_U);
    cl.MemcpyHostToDevice(cl_Y, signal_cpu.data(),   bytes_Y);
    cl.MemcpyHostToDevice(cl_U, steering_cpu.data(), bytes_U);
    cl.Synchronize();

    ZeroCopyBridge bridge_Y, bridge_U;
    bridge_Y.ImportFromOpenCl(static_cast<cl_mem>(cl_Y), bytes_Y, cl_device);
    bridge_U.ImportFromOpenCl(static_cast<cl_mem>(cl_U), bytes_U, cl_device);

    capon::CaponParams params{P, N, M, 0.01f};
    capon::CaponProcessor processor(&rocm);

    // AdaptiveBeamform(void*, void*, params) — GPU overload с HIP-указателями
    auto result = processor.AdaptiveBeamform(
        bridge_Y.GetHipPtr(),   // gpu_signal   [P × N]
        bridge_U.GetHipPtr(),   // gpu_steering [P × M]
        params);

    // Проверка размерности выхода: должно быть [M × N]
    assert(result.n_directions == M);
    assert(result.n_samples    == N);
    assert(result.output.size() == static_cast<size_t>(M) * N);

    // Проверка конечности — никаких NaN/Inf
    size_t n_finite = 0;
    for (const auto& v : result.output) {
      if (std::isfinite(v.real()) && std::isfinite(v.imag())) ++n_finite;
    }
    assert(n_finite == result.output.size());

    {
      char buf[128];
      std::snprintf(buf, sizeof(buf),
          "  Beamform output: [%u × %u] = %zu elements, all finite ✓",
          result.n_directions, result.n_samples, result.output.size());
      TestPrint(buf);
    }

    cl.Free(cl_Y);
    cl.Free(cl_U);
    ok = true;
  } catch (const std::exception& e) {
    TestPrint(std::string("  EXCEPTION: ") + e.what());
  }

  assert(ok);
  TestPrint("[04] PASS");
}

// ════════════════════════════════════════════════════════════════════════════
// run() — точка входа, вызывается из all_test.hpp
// ════════════════════════════════════════════════════════════════════════════

inline void run() {
  ConsoleOutput::GetInstance().Start();
  TestPrint("=== test_capon_opencl_to_rocm: данные от OpenCL → расчёт Кейпона на ROCm ===");

  test_01_detect_interop();
  test_02_signal_from_opencl();
  test_03_results_match_ref();
  test_04_beamform_from_opencl();

  TestPrint("=== test_capon_opencl_to_rocm DONE ===");
}

}  // namespace test_capon_opencl_to_rocm

// ════════════════════════════════════════════════════════════════════════════
// Заглушка для не-ROCm сборки (Windows / ENABLE_ROCM=OFF)
// ════════════════════════════════════════════════════════════════════════════

#else  // !ENABLE_ROCM

namespace test_capon_opencl_to_rocm {
inline void run() { /* SKIPPED: ENABLE_ROCM not defined */ }
}  // namespace test_capon_opencl_to_rocm

#endif  // ENABLE_ROCM
