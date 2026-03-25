#pragma once

/**
 * @file test_capon_opencl_to_rocm.hpp
 * @brief Данные заказчика -> OpenCL -> Zero Copy -> Capon на ROCm
 *
 * ======================================================================
 * НАЗНАЧЕНИЕ
 * ======================================================================
 *
 * Демонстрация полного production pipeline с РЕАЛЬНЫМИ данными заказчика:
 *
 *   1. Загрузка данных заказчика (MATLAB сигнал, координаты антенн)
 *   2. Запись на GPU через OpenCL (cl_mem) — как в реальной DSP-системе
 *   3. Передача из OpenCL в ROCm (Zero Copy — без копирования!)
 *   4. Расчёт алгоритма Кейпона на ROCm (GPU pipeline)
 *
 * ======================================================================
 * ДАННЫЕ ЗАКАЗЧИКА
 * ======================================================================
 *
 * Файлы из Doc_Addition/Capon/capon_test/build/:
 *   - x_data.txt        — координаты x 340 антенных элементов
 *   - y_data.txt        — координаты y 340 антенных элементов
 *   - signal_matlab.txt — сигнал [341 строка x 1000 complex], формат MATLAB
 *   - z_values.txt      — эталонные значения рельефа (для верификации)
 *
 * Физические параметры:
 *   f0 = 3918 МГц + 3.15 МГц = 3 921 150 000 Гц
 *   c  = 299 792 458 м/с
 *   P  = 85 антенных каналов (подмассив из 340)
 *   N  = 1000 временных отсчётов
 *
 * ======================================================================
 * ТЕСТЫ
 * ======================================================================
 *
 *   01. detect_interop              — возможности Zero Copy на данном GPU
 *   02. customer_data_pipeline      — ПОЛНЫЙ PIPELINE с данными заказчика
 *   03. zerocopy_matches_direct     — доказать: ZeroCopy путь == прямой путь
 *   04. beamform_customer_data      — адаптивное ДО с данными заказчика
 *
 * ======================================================================
 * ЗАВИСИМОСТИ
 * ======================================================================
 *
 *   - OpenCLBackend  (DrvGPU/backends/opencl/)
 *   - ROCmBackend    (DrvGPU/backends/rocm/)
 *   - ZeroCopyBridge (DrvGPU/backends/rocm/zero_copy_bridge.hpp)
 *   - CaponProcessor (modules/capon/include/capon_processor.hpp)
 *
 * @note  ТОЛЬКО Linux + AMD GPU с HSA Probe, DMA-BUF или SVM.
 * @note  OpenCL и ROCm должны использовать одно физическое GPU (device 0).
 *
 * @author Кодо (AI Assistant)
 * @date   2026-03-24
 */

#if ENABLE_ROCM

// -- Алгоритм Кейпона ---------------------------------------------------
#include "capon_processor.hpp"
#include "capon_test_helpers.hpp"

// -- GPU инфраструктура --------------------------------------------------
#include "backends/opencl/opencl_backend.hpp"
#include "backends/opencl/opencl_export.hpp"
#include "backends/rocm/rocm_backend.hpp"
#include "backends/rocm/zero_copy_bridge.hpp"
#include "services/console_output.hpp"

// -- OpenCL и HIP --------------------------------------------------------
#include <CL/cl.h>
#include <hip/hip_runtime.h>

// -- Стандартные ---------------------------------------------------------
#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace test_capon_opencl_to_rocm {

using cx = std::complex<float>;
using namespace drv_gpu_lib;
using namespace capon_test_helpers;

// ========================================================================
// Консольный вывод (мультиGPU-безопасный)
// ========================================================================

inline void TestPrint(const std::string& msg) {
  ConsoleOutput::GetInstance().Print(0, "Capon[OCL->ROCm]", msg);
}

// ========================================================================
// Backend-синглтоны
// ========================================================================

inline OpenCLBackend& GetClBackend() {
  static OpenCLBackend cl;
  static bool inited = false;
  if (!inited) {
    cl.Initialize(0);
    inited = true;
  }
  return cl;
}

inline ROCmBackend& GetRocmBackend() {
  return capon_test_helpers::GetROCmBackend();
}

// ========================================================================
// Проверка доступности Zero Copy
// ========================================================================

inline bool CheckZeroCopyAvailable(const char* test_name) {
  auto& cl = GetClBackend();
  cl_device_id cl_device = static_cast<cl_device_id>(cl.GetNativeDevice());
  auto method = DetectBestZeroCopyMethod(cl_device);

  // ImportFromOpenCl поддерживает: HSA_PROBE, DMA_BUF и SVM (fallback)
  if (method == ZeroCopyMethod::NONE) {
    char buf[128];
    std::snprintf(buf, sizeof(buf),
        "[%s] SKIP -- ZeroCopy not supported (method=%s)",
        test_name, ZeroCopyMethodToString(method));
    TestPrint(buf);
    return false;
  }
  return true;
}

// ========================================================================
// Test 01: detect_interop
// ========================================================================

inline void test_01_detect_interop() {
  TestPrint("[01] detect_interop -- capabilities of Zero Copy on this GPU");

  auto& cl = GetClBackend();
  cl_device_id cl_device = static_cast<cl_device_id>(cl.GetNativeDevice());

  bool has_hsa     = SupportsHsaProbe();
  bool has_dma_buf = SupportsDmaBufExport(cl_device);
  bool has_svm     = SupportsSVMZeroCopy(cl_device);
  auto method      = DetectBestZeroCopyMethod(cl_device);

  char buf[256];
  std::snprintf(buf, sizeof(buf),
      "  Capabilities: HSA-Probe=%s  DMA-BUF=%s  SVM=%s",
      has_hsa     ? "YES" : "NO",
      has_dma_buf ? "YES" : "NO",
      has_svm     ? "YES" : "NO");
  TestPrint(buf);

  std::snprintf(buf, sizeof(buf),
      "  Selected method: %s", ZeroCopyMethodToString(method));
  TestPrint(buf);

  if (method != ZeroCopyMethod::NONE) {
    TestPrint("[01] PASS -- Zero Copy ready (" +
             std::string(ZeroCopyMethodToString(method)) + ")");
  } else {
    TestPrint("[01] SKIP -- hardware Zero Copy unavailable");
  }
}

// ========================================================================
//
//   Test 02: ПОЛНЫЙ PIPELINE С ДАННЫМИ ЗАКАЗЧИКА
//
//   Этот тест демонстрирует production-сценарий от начала до конца:
//   данные заказчика проходят весь путь через систему.
//
// ========================================================================

inline void test_02_customer_data_pipeline() {
  TestPrint("[02] ===== ПОЛНЫЙ PIPELINE: данные заказчика -> OpenCL -> ZeroCopy -> Capon =====");

  if (!CheckZeroCopyAvailable("02")) return;

  auto& cl   = GetClBackend();
  auto& rocm = GetRocmBackend();
  cl_device_id cl_device = static_cast<cl_device_id>(cl.GetNativeDevice());

  // ╔══════════════════════════════════════════════════════════════════════╗
  // ║                                                                      ║
  // ║   ЭТАП 1: ЗАГРУЗКА ДАННЫХ ЗАКАЗЧИКА                                  ║
  // ║                                                                      ║
  // ║   Источник: MATLAB файлы из Doc_Addition/Capon/capon_test/build/     ║
  // ║     - x_data.txt, y_data.txt — координаты 340 антенных элементов     ║
  // ║     - signal_matlab.txt      — сигнал [341 x 1000] complex           ║
  // ║                                                                      ║
  // ║   Используем:                                                        ║
  // ║     P = 85 каналов (подмассив антенной решётки)                      ║
  // ║     N = 1000 временных отсчётов                                      ║
  // ║     M = 37 направлений (1D сканирование)                             ║
  // ║                                                                      ║
  // ╚══════════════════════════════════════════════════════════════════════╝

  TestPrint("  ---[ ЭТАП 1: ЗАГРУЗКА ДАННЫХ ЗАКАЗЧИКА ]---");

  // 1.1. Координаты антенных элементов
  std::vector<float> x_all, y_all;
  if (!LoadRealVector(kDataDir + "x_data.txt", x_all) ||
      !LoadRealVector(kDataDir + "y_data.txt", y_all)) {
    TestPrint("[02] SKIP -- customer data not found: x_data.txt / y_data.txt");
    return;
  }

  const uint32_t P = 85;   // число антенных каналов (подмассив)
  const uint32_t N = 1000;  // число временных отсчётов

  if (x_all.size() < P || y_all.size() < P) {
    TestPrint("[02] SKIP -- not enough antenna elements in coordinate files");
    return;
  }

  // Первые P элементов — рабочий подмассив
  std::vector<float> x_sub(x_all.begin(), x_all.begin() + P);
  std::vector<float> y_sub(y_all.begin(), y_all.begin() + P);

  // 1.2. Сигнальная матрица от заказчика (MATLAB формат)
  std::vector<cx> signal;
  if (!LoadSignalMatlab(kDataDir + "signal_matlab.txt", P, N, signal)) {
    TestPrint("[02] SKIP -- signal_matlab.txt not found or parse error");
    return;
  }

  // 1.3. Управляющие векторы по реальным координатам
  //      1D сканирование: M = 37 направлений, v = 0
  auto u0 = MakeScanGrid1D(3.25, 0.00312);
  const uint32_t M = static_cast<uint32_t>(u0.size());  // ~37
  std::vector<float> v0(M, 0.0f);  // 1D сканирование: v = 0

  auto steering = MakePhysicalSteering(x_sub, y_sub, u0, v0, kF0, kC);

  {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "  Loaded: P=%u channels, N=%u samples, M=%u directions",
        P, N, M);
    TestPrint(buf);
    std::snprintf(buf, sizeof(buf),
        "  Signal: %zu complex values (%.1f KB)",
        signal.size(), signal.size() * sizeof(cx) / 1024.0);
    TestPrint(buf);
    std::snprintf(buf, sizeof(buf),
        "  Steering: %zu complex values (%.1f KB)",
        steering.size(), steering.size() * sizeof(cx) / 1024.0);
    TestPrint(buf);
  }

  const size_t bytes_Y = signal.size()   * sizeof(cx);  // P*N * 8
  const size_t bytes_U = steering.size() * sizeof(cx);   // P*M * 8

  // ╔══════════════════════════════════════════════════════════════════════╗
  // ║                                                                      ║
  // ║   ЭТАП 2: ЗАПИСЬ НА GPU ЧЕРЕЗ OpenCL (cl_mem)                        ║
  // ║                                                                      ║
  // ║   В реальной DSP-системе данные уже находятся в OpenCL:              ║
  // ║   вышли из предыдущего этапа обработки (FFT, фильтрация и т.д.)      ║
  // ║                                                                      ║
  // ║   Здесь мы эмулируем этот сценарий:                                  ║
  // ║     CPU RAM ──[ clEnqueueWriteBuffer ]──> GPU VRAM (cl_mem)          ║
  // ║                                                                      ║
  // ║   Это ЕДИНСТВЕННАЯ операция копирования данных в этом pipeline.       ║
  // ║                                                                      ║
  // ╚══════════════════════════════════════════════════════════════════════╝

  TestPrint("  ---[ ЭТАП 2: ЗАПИСЬ НА GPU ЧЕРЕЗ OpenCL ]---");

  // 2.1. Выделить OpenCL буферы в VRAM
  void* cl_Y = cl.Allocate(bytes_Y);  // cl_mem для сигнала Y [P x N]
  void* cl_U = cl.Allocate(bytes_U);  // cl_mem для управляющих векторов U [P x M]

  // 2.2. Загрузить данные заказчика CPU -> GPU (один DMA-трансфер)
  cl.MemcpyHostToDevice(cl_Y, signal.data(),   bytes_Y);
  cl.MemcpyHostToDevice(cl_U, steering.data(), bytes_U);

  // 2.3. Синхронизация OpenCL (clFinish)
  //      КРИТИЧНО: без этого ROCm может начать читать незаписанные данные!
  cl.Synchronize();

  {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "  cl_mem Y: %zu bytes (signal [%u x %u])", bytes_Y, P, N);
    TestPrint(buf);
    std::snprintf(buf, sizeof(buf),
        "  cl_mem U: %zu bytes (steering [%u x %u])", bytes_U, P, M);
    TestPrint(buf);
    TestPrint("  clFinish() -- OpenCL synchronized");
  }

  bool ok = false;
  try {

    // ╔══════════════════════════════════════════════════════════════════════╗
    // ║                                                                      ║
    // ║   ЭТАП 3: ПЕРЕДАЧА ИЗ OpenCL В ROCm (ZERO COPY)                    ║
    // ║                                                                      ║
    // ║   КЛЮЧЕВОЙ МОМЕНТ: данные НЕ копируются!                            ║
    // ║                                                                      ║
    // ║   OpenCL и HIP/ROCm на одном AMD GPU разделяют одну VRAM.           ║
    // ║   ZeroCopyBridge передаёт только GPU-адрес, а не данные:             ║
    // ║                                                                      ║
    // ║     cl_mem ──[ AMD GPU VA / DMA-BUF ]──> void* hip_ptr              ║
    // ║                                                                      ║
    // ║   Результат: hip_ptr указывает на ТЕ ЖЕ байты в VRAM.              ║
    // ║   Время передачи: ~наносекунды (только адрес, не данные).           ║
    // ║                                                                      ║
    // ╚══════════════════════════════════════════════════════════════════════╝

    TestPrint("  ---[ ЭТАП 3: ZERO COPY OpenCL -> ROCm ]---");

    // 3.1. Импорт cl_mem -> HIP device pointer
    ZeroCopyBridge bridge_Y;
    ZeroCopyBridge bridge_U;

    bridge_Y.ImportFromOpenCl(static_cast<cl_mem>(cl_Y), bytes_Y, cl_device);
    bridge_U.ImportFromOpenCl(static_cast<cl_mem>(cl_U), bytes_U, cl_device);

    assert(bridge_Y.IsActive() && bridge_Y.GetHipPtr() != nullptr);
    assert(bridge_U.IsActive() && bridge_U.GetHipPtr() != nullptr);

    {
      char buf[256];
      std::snprintf(buf, sizeof(buf),
          "  ZeroCopy method: %s", ZeroCopyMethodToString(bridge_Y.GetMethod()));
      TestPrint(buf);
      std::snprintf(buf, sizeof(buf),
          "  hip_Y = 0x%zx  (same VRAM, zero data movement)",
          reinterpret_cast<size_t>(bridge_Y.GetHipPtr()));
      TestPrint(buf);
      std::snprintf(buf, sizeof(buf),
          "  hip_U = 0x%zx  (same VRAM, zero data movement)",
          reinterpret_cast<size_t>(bridge_U.GetHipPtr()));
      TestPrint(buf);
    }

    // ╔══════════════════════════════════════════════════════════════════════╗
    // ║                                                                      ║
    // ║   ЭТАП 4: РАСЧЁТ АЛГОРИТМА КЕЙПОНА НА ROCm                         ║
    // ║                                                                      ║
    // ║   CaponProcessor выполняет полный GPU pipeline:                      ║
    // ║     1) CovarianceMatrixOp:  R = (1/N)*Y*Y^H + mu*I   (rocBLAS)     ║
    // ║     2) CaponInvertOp:       R^{-1}                    (rocSOLVER)   ║
    // ║     3) ComputeWeightsOp:    W = R^{-1}*U              (rocBLAS)     ║
    // ║     4) CaponReliefOp:       z[m] = 1/Re(u^H*W[:,m])   (HIP kernel) ║
    // ║                                                                      ║
    // ║   Входные данные: hip_ptr (от ZeroCopy, не от CPU!)                 ║
    // ║   Результат: M вещественных значений рельефа Кейпона                ║
    // ║                                                                      ║
    // ╚══════════════════════════════════════════════════════════════════════╝

    TestPrint("  ---[ ЭТАП 4: РАСЧЁТ КЕЙПОНА НА ROCm ]---");

    // 4.1. Параметры алгоритма
    capon::CaponParams params;
    params.n_channels   = P;      // 85 антенных каналов
    params.n_samples    = N;      // 1000 временных отсчётов
    params.n_directions = M;      // ~37 направлений сканирования
    params.mu           = 1.0f;   // регуляризация (GPU: R = Y*Y^H/N + mu*I)

    // 4.2. Запуск GPU pipeline с HIP-указателями от Zero Copy
    capon::CaponProcessor processor(&rocm);
    auto result = processor.ComputeRelief(
        bridge_Y.GetHipPtr(),   // void* -- сигнал Y в VRAM (от Zero Copy!)
        bridge_U.GetHipPtr(),   // void* -- steering U в VRAM (от Zero Copy!)
        params);

    // 4.3. Проверка результата
    assert(result.relief.size() == M);

    float z_min = result.relief[0];
    float z_max = result.relief[0];
    bool all_ok = true;
    for (uint32_t m = 0; m < M; ++m) {
      if (!std::isfinite(result.relief[m]) || result.relief[m] <= 0.0f) {
        all_ok = false;
      }
      if (result.relief[m] < z_min) z_min = result.relief[m];
      if (result.relief[m] > z_max) z_max = result.relief[m];
    }

    assert(all_ok && "Capon relief must be > 0 and finite for all directions");

    {
      char buf[256];
      std::snprintf(buf, sizeof(buf),
          "  Capon relief [M=%u]: min=%.4g  max=%.4g  ratio=%.1f",
          M, z_min, z_max, z_max / (z_min + 1e-30f));
      TestPrint(buf);
      TestPrint("  All z[m] > 0 and finite -- GPU pipeline OK");
    }

    ok = true;

  } catch (const std::exception& e) {
    TestPrint(std::string("  EXCEPTION: ") + e.what());
  }

  // Освободить cl_mem ПОСЛЕ уничтожения bridge (LIFO порядок!)
  cl.Free(cl_Y);
  cl.Free(cl_U);

  assert(ok);
  TestPrint("[02] PASS -- full customer data pipeline completed successfully");
}

// ========================================================================
//
//   Test 03: ДОКАЗАТЕЛЬСТВО ПРОЗРАЧНОСТИ ZERO COPY
//
//   Гарантия: Zero Copy путь даёт ИДЕНТИЧНЫЙ результат прямой загрузке.
//   Одни и те же данные заказчика -> два разных пути -> один результат.
//
// ========================================================================

inline void test_03_zerocopy_matches_direct() {
  TestPrint("[03] zerocopy_matches_direct -- Zero Copy == direct path (customer data)");

  if (!CheckZeroCopyAvailable("03")) return;

  auto& cl   = GetClBackend();
  auto& rocm = GetRocmBackend();
  cl_device_id cl_device = static_cast<cl_device_id>(cl.GetNativeDevice());

  // Загрузка данных заказчика (подмножество для скорости)
  std::vector<float> x_all, y_all;
  if (!LoadRealVector(kDataDir + "x_data.txt", x_all) ||
      !LoadRealVector(kDataDir + "y_data.txt", y_all)) {
    TestPrint("[03] SKIP -- customer data not found");
    return;
  }

  const uint32_t P = 16;   // подмножество каналов
  const uint32_t N = 128;  // подмножество отсчётов

  if (x_all.size() < P || y_all.size() < P) {
    TestPrint("[03] SKIP -- not enough antenna elements");
    return;
  }

  std::vector<float> x_sub(x_all.begin(), x_all.begin() + P);
  std::vector<float> y_sub(y_all.begin(), y_all.begin() + P);

  std::vector<cx> signal;
  if (!LoadSignalMatlab(kDataDir + "signal_matlab.txt", P, N, signal)) {
    TestPrint("[03] SKIP -- signal_matlab.txt not found");
    return;
  }

  // Steering (1D, M=16)
  const uint32_t M = 16;
  const double ulim = std::sin(3.25 * M_PI / 180.0);
  std::vector<float> u_dirs(M), v_dirs(M, 0.0f);
  for (uint32_t m = 0; m < M; ++m) {
    u_dirs[m] = static_cast<float>(-ulim + 2.0 * ulim * m / (M - 1));
  }
  auto steering = MakePhysicalSteering(x_sub, y_sub, u_dirs, v_dirs, kF0, kC);

  const size_t bytes_Y = signal.size()   * sizeof(cx);
  const size_t bytes_U = steering.size() * sizeof(cx);

  capon::CaponParams params{P, N, M, 1.0f};

  // -- Путь A: ПРЯМОЙ (CPU vector -> CaponProcessor) --
  capon::CaponProcessor proc_ref(&rocm);
  auto relief_ref = proc_ref.ComputeRelief(signal, steering, params);

  // -- Путь B: через OpenCL cl_mem -> Zero Copy -> CaponProcessor --
  capon::CaponReliefResult relief_ocl;
  bool ok = false;

  try {
    void* cl_Y = cl.Allocate(bytes_Y);
    void* cl_U = cl.Allocate(bytes_U);
    cl.MemcpyHostToDevice(cl_Y, signal.data(),   bytes_Y);
    cl.MemcpyHostToDevice(cl_U, steering.data(), bytes_U);
    cl.Synchronize();

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

  // -- Сравнение: оба пути должны дать идентичный результат --
  float max_diff    = 0.0f;
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
        "  max |z_direct - z_zerocopy| = %.3e   max_rel = %.3e   (tolerance < 1e-4)",
        max_diff, max_reldiff);
    TestPrint(buf);
  }

  assert(max_diff < 1e-4f &&
         "Zero Copy corrupted data: results don't match direct path");
  TestPrint("[03] PASS -- Zero Copy is transparent: results are identical");
}

// ========================================================================
//
//   Test 04: АДАПТИВНОЕ ДО С ДАННЫМИ ЗАКАЗЧИКА
//
//   AdaptiveBeamform: W = R^{-1}*U,  Y_out = W^H * Y  -> [M x N]
//   Вход через OpenCL cl_mem -> Zero Copy -> HIP ptr.
//
// ========================================================================

inline void test_04_beamform_customer_data() {
  TestPrint("[04] beamform_customer_data -- adaptive beamforming via OpenCL -> ROCm");

  if (!CheckZeroCopyAvailable("04")) return;

  auto& cl   = GetClBackend();
  auto& rocm = GetRocmBackend();
  cl_device_id cl_device = static_cast<cl_device_id>(cl.GetNativeDevice());

  // Загрузка данных заказчика (подмножество)
  std::vector<float> x_all, y_all;
  if (!LoadRealVector(kDataDir + "x_data.txt", x_all) ||
      !LoadRealVector(kDataDir + "y_data.txt", y_all)) {
    TestPrint("[04] SKIP -- customer data not found");
    return;
  }

  const uint32_t P = 16;
  const uint32_t N = 128;
  const uint32_t M = 8;

  if (x_all.size() < P || y_all.size() < P) {
    TestPrint("[04] SKIP -- not enough antenna elements");
    return;
  }

  std::vector<float> x_sub(x_all.begin(), x_all.begin() + P);
  std::vector<float> y_sub(y_all.begin(), y_all.begin() + P);

  std::vector<cx> signal;
  if (!LoadSignalMatlab(kDataDir + "signal_matlab.txt", P, N, signal)) {
    TestPrint("[04] SKIP -- signal_matlab.txt not found");
    return;
  }

  // Steering (1D, M=8 направлений)
  const double ulim = std::sin(3.25 * M_PI / 180.0);
  std::vector<float> u_dirs(M), v_dirs(M, 0.0f);
  for (uint32_t m = 0; m < M; ++m) {
    u_dirs[m] = static_cast<float>(-ulim + 2.0 * ulim * m / (M - 1));
  }
  auto steering = MakePhysicalSteering(x_sub, y_sub, u_dirs, v_dirs, kF0, kC);

  const size_t bytes_Y = signal.size()   * sizeof(cx);
  const size_t bytes_U = steering.size() * sizeof(cx);

  bool ok = false;
  try {
    // ЭТАП 2: OpenCL upload
    void* cl_Y = cl.Allocate(bytes_Y);
    void* cl_U = cl.Allocate(bytes_U);
    cl.MemcpyHostToDevice(cl_Y, signal.data(),   bytes_Y);
    cl.MemcpyHostToDevice(cl_U, steering.data(), bytes_U);
    cl.Synchronize();

    // ЭТАП 3: Zero Copy
    ZeroCopyBridge bridge_Y, bridge_U;
    bridge_Y.ImportFromOpenCl(static_cast<cl_mem>(cl_Y), bytes_Y, cl_device);
    bridge_U.ImportFromOpenCl(static_cast<cl_mem>(cl_U), bytes_U, cl_device);

    // ЭТАП 4: Расчёт AdaptiveBeamform
    capon::CaponParams params{P, N, M, 1.0f};
    capon::CaponProcessor processor(&rocm);

    auto result = processor.AdaptiveBeamform(
        bridge_Y.GetHipPtr(),
        bridge_U.GetHipPtr(),
        params);

    // Проверка размерности: [M x N]
    assert(result.n_directions == M);
    assert(result.n_samples    == N);
    assert(result.output.size() == static_cast<size_t>(M) * N);

    // Проверка конечности
    size_t n_finite = 0;
    for (const auto& v : result.output) {
      if (std::isfinite(v.real()) && std::isfinite(v.imag())) ++n_finite;
    }
    assert(n_finite == result.output.size());

    {
      char buf[128];
      std::snprintf(buf, sizeof(buf),
          "  Beamform output: [%u x %u] = %zu elements, all finite",
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

// ========================================================================
//
//   Test 05: ПУТЬ ЧЕРЕЗ SVM — данные заказчика → clSVMAlloc → ROCm
//
//   Явный SVM pipeline:
//     1. clSVMAlloc (fine-grain SVM)
//     2. clEnqueueSVMMemcpy (CPU → SVM на GPU)
//     3. ZeroCopyBridge::ImportFromSVM (SVM pointer → HIP)
//     4. CaponProcessor (ROCm pipeline)
//     5. Сравнение с прямым путём
//
// ========================================================================

inline void test_05_svm_customer_data() {
  TestPrint("[05] ===== SVM PATH: данные заказчика -> clSVMAlloc -> ROCm Capon =====");

  auto& cl   = GetClBackend();
  auto& rocm = GetRocmBackend();
  cl_device_id cl_device = static_cast<cl_device_id>(cl.GetNativeDevice());

  // Проверить поддержку fine-grain SVM
  cl_device_svm_capabilities svm_caps = 0;
  cl_int svm_err = clGetDeviceInfo(cl_device, CL_DEVICE_SVM_CAPABILITIES,
                                    sizeof(svm_caps), &svm_caps, nullptr);
  if (svm_err != CL_SUCCESS || !(svm_caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER)) {
    TestPrint("[05] SKIP -- fine-grain SVM not supported on this device");
    return;
  }

  // ── Загрузка данных заказчика ──────────────────────────────────────────
  std::vector<float> x_all, y_all;
  if (!LoadRealVector(kDataDir + "x_data.txt", x_all) ||
      !LoadRealVector(kDataDir + "y_data.txt", y_all)) {
    TestPrint("[05] SKIP -- customer data not found");
    return;
  }

  const uint32_t P = 16;
  const uint32_t N = 128;

  if (x_all.size() < P || y_all.size() < P) {
    TestPrint("[05] SKIP -- not enough antenna elements");
    return;
  }

  std::vector<float> x_sub(x_all.begin(), x_all.begin() + P);
  std::vector<float> y_sub(y_all.begin(), y_all.begin() + P);

  std::vector<cx> signal;
  if (!LoadSignalMatlab(kDataDir + "signal_matlab.txt", P, N, signal)) {
    TestPrint("[05] SKIP -- signal_matlab.txt not found");
    return;
  }

  // Steering (1D, M=16)
  const uint32_t M = 16;
  const double ulim = std::sin(3.25 * M_PI / 180.0);
  std::vector<float> u_dirs(M), v_dirs(M, 0.0f);
  for (uint32_t m = 0; m < M; ++m) {
    u_dirs[m] = static_cast<float>(-ulim + 2.0 * ulim * m / (M - 1));
  }
  auto steering = MakePhysicalSteering(x_sub, y_sub, u_dirs, v_dirs, kF0, kC);

  const size_t bytes_Y = signal.size()   * sizeof(cx);
  const size_t bytes_U = steering.size() * sizeof(cx);

  capon::CaponParams params{P, N, M, 1.0f};

  // ── Путь A: Прямой (CPU → CaponProcessor, без SVM) ────────────────────
  capon::CaponProcessor proc_ref(&rocm);
  auto relief_ref = proc_ref.ComputeRelief(signal, steering, params);

  // ── Путь B: Через SVM ──────────────────────────────────────────────────
  // Получить OpenCL context из backend
  cl_context cl_ctx = static_cast<cl_context>(cl.GetNativeContext());

  bool ok = false;
  try {
    TestPrint("  ---[ ЭТАП 1: clSVMAlloc (fine-grain SVM) ]---");

    // 1. Аллокация fine-grain SVM буферов
    void* svm_Y = clSVMAlloc(cl_ctx,
        CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_READ_WRITE, bytes_Y, 0);
    void* svm_U = clSVMAlloc(cl_ctx,
        CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_READ_WRITE, bytes_U, 0);

    if (!svm_Y || !svm_U) {
      if (svm_Y) clSVMFree(cl_ctx, svm_Y);
      if (svm_U) clSVMFree(cl_ctx, svm_U);
      TestPrint("[05] SKIP -- clSVMAlloc failed");
      return;
    }

    {
      char buf[256];
      std::snprintf(buf, sizeof(buf),
          "  SVM alloc: Y=%zu bytes (%p), U=%zu bytes (%p)",
          bytes_Y, svm_Y, bytes_U, svm_U);
      TestPrint(buf);
    }

    TestPrint("  ---[ ЭТАП 2: Запись данных заказчика в SVM ]---");

    // 2. Fine-grain SVM доступен из CPU напрямую — memcpy!
    std::memcpy(svm_Y, signal.data(),   bytes_Y);
    std::memcpy(svm_U, steering.data(), bytes_U);

    TestPrint("  memcpy CPU→SVM done (fine-grain: прямой доступ из CPU)");

    TestPrint("  ---[ ЭТАП 3: ImportFromSVM → HIP pointer ]---");

    // 3. SVM pointer → HIP через ZeroCopyBridge
    ZeroCopyBridge bridge_Y, bridge_U;
    bridge_Y.ImportFromSVM(svm_Y, bytes_Y);
    bridge_U.ImportFromSVM(svm_U, bytes_U);

    assert(bridge_Y.IsActive() && bridge_Y.GetHipPtr() != nullptr);
    assert(bridge_U.IsActive() && bridge_U.GetHipPtr() != nullptr);

    {
      char buf[256];
      std::snprintf(buf, sizeof(buf),
          "  hip_Y=%p  hip_U=%p  (SVM→HIP unified VA)",
          bridge_Y.GetHipPtr(), bridge_U.GetHipPtr());
      TestPrint(buf);
    }

    TestPrint("  ---[ ЭТАП 4: Capon на ROCm через SVM pointers ]---");

    // 4. Capon на ROCm
    capon::CaponProcessor proc_svm(&rocm);
    auto relief_svm = proc_svm.ComputeRelief(
        bridge_Y.GetHipPtr(),
        bridge_U.GetHipPtr(),
        params);

    assert(relief_svm.relief.size() == M);

    // 5. Сравнение: SVM путь == прямой путь
    float max_diff    = 0.0f;
    float max_reldiff = 0.0f;
    for (uint32_t m = 0; m < M; ++m) {
      float diff    = std::fabs(relief_ref.relief[m] - relief_svm.relief[m]);
      float reldiff = diff / (std::fabs(relief_ref.relief[m]) + 1e-30f);
      if (diff    > max_diff)    max_diff    = diff;
      if (reldiff > max_reldiff) max_reldiff = reldiff;
    }

    {
      char buf[256];
      std::snprintf(buf, sizeof(buf),
          "  |z_direct - z_svm| max=%.3e  max_rel=%.3e  (tolerance < 1e-4)",
          max_diff, max_reldiff);
      TestPrint(buf);
    }

    assert(max_diff < 1e-4f && "SVM path data corruption: results don't match direct path");

    // Cleanup: мост уничтожается перед SVM
    bridge_Y.Release();
    bridge_U.Release();
    clSVMFree(cl_ctx, svm_Y);
    clSVMFree(cl_ctx, svm_U);

    ok = true;
  } catch (const std::exception& e) {
    TestPrint(std::string("  EXCEPTION: ") + e.what());
  }

  assert(ok);
  TestPrint("[05] PASS -- SVM customer data pipeline: clSVMAlloc → ROCm Capon verified");
}

// ========================================================================
// run() -- точка входа, вызывается из all_test.hpp
// ========================================================================

inline void run() {
  ConsoleOutput::GetInstance().Start();
  TestPrint("========================================================");
  TestPrint("  test_capon_opencl_to_rocm");
  TestPrint("  Данные заказчика -> OpenCL -> Zero Copy -> Capon ROCm");
  TestPrint("========================================================");

  test_01_detect_interop();
  test_02_customer_data_pipeline();
  test_03_zerocopy_matches_direct();
  test_04_beamform_customer_data();
  test_05_svm_customer_data();

  TestPrint("=== test_capon_opencl_to_rocm DONE ===");
}

}  // namespace test_capon_opencl_to_rocm

// ========================================================================
// Заглушка для не-ROCm сборки (Windows / ENABLE_ROCM=OFF)
// ========================================================================

#else  // !ENABLE_ROCM

namespace test_capon_opencl_to_rocm {
inline void run() { /* SKIPPED: ENABLE_ROCM not defined */ }
}  // namespace test_capon_opencl_to_rocm

#endif  // ENABLE_ROCM
