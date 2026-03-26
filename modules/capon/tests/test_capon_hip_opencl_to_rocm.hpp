#pragma once

/**
 * @file test_capon_hip_opencl_to_rocm.hpp
 * @brief HIP выделяет память → OpenCL пишет данные → Capon считает на ROCm
 *
 * ======================================================================
 * ДЛЯ ЧАЙНИКОВ: О ЧЁМ ЭТОТ ТЕСТ?
 * ======================================================================
 *
 * Представь три игрока, которые используют одну и ту же видеопамять GPU:
 *
 *   HIP     — "C++ для GPU от AMD". Умеет выделять память (hipMalloc),
 *             запускать вычисления (<<<>>>). Язык ROCm.
 *
 *   OpenCL  — "Универсальный GPU-язык". Работает на AMD, NVIDIA, Intel.
 *             Умеет читать/писать память, запускать kernels.
 *
 *   CAPON   — наш алгоритм Кейпона. Работает на HIP/ROCm.
 *             Получает указатели на GPU-данные → вычисляет.
 *
 * Проблема: HIP и OpenCL — разные API. Как они делят память GPU?
 *
 * Решение (AMD-специфика): на AMD GPU оба рантайма построены поверх
 * одного слоя — HSA Runtime (ROCR). Это значит что VRAM физически одна,
 * и указатели в ней одинаковые для обоих API!
 *
 *   hipMalloc() выделяет страницу VRAM → даёт указатель 0x7f1234560000
 *   clEnqueueSVMMemcpy(dst=0x7f1234560000, ...) пишет в ТЕ ЖЕ байты
 *   CaponProcessor(ptr=0x7f1234560000) читает ТЕ ЖЕ байты
 *
 * Никакого копирования между API! Один буфер в VRAM, три участника.
 *
 * ======================================================================
 * ЗАЧЕМ ЭТО НУЖНО? (Сравнение с test_capon_opencl_to_rocm)
 * ======================================================================
 *
 * СТАРЫЙ СПОСОБ (test_capon_opencl_to_rocm.hpp):
 *
 *   1. OpenCL выделяет cl_mem (OpenCL "владеет" памятью)
 *   2. Данные пишутся в cl_mem через OpenCL
 *   3. ZeroCopyBridge конвертирует cl_mem → HIP-указатель
 *      (это как "переводчик" между двумя API)
 *   4. Capon получает HIP-указатель
 *
 *   cl_mem ──[ZeroCopyBridge]──► void* hip_ptr ──► Capon
 *
 * НОВЫЙ СПОСОБ (этот файл):
 *
 *   1. HIP выделяет hipMalloc (HIP "владеет" памятью)
 *   2. Данные пишутся через OpenCL напрямую в hipMalloc-память
 *      (ZeroCopyBridge НЕ нужен — памятью и так владеет HIP!)
 *   3. Capon получает тот же HIP-указатель от шага 1
 *
 *   hipMalloc ──[OpenCL пишет]──► void* hip_ptr ──► Capon
 *
 * ДЛЯ 4 ГБ ДАННЫХ это важно потому что:
 *   - Нет staging-буфера (не нужно 4 ГБ лишней памяти)
 *   - Нет GPU-to-GPU копирования (staging → VRAM)
 *   - clEnqueueSVMMemcpy делает один DMA-трансфер CPU → VRAM
 *
 * ======================================================================
 * ЧТО ТАКОЕ SVM? (Shared Virtual Memory)
 * ======================================================================
 *
 * SVM — режим работы OpenCL, при котором GPU и CPU видят ОДНИ и те же
 * виртуальные адреса. Без SVM: GPU не знает ничего про CPU-адреса.
 * С SVM: указатель из hipMalloc (GPU) может быть передан в OpenCL.
 *
 * Уровни SVM (от простого к сложному):
 *   Coarse-grain buffer — можно передавать GPU-указатели в OpenCL kernels
 *                         через clSetKernelArgSVMPointer (как в HIPmemTest)
 *   Fine-grain buffer   — можно выделять SVM через clSVMAlloc, CPU пишет memcpy
 *   Fine-grain system   — ЛЮБАЯ host-память видна как SVM (malloc, std::vector)
 *
 * Для clEnqueueSVMMemcpy(dst=hipMalloc_ptr, src=std::vector_data, bytes):
 *   dst = hipMalloc — coarse-grain SVM (работает на AMD всегда)
 *   src = host ptr  — нужен fine-grain system SVM
 *
 * На AMD gfx1201 (Radeon 9070): все уровни SVM поддерживаются.
 *
 * ======================================================================
 * ТЕСТЫ
 * ======================================================================
 *
 *   01. detect_hip_svm            — показываем capabilities GPU
 *   02. hip_opencl_capon_pipeline — ПОЛНЫЙ PIPELINE (данные заказчика)
 *   03. hip_opencl_matches_direct — доказываем: новый путь == прямой
 *
 * ======================================================================
 * ТРЕБОВАНИЯ
 * ======================================================================
 *
 *   - AMD GPU с ROCm 7.2+ (gfx1201 / Radeon 9070 — проверено)
 *   - OpenCL coarse-grain SVM (как в HIPmemTest)
 *   - Данные заказчика: Doc_Addition/Capon/capon_test/build/
 *   - Proof of concept: /home/alex/C++/HIPmemTest/
 *
 * @author Кодо (AI Assistant)
 * @date   2026-03-26
 */

#if ENABLE_ROCM

// ──────────────────────────────────────────────────────────────────────
// Инклюды
// ──────────────────────────────────────────────────────────────────────

// Алгоритм Кейпона и вспомогательные утилиты для загрузки данных
#include "capon_processor.hpp"
#include "capon_test_helpers.hpp"

// Обёртки GPUWorkLib над OpenCL и ROCm backend'ами
#include "backends/opencl/opencl_backend.hpp"
#include "backends/rocm/rocm_backend.hpp"
#include "services/console_output.hpp"

// Нативные OpenCL и HIP заголовки — нужны для прямых вызовов API
#include <CL/cl.h>
#include <hip/hip_runtime.h>

// Стандартная библиотека
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

namespace test_capon_hip_opencl_to_rocm {

// cx = complex<float> — комплексное число с плавающей точкой (8 байт)
using cx = std::complex<float>;
using namespace drv_gpu_lib;
using namespace capon_test_helpers;

// ========================================================================
// Утилита вывода — все сообщения через ConsoleOutput, а не printf/cout!
// (На 10 GPU одновременно printf смешивает строки из разных потоков)
// ========================================================================

inline void TestPrint(const std::string& msg) {
  ConsoleOutput::GetInstance().Print(0, "Capon[HIP->OCL->ROCm]", msg);
}

// ========================================================================
// Backend-синглтоны — создаём один раз, переиспользуем во всех тестах
//
// OpenCLBackend  — обёртка над голым OpenCL API (cl_context, cl_queue...)
// ROCmBackend    — обёртка над HIP/ROCm (hipStream, rocblas_handle...)
// ========================================================================

inline OpenCLBackend& GetClBackend() {
  static OpenCLBackend cl;       // создаётся один раз (static local)
  static bool inited = false;
  if (!inited) {
    cl.Initialize(0);            // инициализировать на GPU device 0
    inited = true;
  }
  return cl;
}

inline ROCmBackend& GetRocmBackend() {
  return capon_test_helpers::GetROCmBackend();
}

// ========================================================================
// Проверка ошибок HIP и OpenCL
//
// Вместо того чтобы каждый раз писать:
//   hipError_t e = hipMalloc(...);
//   if (e != hipSuccess) { ... длинный код обработки ... }
//
// Пишем просто:
//   HipOk(hipMalloc(...), "hipMalloc Y");
//
// Если ошибка — бросает исключение с понятным сообщением.
// Исключение поймается в try/catch в тесте → TestPrint("[EXCEPTION: ...]")
// ========================================================================

inline void HipOk(hipError_t e, const char* msg) {
  if (e != hipSuccess)
    throw std::runtime_error(std::string(msg) + ": " + hipGetErrorString(e));
}

inline void ClOk(cl_int e, const char* msg) {
  if (e != CL_SUCCESS) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), ": cl_err=%d", e);
    throw std::runtime_error(std::string(msg) + buf);
  }
}

// ========================================================================
//
//   Тест 01: detect_hip_svm
//   ─────────────────────────────────────────────────────────────────────
//   Просто ПОКАЗЫВАЕМ что умеет наш GPU в плане HIP + OpenCL interop.
//   Никаких данных не обрабатываем — только читаем свойства устройства.
//
//   Что выводим:
//     - Название GPU, архитектура (gfx1201 = RDNA4), объём VRAM
//     - Версия OpenCL
//     - Уровни SVM: coarse / fine / system / atomics
//     - Что из этого нужно для pipeline и почему
//
// ========================================================================

inline void test_01_detect_hip_svm() {
  TestPrint("[01] detect_hip_svm -- HIP + OpenCL SVM interop capabilities");

  // ── Информация о GPU через HIP API ───────────────────────────────────
  // hipDeviceProp_t — структура со всеми свойствами GPU
  hipDeviceProp_t props{};
  HipOk(hipGetDeviceProperties(&props, 0), "hipGetDeviceProperties");

  {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "  HIP: %s  arch: %s  VRAM: %zu MB",
        props.name,                               // "AMD Radeon RX 9070"
        props.gcnArchName,                        // "gfx1201" — код архитектуры
        props.totalGlobalMem / (1024 * 1024));    // в мегабайтах
    TestPrint(buf);
  }

  // ── Информация об OpenCL-устройстве ──────────────────────────────────
  // GetNativeDevice() возвращает void* который на самом деле cl_device_id.
  // static_cast — явное приведение типов, говорим компилятору "мы знаем что делаем".
  auto& cl    = GetClBackend();
  cl_device_id cl_dev = static_cast<cl_device_id>(cl.GetNativeDevice());

  char dev_name[256] = {};
  char dev_ver[256]  = {};
  // clGetDeviceInfo — запрашиваем конкретное свойство устройства
  clGetDeviceInfo(cl_dev, CL_DEVICE_NAME,    sizeof(dev_name), dev_name, nullptr);
  clGetDeviceInfo(cl_dev, CL_DEVICE_VERSION, sizeof(dev_ver),  dev_ver,  nullptr);
  TestPrint(std::string("  OpenCL: ") + dev_name + "  " + dev_ver);

  // ── SVM Capabilities — что поддерживает GPU ──────────────────────────
  // svm_caps — битовое поле: каждый бит = одна возможность
  cl_device_svm_capabilities svm_caps = 0;
  clGetDeviceInfo(cl_dev, CL_DEVICE_SVM_CAPABILITIES,
                  sizeof(svm_caps), &svm_caps, nullptr);

  // Проверяем каждый бит через маску (&)
  const bool coarse = (svm_caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) != 0;
  const bool fine   = (svm_caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER)   != 0;
  const bool system = (svm_caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM)   != 0;
  const bool atomic = (svm_caps & CL_DEVICE_SVM_ATOMICS)             != 0;

  {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "  SVM coarse: %-4s | fine: %-4s | system: %-4s | atomics: %-4s",
        coarse ? "YES" : "NO",
        fine   ? "YES" : "NO",
        system ? "YES" : "NO",
        atomic ? "YES" : "NO");
    TestPrint(buf);
  }

  // ── Что нужно для нашего pipeline ────────────────────────────────────
  //
  // coarse-grain SVM:
  //   Позволяет передать hipMalloc-указатель в OpenCL через
  //   clSetKernelArgSVMPointer. Именно это делает HIPmemTest.
  //   Минимальное требование для HIP ↔ OpenCL interop.
  //
  // fine-grain system SVM:
  //   Позволяет использовать ОБЫЧНУЮ host-память (std::vector::data(),
  //   malloc, new) как src в clEnqueueSVMMemcpy.
  //   Нужно нам для: clEnqueueSVMMemcpy(d_hip_Y, h_signal.data(), bytes)

  if (!coarse)
    TestPrint("  [!] coarse-grain: НЕТ — hipMalloc→OpenCL невозможен на этом GPU");
  else
    TestPrint("  [+] coarse-grain: ДА  — hipMalloc ptr валиден как SVM (как HIPmemTest)");

  if (!system)
    TestPrint("  [!] system SVM:   НЕТ — clEnqueueSVMMemcpy(host_ptr) может не работать");
  else
    TestPrint("  [+] system SVM:   ДА  — std::vector::data() валиден как SVM src");

  // Итоговое объяснение принципа
  TestPrint("  Принцип: hipMalloc → hsa_amd_memory_pool_allocate → VRAM");
  TestPrint("           OpenCL видит тот же адрес как SVM ptr (единое HSA VA)");

  if (coarse)
    TestPrint("[01] PASS -- HIP+OpenCL SVM interop доступен");
  else
    TestPrint("[01] SKIP -- SVM coarse-grain недоступен");
}

// ========================================================================
//
//   Тест 02: ПОЛНЫЙ PIPELINE С ДАННЫМИ ЗАКАЗЧИКА
//   ─────────────────────────────────────────────────────────────────────
//
//   Это ГЛАВНЫЙ тест — демонстрация для заказчика.
//   Полный путь от файлов на диске до результата Capon на GPU.
//
//   ЭТАП 1: CPU загружает данные из файлов в std::vector (обычная RAM)
//   ЭТАП 2: HIP выделяет VRAM через hipMalloc, обнуляет hipMemset
//   ЭТАП 3: OpenCL пишет данные в hipMalloc-VRAM через clEnqueueSVMMemcpy
//   ЭТАП 4: CaponProcessor считает рельеф, используя HIP-указатели напрямую
//
//   Почему hipMemset(0) важен:
//     Мы обнуляем буферы сразу после hipMalloc, ПЕРЕД тем как OpenCL пишет.
//     Если в результате есть ненулевые данные — значит OpenCL ДЕЙСТВИТЕЛЬНО
//     записал их (а не остался мусор от предыдущего использования).
//
//   Почему clFinish() критичен:
//     clEnqueueSVMMemcpy — АСИНХРОННАЯ операция. Она ставится в очередь
//     OpenCL и выполняется в фоне. Без clFinish() Capon может начать
//     читать ещё незаписанные данные и посчитает мусор.
//     clFinish() = "жди пока все операции в очереди завершатся".
//
// ========================================================================

inline void test_02_hip_opencl_capon_pipeline() {
  TestPrint("[02] ===== ПОЛНЫЙ PIPELINE: hipMalloc → clEnqueueSVMMemcpy → Capon =====");

  auto& cl   = GetClBackend();
  auto& rocm = GetRocmBackend();

  // Получаем "нативные" OpenCL объекты из обёртки GPUWorkLib.
  // GetNativeDevice()  → cl_device_id  (идентификатор GPU в OpenCL)
  // GetNativeQueue()   → cl_command_queue (очередь команд OpenCL)
  cl_device_id     cl_dev   = static_cast<cl_device_id>(cl.GetNativeDevice());
  cl_command_queue cl_queue = static_cast<cl_command_queue>(cl.GetNativeQueue());

  // Проверяем минимальное требование: coarse-grain SVM
  // (без него hipMalloc-указатели не работают в OpenCL)
  cl_device_svm_capabilities svm_caps = 0;
  clGetDeviceInfo(cl_dev, CL_DEVICE_SVM_CAPABILITIES,
                  sizeof(svm_caps), &svm_caps, nullptr);
  if (!(svm_caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)) {
    TestPrint("[02] SKIP -- coarse-grain SVM недоступен");
    return;
  }

  // ╔══════════════════════════════════════════════════════════════════════╗
  // ║   ЭТАП 1: ЗАГРУЗКА ДАННЫХ ЗАКАЗЧИКА В RAM (CPU)                      ║
  // ║                                                                      ║
  // ║   Данные живут на диске в текстовых файлах MATLAB-формата.           ║
  // ║   Загружаем в std::vector<> — обычная оперативная память.            ║
  // ║   GPU эти данные пока НЕ видит — они только на CPU.                 ║
  // ╚══════════════════════════════════════════════════════════════════════╝

  TestPrint("  ---[ ЭТАП 1: ЗАГРУЗКА ДАННЫХ ЗАКАЗЧИКА ]---");

  // LoadRealVector — утилита из capon_test_helpers, читает столбец float из файла
  std::vector<float> x_all, y_all;
  if (!LoadRealVector(kDataDir + "x_data.txt", x_all) ||
      !LoadRealVector(kDataDir + "y_data.txt", y_all)) {
    TestPrint("[02] SKIP -- данные заказчика не найдены (x_data.txt / y_data.txt)");
    return;
  }

  // Физические параметры антенной решётки заказчика:
  //   P = 85  — число антенных каналов в рабочем подмассиве (из 340 всего)
  //   N = 1000 — число временных отсчётов сигнала
  //   M = ~37  — число направлений сканирования (вычисляется из сетки)
  const uint32_t P = 85;
  const uint32_t N = 1000;

  if (x_all.size() < P || y_all.size() < P) {
    TestPrint("[02] SKIP -- недостаточно антенных элементов в файлах");
    return;
  }

  // Берём первые P элементов (рабочий подмассив антенной решётки)
  std::vector<float> x_sub(x_all.begin(), x_all.begin() + P);
  std::vector<float> y_sub(y_all.begin(), y_all.begin() + P);

  // LoadSignalMatlab — читает матрицу [P × N] комплексных чисел из MATLAB-файла
  // signal — вектор размера P*N = 85*1000 = 85000 комплексных float
  std::vector<cx> signal;
  if (!LoadSignalMatlab(kDataDir + "signal_matlab.txt", P, N, signal)) {
    TestPrint("[02] SKIP -- signal_matlab.txt не найден");
    return;
  }

  // MakeScanGrid1D — создаёт равномерную сетку направлений в пределах ±3.25°
  // MakePhysicalSteering — вычисляет управляющие векторы e^(2πi/λ * r·u)
  // steering — вектор размера P*M комплексных float
  auto u0           = MakeScanGrid1D(3.25, 0.00312);
  const uint32_t M  = static_cast<uint32_t>(u0.size());
  std::vector<float> v0(M, 0.0f);
  auto steering     = MakePhysicalSteering(x_sub, y_sub, u0, v0, kF0, kC);

  // Размеры данных в байтах
  // sizeof(cx) = 8 байт (два float по 4 байта — реальная и мнимая части)
  const size_t bytes_Y = signal.size()   * sizeof(cx);   // ~680 KB для P=85, N=1000
  const size_t bytes_U = steering.size() * sizeof(cx);   // ~20 KB для P=85, M=37

  {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "  P=%u каналов, N=%u отсчётов, M=%u направлений", P, N, M);
    TestPrint(buf);
    std::snprintf(buf, sizeof(buf),
        "  Y: %zu cf (%.1f KB)  U: %zu cf (%.1f KB)",
        signal.size(), bytes_Y / 1024.0, steering.size(), bytes_U / 1024.0);
    TestPrint(buf);
  }

  // GPU-указатели объявляем здесь чтобы hipFree работал и после исключения
  bool  ok      = false;
  void* d_hip_Y = nullptr;  // Y — матрица сигнала [P × N] в VRAM
  void* d_hip_U = nullptr;  // U — управляющие векторы [P × M] в VRAM

  try {

    // ╔══════════════════════════════════════════════════════════════════╗
    // ║   ЭТАП 2: hipMalloc — HIP выделяет VRAM                         ║
    // ║                                                                  ║
    // ║   hipMalloc — аналог malloc(), но выделяет память не в RAM,     ║
    // ║   а в VRAM (видеопамяти GPU). Возвращает указатель в VRAM.       ║
    // ║                                                                  ║
    // ║   КЛЮЧЕВОЙ МОМЕНТ:                                               ║
    // ║   Этот указатель живёт в HSA-адресном пространстве.              ║
    // ║   OpenCL знает про HSA — значит тот же адрес валиден для OCL!   ║
    // ║                                                                  ║
    // ║   hipMemset(ptr, 0, bytes) — заполняет GPU-память нулями.       ║
    // ║   Как memset(), но GPU-async. После hipDeviceSynchronize() —    ║
    // ║   гарантированно выполнено.                                      ║
    // ╚══════════════════════════════════════════════════════════════════╝

    TestPrint("  ---[ ЭТАП 2: hipMalloc — HIP выделяет VRAM ]---");

    HipOk(hipMalloc(&d_hip_Y, bytes_Y), "hipMalloc Y");  // выделяем VRAM для сигнала
    HipOk(hipMalloc(&d_hip_U, bytes_U), "hipMalloc U");  // выделяем VRAM для steering

    // Обнуляем — чтобы доказать: именно OpenCL запишет данные, а не мусор
    HipOk(hipMemset(d_hip_Y, 0, bytes_Y), "hipMemset Y");
    HipOk(hipMemset(d_hip_U, 0, bytes_U), "hipMemset U");
    // hipDeviceSynchronize() — ждём завершения всех асинхронных GPU-операций
    HipOk(hipDeviceSynchronize(), "hipDeviceSynchronize");

    {
      char buf[256];
      // Выводим адреса в VRAM — доказательство что память реально выделена
      std::snprintf(buf, sizeof(buf),
          "  d_hip_Y = %p  (VRAM, обнулён hipMemset)", d_hip_Y);
      TestPrint(buf);
      std::snprintf(buf, sizeof(buf),
          "  d_hip_U = %p  (VRAM, обнулён hipMemset)", d_hip_U);
      TestPrint(buf);
    }

    // ╔══════════════════════════════════════════════════════════════════╗
    // ║   ЭТАП 3: OpenCL ПИШЕТ ДАННЫЕ В hipMalloc-VRAM                  ║
    // ║                                                                  ║
    // ║   clEnqueueSVMMemcpy — OpenCL-версия memcpy для SVM-памяти.     ║
    // ║                                                                  ║
    // ║   dst = d_hip_Y — hipMalloc VRAM, валидный coarse-grain SVM ptr ║
    // ║   src = pinned_Y — hipHostMalloc пинованная память              ║
    // ║                                                                  ║
    // ║   ПОЧЕМУ hipHostMalloc, а не std::vector::data()?               ║
    // ║                                                                  ║
    // ║   std::vector выделяет обычную RAM (система показала:            ║
    // ║   CL_DEVICE_SVM_FINE_GRAIN_SYSTEM = NO на gfx1201).             ║
    // ║   Это значит: обычные host-указатели — НЕ валидные SVM ptr'ы.   ║
    // ║                                                                  ║
    // ║   hipHostMalloc — пинованная (непагинируемая) память в HSA VA.  ║
    // ║   Она доступна как GPU через DMA и как OpenCL SVM ptr.          ║
    // ║   CPU пишет в неё через обычный memcpy.                         ║
    // ║   OpenCL читает её через clEnqueueSVMMemcpy → пишет в VRAM.     ║
    // ║                                                                  ║
    // ║   Для 4 ГБ: hipHostMalloc выделяет ОДИН раз, передача DMA.     ║
    // ║   NO clSVMAlloc, NO GPU-side staging, NO copy-kernel.           ║
    // ║                                                                  ║
    // ║   Путь данных:                                                   ║
    // ║     CPU RAM → memcpy → pinned_Y (hipHostMalloc)                  ║
    // ║     pinned_Y → clEnqueueSVMMemcpy → d_hip_Y (VRAM)              ║
    // ║     ← всё это выглядит как один DMA трансфер для GPU            ║
    // ╚══════════════════════════════════════════════════════════════════╝

    TestPrint("  ---[ ЭТАП 3: OpenCL пишет в hipMalloc VRAM (clEnqueueSVMMemcpy) ]---");
    TestPrint("  src=hipHostMalloc (пинованная HSA память), dst=d_hip_Y (VRAM)");
    TestPrint("  NO clSVMAlloc, NO GPU staging, NO copy-kernel");

    // hipHostMalloc — выделяем пинованную host-память в HSA VA
    // Она доступна как SVM src для OpenCL (в отличие от обычного std::vector)
    void* pinned_Y = nullptr;
    void* pinned_U = nullptr;
    HipOk(hipHostMalloc(&pinned_Y, bytes_Y, hipHostMallocDefault), "hipHostMalloc Y");
    HipOk(hipHostMalloc(&pinned_U, bytes_U, hipHostMallocDefault), "hipHostMalloc U");

    // CPU копирует данные заказчика в пинованную память (обычный memcpy)
    std::memcpy(pinned_Y, signal.data(),   bytes_Y);
    std::memcpy(pinned_U, steering.data(), bytes_U);
    TestPrint("  memcpy CPU→pinned OK (данные заказчика в HSA-адресном пространстве)");

    // clEnqueueSVMMemcpy: pinned (SVM src) → d_hip_Y (SVM dst, hipMalloc VRAM)
    // Оба указателя живут в HSA VA — OpenCL инициирует DMA-трансфер
    ClOk(clEnqueueSVMMemcpy(
        cl_queue,   // очередь OpenCL
        CL_FALSE,   // асинхронно — ставим в очередь, не ждём
        d_hip_Y,    // dst: hipMalloc VRAM (coarse-grain SVM)
        pinned_Y,   // src: hipHostMalloc пинованная (HSA SVM)
        bytes_Y,
        0, nullptr, nullptr), "clEnqueueSVMMemcpy Y");

    ClOk(clEnqueueSVMMemcpy(
        cl_queue,
        CL_FALSE,
        d_hip_U,
        pinned_U,
        bytes_U,
        0, nullptr, nullptr), "clEnqueueSVMMemcpy U");

    // Ждём завершения ОБОИХ DMA-трансфертов перед Capon
    ClOk(clFinish(cl_queue), "clFinish");
    TestPrint("  clFinish() — оба DMA завершены, данные в VRAM");

    // Пинованная память больше не нужна — данные уже в d_hip_Y/U
    hipHostFree(pinned_Y);
    hipHostFree(pinned_U);
    pinned_Y = nullptr;
    pinned_U = nullptr;

    {
      char buf[256];
      std::snprintf(buf, sizeof(buf),
          "  d_hip_Y=%p  d_hip_U=%p  (заполнены OpenCL, zero copy OCL↔HIP)",
          d_hip_Y, d_hip_U);
      TestPrint(buf);
    }

    // ╔══════════════════════════════════════════════════════════════════╗
    // ║   ЭТАП 4: РАСЧЁТ КЕЙПОНА НА ROCm                                ║
    // ║                                                                  ║
    // ║   CaponProcessor::ComputeRelief(void* Y, void* U, params)       ║
    // ║                                                                  ║
    // ║   Принимает void* — просто адрес в VRAM (не важно кто выделил). ║
    // ║   Передаём d_hip_Y и d_hip_U — те же указатели от hipMalloc.   ║
    // ║                                                                  ║
    // ║   ZeroCopyBridge НЕ НУЖЕН потому что:                           ║
    // ║     - В старом подходе: cl_mem ──[мост]──► void* hip_ptr        ║
    // ║       Мост нужен чтобы "перевести" OpenCL-адрес в HIP-адрес.    ║
    // ║     - Здесь: hipMalloc уже дал нам HIP-адрес напрямую!          ║
    // ║       d_hip_Y IS a HIP pointer. Никакого перевода не нужно.     ║
    // ║                                                                  ║
    // ║   GPU pipeline Кейпона:                                          ║
    // ║     1. R = (1/N)*Y*Y^H + μI   (rocBLAS CGEMM)                   ║
    // ║     2. R^{-1}                 (rocSOLVER POTRF + POTRI)          ║
    // ║     3. W = R^{-1} * U         (rocBLAS CGEMM)                   ║
    // ║     4. z[m] = 1/Re(u_m^H*W)  (HIP kernel)                       ║
    // ╚══════════════════════════════════════════════════════════════════╝

    TestPrint("  ---[ ЭТАП 4: РАСЧЁТ КЕЙПОНА НА ROCm (без ZeroCopyBridge!) ]---");

    capon::CaponParams params;
    params.n_channels   = P;    // размер антенного подмассива
    params.n_samples    = N;    // длина временного ряда
    params.n_directions = M;    // число направлений сканирования
    params.mu           = 1.0f; // коэффициент регуляризации (стабилизирует инверсию)

    capon::CaponProcessor processor(&rocm);
    // Передаём НАПРЯМУЮ d_hip_Y и d_hip_U — указатели от hipMalloc
    auto result = processor.ComputeRelief(d_hip_Y, d_hip_U, params);

    assert(result.relief.size() == M);

    // Проверяем физический смысл: рельеф Кейпона ДОЛЖЕН быть > 0 и конечным
    float z_min  = result.relief[0];
    float z_max  = result.relief[0];
    bool  all_ok = true;
    for (uint32_t m = 0; m < M; ++m) {
      if (!std::isfinite(result.relief[m]) || result.relief[m] <= 0.0f)
        all_ok = false;  // NaN или Inf или отрицательное → алгоритм сломан
      if (result.relief[m] < z_min) z_min = result.relief[m];
      if (result.relief[m] > z_max) z_max = result.relief[m];
    }
    assert(all_ok && "Capon relief must be > 0 and finite");

    {
      char buf[256];
      // ratio = z_max/z_min — динамический диапазон, показывает чёткость пика
      std::snprintf(buf, sizeof(buf),
          "  Capon relief [M=%u]: min=%.4g  max=%.4g  ratio=%.1f",
          M, z_min, z_max, z_max / (z_min + 1e-30f));
      TestPrint(buf);
    }
    TestPrint("  Все z[m] > 0 и конечны — GPU pipeline OK");

    ok = true;

  } catch (const std::exception& e) {
    // Поймали исключение из HipOk() или ClOk() — выводим и продолжаем
    TestPrint(std::string("  EXCEPTION: ") + e.what());
  }

  // Освобождаем VRAM — даже если тест упал (cleanup после catch)
  // hipFree на nullptr безопасен (ничего не делает)
  if (d_hip_Y) hipFree(d_hip_Y);
  if (d_hip_U) hipFree(d_hip_U);

  // assert(ok) — если тест упал с исключением, тут программа завершится
  assert(ok);
  TestPrint("[02] PASS -- hipMalloc → clEnqueueSVMMemcpy → Capon: pipeline успешен");
}

// ========================================================================
//
//   Тест 03: СРАВНЕНИЕ ДВУХ ПУТЕЙ
//   ─────────────────────────────────────────────────────────────────────
//   Доказываем: новый pipeline (hipMalloc + OpenCL write) даёт ТОЧНО
//   такой же результат как прямой вызов (CPU vector → Capon).
//
//   Путь A (прямой, референс):
//     CaponProcessor.ComputeRelief(signal, steering, params)
//     Capon сам копирует данные с CPU внутри метода (hipMemcpy).
//     Это проверенный путь — используем как эталон.
//
//   Путь B (новый, тестируемый):
//     hipMalloc → hipMemset(0) → clEnqueueSVMMemcpy → clFinish
//     CaponProcessor.ComputeRelief(d_hip_Y, d_hip_U, params)
//
//   Если max|z_A[m] - z_B[m]| < 1e-4 для всех m → тест PASS.
//   Это значит: данные дошли до Capon без искажений через новый путь.
//
//   Используем подмножество данных (P=16, N=128) для скорости.
//
// ========================================================================

inline void test_03_hip_opencl_matches_direct() {
  TestPrint("[03] hip_opencl_matches_direct -- hipMalloc→clSVMMemcpy path == direct");

  auto& cl   = GetClBackend();
  auto& rocm = GetRocmBackend();

  cl_device_id     cl_dev   = static_cast<cl_device_id>(cl.GetNativeDevice());
  cl_command_queue cl_queue = static_cast<cl_command_queue>(cl.GetNativeQueue());

  cl_device_svm_capabilities svm_caps = 0;
  clGetDeviceInfo(cl_dev, CL_DEVICE_SVM_CAPABILITIES,
                  sizeof(svm_caps), &svm_caps, nullptr);
  if (!(svm_caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)) {
    TestPrint("[03] SKIP -- coarse-grain SVM недоступен");
    return;
  }

  // Загружаем данные заказчика (подмножество — для скорости теста)
  std::vector<float> x_all, y_all;
  if (!LoadRealVector(kDataDir + "x_data.txt", x_all) ||
      !LoadRealVector(kDataDir + "y_data.txt", y_all)) {
    TestPrint("[03] SKIP -- данные заказчика не найдены");
    return;
  }

  // P=16, N=128 — маленькое подмножество, тест проходит быстро
  // (полный P=85, N=1000 уже проверен в test_02)
  const uint32_t P = 16;
  const uint32_t N = 128;

  if (x_all.size() < P || y_all.size() < P) {
    TestPrint("[03] SKIP -- not enough antenna elements");
    return;
  }

  std::vector<float> x_sub(x_all.begin(), x_all.begin() + P);
  std::vector<float> y_sub(y_all.begin(), y_all.begin() + P);

  std::vector<cx> signal;
  if (!LoadSignalMatlab(kDataDir + "signal_matlab.txt", P, N, signal)) {
    TestPrint("[03] SKIP -- signal_matlab.txt не найден");
    return;
  }

  // M=16 направлений, равномерная сетка ±3.25°
  const uint32_t M    = 16;
  const double   ulim = std::sin(3.25 * M_PI / 180.0);  // синус угла в радианах
  std::vector<float> u_dirs(M), v_dirs(M, 0.0f);
  for (uint32_t m = 0; m < M; ++m)
    u_dirs[m] = static_cast<float>(-ulim + 2.0 * ulim * m / (M - 1));
  auto steering = MakePhysicalSteering(x_sub, y_sub, u_dirs, v_dirs, kF0, kC);

  const size_t bytes_Y = signal.size()   * sizeof(cx);
  const size_t bytes_U = steering.size() * sizeof(cx);

  capon::CaponParams params{P, N, M, 1.0f};

  // ── Путь A: ПРЯМОЙ (референс) ─────────────────────────────────────────
  // ComputeRelief(vector, vector, params) — перегрузка с CPU-данными.
  // Внутри Capon сам делает hipMemcpy с CPU в VRAM.
  capon::CaponProcessor proc_ref(&rocm);
  auto relief_ref = proc_ref.ComputeRelief(signal, steering, params);

  // ── Путь B: hipMalloc → clEnqueueSVMMemcpy → Capon ───────────────────
  capon::CaponReliefResult relief_hip;  // результат нового пути
  bool  ok      = false;
  void* d_hip_Y = nullptr;
  void* d_hip_U = nullptr;

  try {
    // Выделяем HIP-память, обнуляем
    HipOk(hipMalloc(&d_hip_Y, bytes_Y), "hipMalloc Y");
    HipOk(hipMalloc(&d_hip_U, bytes_U), "hipMalloc U");
    HipOk(hipMemset(d_hip_Y, 0, bytes_Y), "hipMemset Y");
    HipOk(hipMemset(d_hip_U, 0, bytes_U), "hipMemset U");
    HipOk(hipDeviceSynchronize(), "hipDeviceSynchronize");

    // OpenCL пишет данные в HIP-память (тот же подход что в тесте 02).
    //
    // ВАЖНО: clEnqueueSVMMemcpy требует SVM-указатель в src!
    // std::vector::data() — обычная host-память (не SVM на gfx1201).
    // Решение: hipHostMalloc → пинованная память в HSA VA → валидный SVM src.
    void* pinned_Y = nullptr;
    void* pinned_U = nullptr;
    HipOk(hipHostMalloc(&pinned_Y, bytes_Y, hipHostMallocDefault), "hipHostMalloc Y");
    HipOk(hipHostMalloc(&pinned_U, bytes_U, hipHostMallocDefault), "hipHostMalloc U");
    std::memcpy(pinned_Y, signal.data(),   bytes_Y);
    std::memcpy(pinned_U, steering.data(), bytes_U);

    ClOk(clEnqueueSVMMemcpy(cl_queue, CL_FALSE,
        d_hip_Y, pinned_Y, bytes_Y,
        0, nullptr, nullptr), "clEnqueueSVMMemcpy Y");

    ClOk(clEnqueueSVMMemcpy(cl_queue, CL_FALSE,
        d_hip_U, pinned_U, bytes_U,
        0, nullptr, nullptr), "clEnqueueSVMMemcpy U");

    // Ждём завершения DMA-трансфертов — обязательно перед Capon!
    ClOk(clFinish(cl_queue), "clFinish");

    // Пинованная память больше не нужна
    hipHostFree(pinned_Y);
    hipHostFree(pinned_U);

    // Capon с HIP-указателями напрямую
    capon::CaponProcessor proc_hip(&rocm);
    relief_hip = proc_hip.ComputeRelief(d_hip_Y, d_hip_U, params);

    ok = true;
  } catch (const std::exception& e) {
    TestPrint(std::string("  EXCEPTION: ") + e.what());
  }

  // Cleanup
  if (d_hip_Y) hipFree(d_hip_Y);
  if (d_hip_U) hipFree(d_hip_U);

  assert(ok);
  assert(relief_hip.relief.size() == M);

  // ── Сравнение результатов ─────────────────────────────────────────────
  // Ищем максимальное абсолютное и относительное отклонение
  float max_diff    = 0.0f;  // max|z_A[m] - z_B[m]|
  float max_reldiff = 0.0f;  // max|z_A[m] - z_B[m]| / |z_A[m]|

  for (uint32_t m = 0; m < M; ++m) {
    float diff    = std::fabs(relief_ref.relief[m] - relief_hip.relief[m]);
    float reldiff = diff / (std::fabs(relief_ref.relief[m]) + 1e-30f);  // +eps от деления на 0
    if (diff    > max_diff)    max_diff    = diff;
    if (reldiff > max_reldiff) max_reldiff = reldiff;
  }

  {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "  max |z_direct - z_hip_ocl| = %.3e   max_rel = %.3e   (tolerance < 1e-4)",
        max_diff, max_reldiff);
    TestPrint(buf);
  }

  // Допуск 1e-4: разница между float32 операциями на разных путях
  // (hipMemcpy vs clEnqueueSVMMemcpy) не должна влиять на результат Capon
  assert(max_diff < 1e-4f &&
         "hipMalloc→clSVMMemcpy path: data mismatch vs direct path");
  TestPrint("[03] PASS -- hipMalloc→clSVMMemcpy→Capon идентичен прямому пути");
}

// ========================================================================
// run() — точка входа, вызывается из all_test.hpp
//
// Запускает все три теста последовательно.
// ConsoleOutput::GetInstance().Start() — инициализирует мультиGPU-вывод.
// ========================================================================

inline void run() {
  ConsoleOutput::GetInstance().Start();
  TestPrint("========================================================");
  TestPrint("  test_capon_hip_opencl_to_rocm");
  TestPrint("  HIP выделяет VRAM, OpenCL пишет данные, Capon считает");
  TestPrint("  NO clSVMAlloc  NO staging  NO ZeroCopyBridge");
  TestPrint("========================================================");

  test_01_detect_hip_svm();
  test_02_hip_opencl_capon_pipeline();
  test_03_hip_opencl_matches_direct();

  TestPrint("=== test_capon_hip_opencl_to_rocm DONE ===");
}

}  // namespace test_capon_hip_opencl_to_rocm

#endif  // ENABLE_ROCM
