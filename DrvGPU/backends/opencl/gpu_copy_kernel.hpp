#pragma once

/**
 * @file gpu_copy_kernel.hpp
 * @brief OpenCL kernel для VRAM→VRAM копии: cl_mem → coarse-grain SVM
 *
 * Данные НЕ покидают GPU. Копия внутри VRAM через PCIe не идёт.
 * ~8-15мс для 4ГБ (зависит от bandwidth GPU).
 *
 * Оптимизация: uint4 (16 байт на work-item) для максимальной пропускной
 * способности. Остаток обрабатывается побайтово.
 *
 * Кеш: cl_program компилируется один раз per cl_context (static map + mutex).
 * Первый вызов ~1мс (JIT), последующие ~0 (lookup).
 *
 * Использование:
 * @code
 * void* svm = clSVMAlloc(ctx, CL_MEM_READ_WRITE, size, 0);  // coarse-grain VRAM
 * GpuCopyClMemToSVM(queue, ctx, cl_buffer, svm, size);
 * // svm содержит копию cl_buffer, данные в VRAM
 * @endcode
 *
 * @note Linux only, требует OpenCL 2.0+ SVM coarse-grain support
 * @author Кодо (AI Assistant)
 * @date 2026-03-24
 */

#include <CL/cl.h>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Kernel source (inline — не зависит от файловой системы)
// ════════════════════════════════════════════════════════════════════════════

static constexpr const char* kGpuCopyKernelSource = R"CL(
// Wide copy: 16 байт (uint4) за один work-item → максимальная пропускная способность
__kernel void copy_wide(
    __global const uint4* restrict src,
    __global uint4* restrict dst,
    const uint n_uint4)
{
  uint i = get_global_id(0);
  if (i < n_uint4) dst[i] = src[i];
}

// Tail copy: остаток байт (0..15) после wide copy
__kernel void copy_tail(
    __global const uchar* restrict src,
    __global uchar* restrict dst,
    const uint tail_offset,
    const uint n_tail)
{
  uint i = get_global_id(0);
  if (i < n_tail) dst[tail_offset + i] = src[tail_offset + i];
}
)CL";

// ════════════════════════════════════════════════════════════════════════════
// Кеш скомпилированных программ (per cl_context, thread-safe)
// ════════════════════════════════════════════════════════════════════════════

struct GpuCopyKernels {
  cl_program program = nullptr;
  cl_kernel  k_wide  = nullptr;
  cl_kernel  k_tail  = nullptr;
};

/**
 * @brief Получить или скомпилировать copy kernels для данного context.
 *
 * Первый вызов: clBuildProgram (~1мс JIT).
 * Последующие: O(1) lookup в static unordered_map.
 * Thread-safe через static mutex.
 *
 * @param ctx OpenCL context
 * @return указатель на закешированные kernels, nullptr при ошибке
 *
 * @note Кеш живёт до завершения процесса. OpenCL runtime освобождает
 *       ресурсы при exit — явный cleanup не требуется.
 */
inline GpuCopyKernels* GetOrCompileCopyKernels(cl_context ctx) {
  static std::mutex cache_mutex;
  static std::unordered_map<cl_context, GpuCopyKernels> cache;

  std::lock_guard<std::mutex> lock(cache_mutex);

  auto it = cache.find(ctx);
  if (it != cache.end()) return &it->second;

  // Первая компиляция для этого context
  cl_int err;
  const char* src_ptr = kGpuCopyKernelSource;
  cl_program program = clCreateProgramWithSource(ctx, 1, &src_ptr, nullptr, &err);
  if (err != CL_SUCCESS || !program) return nullptr;

  err = clBuildProgram(program, 0, nullptr, "-cl-std=CL2.0", nullptr, nullptr);
  if (err != CL_SUCCESS) {
    clReleaseProgram(program);
    return nullptr;
  }

  cl_kernel k_wide = clCreateKernel(program, "copy_wide", &err);
  if (err != CL_SUCCESS) {
    clReleaseProgram(program);
    return nullptr;
  }

  cl_kernel k_tail = clCreateKernel(program, "copy_tail", &err);
  if (err != CL_SUCCESS) {
    clReleaseKernel(k_wide);
    clReleaseProgram(program);
    return nullptr;
  }

  auto& entry = cache[ctx];
  entry.program = program;
  entry.k_wide  = k_wide;
  entry.k_tail  = k_tail;
  return &entry;
}

// ════════════════════════════════════════════════════════════════════════════
// GpuCopyClMemToSVM — основная функция
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Копировать cl_mem → coarse-grain SVM через OpenCL kernel (VRAM→VRAM)
 *
 * @param queue   OpenCL command queue (должен быть на том же device что и cl_mem)
 * @param ctx     OpenCL context
 * @param src     cl_mem source buffer
 * @param svm_dst coarse-grain SVM pointer (destination, аллоцирован через clSVMAlloc)
 * @param size_bytes Размер копии в байтах
 * @return true если копия успешна, false при ошибке OpenCL
 */
inline bool GpuCopyClMemToSVM(
    cl_command_queue queue,
    cl_context ctx,
    cl_mem src,
    void* svm_dst,
    size_t size_bytes) {

  if (!queue || !ctx || !src || !svm_dst || size_bytes == 0) return false;

  // ── Закешированная компиляция (один раз per context) ───────────────────
  GpuCopyKernels* kk = GetOrCompileCopyKernels(ctx);
  if (!kk) return false;

  cl_int err;

  // ── Wide copy: uint4 (16 байт) per work-item ──────────────────────────
  const uint32_t n_uint4 = static_cast<uint32_t>(size_bytes / 16);
  const uint32_t n_tail  = static_cast<uint32_t>(size_bytes % 16);
  bool ok = true;

  if (n_uint4 > 0) {
    err = clSetKernelArg(kk->k_wide, 0, sizeof(cl_mem), &src);
    if (err != CL_SUCCESS) return false;

    err = clSetKernelArgSVMPointer(kk->k_wide, 1, svm_dst);
    if (err != CL_SUCCESS) return false;

    err = clSetKernelArg(kk->k_wide, 2, sizeof(uint32_t), &n_uint4);
    if (err != CL_SUCCESS) return false;

    err = clSetKernelExecInfo(kk->k_wide, CL_KERNEL_EXEC_INFO_SVM_PTRS,
                               sizeof(void*), &svm_dst);
    if (err != CL_SUCCESS) return false;

    constexpr size_t kLocalSize = 256;
    size_t global_size = ((static_cast<size_t>(n_uint4) + kLocalSize - 1) / kLocalSize) * kLocalSize;

    err = clEnqueueNDRangeKernel(queue, kk->k_wide, 1, nullptr,
                                  &global_size, &kLocalSize, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) return false;
  }

  // ── Tail copy: остаток байт ────────────────────────────────────────────
  if (n_tail > 0) {
    uint32_t tail_offset = n_uint4 * 16;

    err = clSetKernelArg(kk->k_tail, 0, sizeof(cl_mem), &src);
    if (err != CL_SUCCESS) { ok = false; }

    if (ok) err = clSetKernelArgSVMPointer(kk->k_tail, 1, svm_dst);
    if (err != CL_SUCCESS) { ok = false; }

    if (ok) err = clSetKernelArg(kk->k_tail, 2, sizeof(uint32_t), &tail_offset);
    if (err != CL_SUCCESS) { ok = false; }

    if (ok) err = clSetKernelArg(kk->k_tail, 3, sizeof(uint32_t), &n_tail);
    if (err != CL_SUCCESS) { ok = false; }

    if (ok) err = clSetKernelExecInfo(kk->k_tail, CL_KERNEL_EXEC_INFO_SVM_PTRS,
                                       sizeof(void*), &svm_dst);
    if (err != CL_SUCCESS) { ok = false; }

    if (ok) {
      constexpr size_t kLocalSize = 256;
      size_t global_size = ((static_cast<size_t>(n_tail) + kLocalSize - 1) / kLocalSize) * kLocalSize;

      err = clEnqueueNDRangeKernel(queue, kk->k_tail, 1, nullptr,
                                    &global_size, &kLocalSize, 0, nullptr, nullptr);
      if (err != CL_SUCCESS) ok = false;
    }
  }

  // ── Синхронизация: данные должны быть скопированы до возврата ───────────
  if (ok) {
    err = clFinish(queue);
    if (err != CL_SUCCESS) ok = false;
  }

  return ok;
}

}  // namespace drv_gpu_lib
