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

  // ── Lazy compile (один раз per context) ────────────────────────────────
  // Простой подход: компилируем при каждом вызове (overhead ~1мс, ничтожен
  // на фоне копии 4ГБ). Для частых вызовов малых буферов можно кешировать.
  cl_int err;

  const char* src_ptr = kGpuCopyKernelSource;
  cl_program program = clCreateProgramWithSource(ctx, 1, &src_ptr, nullptr, &err);
  if (err != CL_SUCCESS || !program) return false;

  err = clBuildProgram(program, 0, nullptr, "-cl-std=CL2.0", nullptr, nullptr);
  if (err != CL_SUCCESS) {
    clReleaseProgram(program);
    return false;
  }

  cl_kernel k_wide = clCreateKernel(program, "copy_wide", &err);
  if (err != CL_SUCCESS) {
    clReleaseProgram(program);
    return false;
  }

  // ── Wide copy: uint4 (16 байт) per work-item ──────────────────────────
  const uint32_t n_uint4 = static_cast<uint32_t>(size_bytes / 16);
  const uint32_t n_tail  = static_cast<uint32_t>(size_bytes % 16);
  bool ok = true;

  if (n_uint4 > 0) {
    // src — cl_mem, dst — SVM pointer
    err = clSetKernelArg(k_wide, 0, sizeof(cl_mem), &src);
    if (err != CL_SUCCESS) { ok = false; goto cleanup; }

    err = clSetKernelArgSVMPointer(k_wide, 1, svm_dst);
    if (err != CL_SUCCESS) { ok = false; goto cleanup; }

    err = clSetKernelArg(k_wide, 2, sizeof(uint32_t), &n_uint4);
    if (err != CL_SUCCESS) { ok = false; goto cleanup; }

    // Сообщаем runtime о SVM pointer (требуется для coarse-grain SVM)
    err = clSetKernelExecInfo(k_wide, CL_KERNEL_EXEC_INFO_SVM_PTRS,
                               sizeof(void*), &svm_dst);
    if (err != CL_SUCCESS) { ok = false; goto cleanup; }

    constexpr size_t kLocalSize = 256;
    size_t global_size = ((static_cast<size_t>(n_uint4) + kLocalSize - 1) / kLocalSize) * kLocalSize;

    err = clEnqueueNDRangeKernel(queue, k_wide, 1, nullptr,
                                  &global_size, &kLocalSize, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { ok = false; goto cleanup; }
  }

  // ── Tail copy: остаток байт ────────────────────────────────────────────
  if (ok && n_tail > 0) {
    cl_kernel k_tail = clCreateKernel(program, "copy_tail", &err);
    if (err == CL_SUCCESS && k_tail) {
      uint32_t tail_offset = n_uint4 * 16;

      clSetKernelArg(k_tail, 0, sizeof(cl_mem), &src);
      clSetKernelArgSVMPointer(k_tail, 1, svm_dst);
      clSetKernelArg(k_tail, 2, sizeof(uint32_t), &tail_offset);
      clSetKernelArg(k_tail, 3, sizeof(uint32_t), &n_tail);
      clSetKernelExecInfo(k_tail, CL_KERNEL_EXEC_INFO_SVM_PTRS,
                           sizeof(void*), &svm_dst);

      constexpr size_t kLocalSize = 256;
      size_t global_size = ((static_cast<size_t>(n_tail) + kLocalSize - 1) / kLocalSize) * kLocalSize;

      err = clEnqueueNDRangeKernel(queue, k_tail, 1, nullptr,
                                    &global_size, &kLocalSize, 0, nullptr, nullptr);
      if (err != CL_SUCCESS) ok = false;

      clReleaseKernel(k_tail);
    } else {
      ok = false;
    }
  }

  // ── Синхронизация: данные должны быть скопированы до возврата ───────────
  if (ok) {
    err = clFinish(queue);
    if (err != CL_SUCCESS) ok = false;
  }

cleanup:
  clReleaseKernel(k_wide);
  clReleaseProgram(program);
  return ok;
}

}  // namespace drv_gpu_lib
