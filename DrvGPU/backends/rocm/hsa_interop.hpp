#pragma once

/**
 * @file hsa_interop.hpp
 * @brief HSA interop: извлечение GPU VA из cl_mem через HSA runtime
 *
 * На AMD ROCm и OpenCL и HIP используют HSA (ROCr) runtime.
 * cl_mem внутренне содержит HSA-аллокацию с GPU VA в unified address space.
 * Этот модуль извлекает GPU VA через hsa_amd_pointer_info probe.
 *
 * GPU VA из cl_mem доступен в HIP напрямую — TRUE zero-copy:
 * - 0 копий данных
 * - 0 дополнительной памяти
 * - ~микросекунды (только адресная арифметика)
 *
 * Алгоритм:
 * 1. Первый вызов: сканирование cl_mem объекта (каждые 8 байт, 0..2048)
 * 2. Для каждого void* значения: hsa_amd_pointer_info → ищем HSA type + совпадение размера
 * 3. Кешируем offset → последующие вызовы O(1)
 *
 * Проверено на:
 * - AMD Radeon RX 9070 (RDNA4, gfx1201), ROCm 7.2.0
 * - offset = +664 (может отличаться на других версиях ROCm)
 *
 * @note Linux + ENABLE_ROCM only
 * @author Кодо (AI Assistant)
 * @date 2026-03-24
 */

#if ENABLE_ROCM

#include <CL/cl.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <malloc.h>  // malloc_usable_size (glibc) для safe probe boundary
#include <mutex>
#include <string>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Константы
// ════════════════════════════════════════════════════════════════════════════

/// Максимальный размер cl_mem объекта для сканирования (байт)
static constexpr int kMaxProbeBytes = 2048;

/// Шаг сканирования (sizeof(void*) на 64-bit)
static constexpr int kProbeStep = 8;

// ════════════════════════════════════════════════════════════════════════════
// Struct: HsaProbeResult
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Результат HSA probe — GPU VA из cl_mem
 */
struct HsaProbeResult {
  void* gpu_va = nullptr;      ///< GPU virtual address (валиден в HIP)
  size_t alloc_size = 0;       ///< Размер HSA-аллокации
  int offset = -1;             ///< Offset внутри cl_mem handle
  bool valid = false;          ///< Успешно ли найден GPU VA
};

// ════════════════════════════════════════════════════════════════════════════
// HSA Lifecycle
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Инициализация HSA runtime (lazy, thread-safe)
 *
 * Вызывает hsa_init() один раз. Повторные вызовы — no-op.
 * HSA runtime живёт до завершения процесса (hsa_shut_down не вызываем —
 * это безопасно, ROCm runtime делает cleanup при exit).
 *
 * @return true если HSA доступна
 */
inline bool EnsureHsaInitialized() {
  static std::once_flag flag;
  static bool ok = false;
  std::call_once(flag, []() {
    hsa_status_t st = hsa_init();
    ok = (st == HSA_STATUS_SUCCESS);
  });
  return ok;
}

/**
 * @brief Проверить доступность HSA runtime
 */
inline bool IsHsaAvailable() {
  return EnsureHsaInitialized();
}

// ════════════════════════════════════════════════════════════════════════════
// ProbeGpuVA — главная функция
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Извлечь GPU VA из cl_mem через HSA pointer info probe
 *
 * Сканирует внутреннюю структуру cl_mem (amd::Memory в CLR),
 * ищет поле содержащее HSA-managed GPU pointer с совпадающим размером.
 *
 * Первый вызов: полное сканирование (~100μs).
 * Последующие: чтение по кешированному offset (~наносекунды).
 *
 * @param cl_buffer cl_mem handle (от clCreateBuffer)
 * @param expected_size Размер буфера в байтах (для верификации)
 * @return HsaProbeResult с GPU VA или valid=false
 *
 * @note Thread-safe (мьютекс на первый probe)
 * @note Не модифицирует cl_mem
 */
inline HsaProbeResult ProbeGpuVA(cl_mem cl_buffer, size_t expected_size) {
  HsaProbeResult result;

  if (!cl_buffer || expected_size == 0) return result;
  if (!EnsureHsaInitialized()) return result;

  // Кеш offset: -1 = не определён, -2 = probe не нашёл (не пробовать снова)
  static std::atomic<int> cached_offset{-1};
  static std::mutex probe_mutex;

  int off = cached_offset.load(std::memory_order_acquire);

  // Быстрый путь: offset уже известен
  if (off >= 0) {
    auto* raw = reinterpret_cast<uint8_t*>(cl_buffer);
    void* val = *reinterpret_cast<void**>(raw + off);
    if (val && reinterpret_cast<uintptr_t>(val) % 4096 == 0) {
      hsa_amd_pointer_info_t info = {};
      info.size = sizeof(info);
      if (hsa_amd_pointer_info(val, &info, nullptr, nullptr, nullptr) == HSA_STATUS_SUCCESS
          && info.type == HSA_EXT_POINTER_TYPE_HSA
          && info.sizeInBytes >= expected_size
          && info.agentBaseAddress == val) {
        result.gpu_va = val;
        result.alloc_size = info.sizeInBytes;
        result.offset = off;
        result.valid = true;
        return result;
      }
    }
    // Cached offset не подошёл для ЭТОГО буфера — НЕ сбрасываем кеш!
    // Другой поток может успешно использовать этот offset для другого буфера.
    // Просто переходим к полному скану ниже (под мьютексом).
  }

  // Если probe ранее провалился (-2), всё равно пробуем снова —
  // другой буфер может иметь другой размер/alignment


  // Полное сканирование (однократно)
  std::lock_guard<std::mutex> lock(probe_mutex);

  // Double-check после захвата мьютекса (с полной HSA валидацией)
  off = cached_offset.load(std::memory_order_acquire);
  if (off >= 0) {
    auto* raw = reinterpret_cast<uint8_t*>(cl_buffer);
    void* val = *reinterpret_cast<void**>(raw + off);
    if (val && reinterpret_cast<uintptr_t>(val) % 4096 == 0) {
      hsa_amd_pointer_info_t info = {};
      info.size = sizeof(info);
      if (hsa_amd_pointer_info(val, &info, nullptr, nullptr, nullptr) == HSA_STATUS_SUCCESS
          && info.type == HSA_EXT_POINTER_TYPE_HSA
          && info.sizeInBytes >= expected_size
          && info.agentBaseAddress == val) {
        result.gpu_va = val;
        result.alloc_size = info.sizeInBytes;
        result.offset = off;
        result.valid = true;
        return result;
      }
    }
  }

  // Сканируем cl_mem объект — собираем HSA-managed GPU VA кандидатов.
  // ВАЖНО: cl_mem — C++ объект ~500-2000 байт. Сканирование до kMaxProbeBytes
  // может читать за пределами объекта (heap overread). На практике безопасно:
  // heap page обычно 4KB, мусорные значения фильтруются HSA/alignment проверками.
  auto* raw = reinterpret_cast<uint8_t*>(cl_buffer);

  // Ограничиваем скан реальным размером heap-аллокации (glibc extension)
  int max_scan = kMaxProbeBytes;
#ifdef __GLIBC__
  size_t heap_size = malloc_usable_size(static_cast<void*>(cl_buffer));
  if (heap_size > 0 && static_cast<int>(heap_size) < max_scan) {
    max_scan = static_cast<int>(heap_size);
  }
#endif

  // Фаза 1: собрать все подходящие HSA-указатели
  struct Candidate { int offset; void* va; size_t size; };
  Candidate candidates[16];
  int n_candidates = 0;

  for (int probe_off = 0; probe_off < max_scan && n_candidates < 16;
       probe_off += kProbeStep) {
    void* val = *reinterpret_cast<void**>(raw + probe_off);

    if (!val || reinterpret_cast<uintptr_t>(val) < 0x10000) continue;

    // GPU VRAM аллокации page-aligned (4K minimum)
    if (reinterpret_cast<uintptr_t>(val) % 4096 != 0) continue;

    hsa_amd_pointer_info_t info = {};
    info.size = sizeof(info);
    hsa_status_t st = hsa_amd_pointer_info(val, &info, nullptr, nullptr, nullptr);

    if (st != HSA_STATUS_SUCCESS || info.type != HSA_EXT_POINTER_TYPE_HSA) continue;
    if (info.sizeInBytes < expected_size) continue;

    // Ищем начало аллокации (не offset внутри чужой)
    if (info.agentBaseAddress && info.agentBaseAddress != val) continue;

    // Дедупликация: один и тот же VA может быть в нескольких полях
    bool dup = false;
    for (int i = 0; i < n_candidates; i++) {
      if (candidates[i].va == val) { dup = true; break; }
    }
    if (dup) continue;

    candidates[n_candidates++] = {probe_off, val, info.sizeInBytes};
  }

  // Фаза 2: если один кандидат — берём его. Если несколько — выбираем по совпадению размера.
  if (n_candidates == 1) {
    cached_offset.store(candidates[0].offset, std::memory_order_release);
    result.gpu_va = candidates[0].va;
    result.alloc_size = candidates[0].size;
    result.offset = candidates[0].offset;
    result.valid = true;
    return result;
  }

  if (n_candidates > 1) {
    // Предпочитаем кандидата с точным размером
    for (int i = 0; i < n_candidates; i++) {
      if (candidates[i].size == expected_size) {
        cached_offset.store(candidates[i].offset, std::memory_order_release);
        result.gpu_va = candidates[i].va;
        result.alloc_size = candidates[i].size;
        result.offset = candidates[i].offset;
        result.valid = true;
        return result;
      }
    }

    // Нет exact match — берём с ближайшим размером
    size_t best_diff = SIZE_MAX;
    int best_idx = 0;
    for (int i = 0; i < n_candidates; i++) {
      size_t diff = candidates[i].size - expected_size;
      if (diff < best_diff) { best_diff = diff; best_idx = i; }
    }
    cached_offset.store(candidates[best_idx].offset, std::memory_order_release);
    result.gpu_va = candidates[best_idx].va;
    result.alloc_size = candidates[best_idx].size;
    result.offset = candidates[best_idx].offset;
    result.valid = true;
    return result;
  }

  return result;
}

/**
 * @brief Экспорт GPU VA как dma-buf fd (для hipImportExternalMemory)
 *
 * @param gpu_va GPU virtual address (из ProbeGpuVA)
 * @param size Размер буфера
 * @param[out] fd DMA-BUF file descriptor
 * @param[out] offset Offset внутри dma-buf
 * @return true если экспорт успешен
 */
inline bool ExportGpuVAasDmaBuf(void* gpu_va, size_t size,
                                  int* fd, uint64_t* offset) {
  if (!gpu_va || !fd || !offset) return false;
  if (!EnsureHsaInitialized()) return false;

  hsa_amd_pointer_info_t info = {};
  info.size = sizeof(info);
  if (hsa_amd_pointer_info(gpu_va, &info, nullptr, nullptr, nullptr) != HSA_STATUS_SUCCESS) {
    return false;
  }

  void* base = info.agentBaseAddress ? info.agentBaseAddress : gpu_va;
  size_t alloc_size = info.sizeInBytes > 0 ? info.sizeInBytes : size;

  hsa_status_t st = hsa_amd_portable_export_dmabuf(base, alloc_size, fd, offset);
  return (st == HSA_STATUS_SUCCESS && *fd >= 0);
}

/**
 * @brief Закрыть dma-buf fd
 */
inline void CloseDmaBuf(int fd) {
  if (fd >= 0) {
    hsa_amd_portable_close_dmabuf(fd);
  }
}

}  // namespace drv_gpu_lib

#endif  // ENABLE_ROCM
