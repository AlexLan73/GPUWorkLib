#pragma once

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Enum для типов памяти GPU
// ════════════════════════════════════════════════════════════════════════════

enum class MemoryType {
    GPU_READ_ONLY,
    GPU_WRITE_ONLY,
    GPU_READ_WRITE
};

} // namespace ManagerOpenCL

