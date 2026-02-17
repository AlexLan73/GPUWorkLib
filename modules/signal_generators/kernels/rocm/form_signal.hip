/**
 * @file form_signal.hip
 * @brief HIP kernel stub for FormSignalGenerator (ROCm backend)
 *
 * TODO: Implement HIP version of getX formula kernel.
 * Reference: ../form_signal_generator.cpp (inline OpenCL kernel source)
 *
 * Required:
 * - Philox-2x32-10 PRNG (or hipRAND)
 * - Box-Muller transform for Gaussian noise
 * - getX formula: X = a*norm*exp(j*(2pi*f0*t + pi*fdev/ti*((t-ti/2)^2) + phi))
 * - Per-channel delay: FIXED / LINEAR / RANDOM
 * - Window: X=0 if t<0 or t>ti-dt
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-17
 */

#include <hip/hip_runtime.h>

// ── Placeholder: not implemented ──
// When implementing:
// 1. Port Philox PRNG from OpenCL to HIP
// 2. Port getX kernel with same parameter layout
// 3. Use hipLaunchKernelGGL for dispatch
// 4. Test against OpenCL version for numerical equivalence

__global__ void form_signal_kernel(
    float2* __restrict__ output,
    float   fs,
    float   f0,
    float   amplitude,
    float   noise_amplitude,
    float   phase,
    float   fdev,
    float   norm,
    float   tau_base,
    float   tau_step,
    float   tau_min,
    float   tau_max,
    uint    tau_seed,
    uint    noise_seed,
    uint    antennas,
    uint    points,
    uint    delay_mode)
{
    // TODO: implement HIP kernel
    // See form_signal_generator.cpp KERNEL_SOURCE for reference
}
