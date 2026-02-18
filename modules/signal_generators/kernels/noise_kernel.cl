/**
 * @file noise_kernel.cl
 * @brief OpenCL kernels: Noise generation (Philox PRNG + Box-Muller)
 *
 * Requires: prng.cl (prepended at compile time)
 *
 * Two modes:
 * - Gaussian noise (Box-Muller transform)
 * - White noise (uniform [-1, 1])
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-18
 */

/**
 * @brief Gaussian noise: Philox + Box-Muller
 *
 * Each work item generates one complex point (re, im)
 * with normal distribution N(0, std_dev).
 */
__kernel void generate_noise_gaussian(
    __global float2* output,
    const uint total_points,
    const float std_dev,
    const uint seed)
{
    const uint gid = get_global_id(0);
    if (gid >= total_points) return;

    // Generate 2 uniform random numbers via Philox
    uint2 ctr = (uint2)(gid, seed);
    uint2 rnd = philox2x32_10(ctr, 0xCD9E8D57u);

    // Uniform [0, 1)
    float u1 = (float)(rnd.x) / 4294967296.0f + 1e-10f;  // avoid log(0)
    float u2 = (float)(rnd.y) / 4294967296.0f;

    // Box-Muller transform: uniform -> Gaussian
    float r = sqrt(-2.0f * log(u1)) * std_dev;
    float theta = 2.0f * M_PI_F * u2;

    output[gid] = (float2)(r * cos(theta), r * sin(theta));
}

/**
 * @brief White noise (uniform [-amplitude, +amplitude])
 */
__kernel void generate_noise_white(
    __global float2* output,
    const uint total_points,
    const float amplitude,
    const uint seed)
{
    const uint gid = get_global_id(0);
    if (gid >= total_points) return;

    uint2 ctr = (uint2)(gid, seed);
    uint2 rnd = philox2x32_10(ctr, 0xCD9E8D57u);

    // Uniform [-amplitude, +amplitude]
    float re = ((float)(rnd.x) / 4294967296.0f * 2.0f - 1.0f) * amplitude;
    float im = ((float)(rnd.y) / 4294967296.0f * 2.0f - 1.0f) * amplitude;

    output[gid] = (float2)(re, im);
}
