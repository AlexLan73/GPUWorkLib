/**
 * @file prng.cl
 * @brief Philox-2x32-10 PRNG + Box-Muller transform
 *
 * Shared PRNG functions for signal generators.
 * Prepended to kernels that require noise generation.
 *
 * Usage: concatenate this file before kernel .cl files that use:
 *   - philox2x32_10()
 *   - philox_uniform()
 *   - Box-Muller noise pattern (u1, u2 -> r, theta)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-18
 */

// Philox-2x32-10: counter-based PRNG
uint2 philox2x32_round(uint2 ctr, uint key) {
    const uint PHILOX_M = 0xD2511F53u;
    uint hi = mul_hi(ctr.x, PHILOX_M);
    uint lo = ctr.x * PHILOX_M;
    return (uint2)(hi ^ key ^ ctr.y, lo);
}

uint2 philox2x32_10(uint2 ctr, uint key) {
    const uint PHILOX_BUMP = 0x9E3779B9u;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key);
    return ctr;
}

/// Uniform [0, 1) from Philox
float philox_uniform(uint id, uint seed) {
    uint2 ctr = (uint2)(id, seed);
    uint2 rnd = philox2x32_10(ctr, 0xAB12CD34u);
    return (float)(rnd.x) / 4294967296.0f;
}
