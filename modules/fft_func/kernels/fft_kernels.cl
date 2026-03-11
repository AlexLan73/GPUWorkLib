// ============================================================================
// FFT PIPELINE GPU KERNELS
// ============================================================================
// Source: LCH-Farrow01/great-heyrovsky
// Date: 2026-02-04
//
// These kernels are extracted from antenna_fft_proc_max.cpp
// They are embedded as string literals in the C++ code but
// provided here separately for reference and potential external compilation.
// ============================================================================

// ============================================================================
// KERNEL #1: PADDING KERNEL
// ============================================================================
// Purpose: Copy input data and zero-pad to nFFT size
// Supports batch processing via beam_offset (NO DATA COPY!)

__kernel void padding_kernel(
    __global const float2* input,    // Full input buffer (all beams)
    __global float2* output,         // Output: batch_beam_count * nFFT
    uint batch_beam_count,           // Number of beams in current batch
    uint count_points,               // Points per beam (input)
    uint nFFT,                       // FFT size (output)
    uint beam_offset                 // Beam offset for batch processing
) {
    uint gid = get_global_id(0);

    // gid = local_beam_idx * nFFT + pos_in_fft
    uint local_beam_idx = gid / nFFT;
    uint pos_in_fft = gid % nFFT;

    if (local_beam_idx >= batch_beam_count) return;

    // Global beam index = local + offset
    uint global_beam_idx = local_beam_idx + beam_offset;

    if (pos_in_fft < count_points) {
        // Copy from global index, write to local
        uint src_idx = global_beam_idx * count_points + pos_in_fft;
        output[gid] = input[src_idx];
    } else {
        // Zero-padding
        output[gid] = (float2)(0.0f, 0.0f);
    }
}


// ============================================================================
// KERNEL #2: POST KERNEL (UNIFIED)
// ============================================================================
// Purpose:
//   1. Calculate magnitude = sqrt(re^2 + im^2)
//   2. Find top-N maxima using parallel reduction
//   3. Calculate phase = atan2(im, re) * 180/PI
//   4. Parabolic interpolation for frequency refinement (peak #0 only)
//
// Architecture:
//   - One work-group per beam
//   - 256 work-items per group
//   - Local memory reduction

// Result structure (must match C++ MaxValue)
typedef struct {
    uint index;
    float real;               // Real part
    float imag;               // Imaginary part
    float magnitude;
    float phase;
    float freq_offset;        // Parabolic interpolation offset (-0.5..+0.5)
    float refined_frequency;  // Refined frequency in Hz
    uint pad;                 // Alignment to 32 bytes
} MaxValue;

__kernel void post_kernel(
    __global const float2* fft_output,     // FFT result: beam_count * nFFT
    __global MaxValue* maxima_output,      // Output: beam_count * max_peaks_count
    uint beam_count,
    uint nFFT,
    uint search_range,                     // Points to analyze (filter)
    uint max_peaks_count,                  // Number of maxima to find (3, 5, 7...)
    float sample_rate                      // Sample rate (default 12 MHz)
) {
    uint beam_idx = get_group_id(0);
    uint lid = get_local_id(0);
    uint local_size = get_local_size(0);

    if (beam_idx >= beam_count) return;

    // Local memory for reduction (MUST be in outermost scope!)
    __local float local_mag[256];
    __local uint local_idx[256];
    __local float2 local_complex[256];
    __local float found_mags[16];
    __local uint found_indices[16];
    __local float2 found_complex[16];

    // =========================================================================
    // STAGE 1: Each thread finds its local maximum
    // =========================================================================
    float my_max_mag = -1.0f;
    uint my_max_idx = 0;
    float2 my_max_complex = (float2)(0.0f, 0.0f);

    // Each thread processes multiple points
    for (uint i = lid; i < search_range; i += local_size) {
        uint fft_idx = beam_idx * nFFT + i;
        float2 val = fft_output[fft_idx];
        float mag = sqrt(val.x * val.x + val.y * val.y);

        if (mag > my_max_mag) {
            my_max_mag = mag;
            my_max_idx = i;
            my_max_complex = val;
        }
    }

    local_mag[lid] = my_max_mag;
    local_idx[lid] = my_max_idx;
    local_complex[lid] = my_max_complex;
    barrier(CLK_LOCAL_MEM_FENCE);

    // =========================================================================
    // STAGE 2: Thread 0 finds top-N maxima sequentially
    // =========================================================================

    if (lid == 0) {
        for (uint peak = 0; peak < max_peaks_count && peak < 16; ++peak) {
            float best_mag = -1.0f;
            uint best_idx = 0;
            float2 best_complex = (float2)(0.0f, 0.0f);
            uint best_local_idx = 0;

            // Find maximum among local_mag
            for (uint j = 0; j < local_size; ++j) {
                if (local_mag[j] > best_mag) {
                    best_mag = local_mag[j];
                    best_idx = local_idx[j];
                    best_complex = local_complex[j];
                    best_local_idx = j;
                }
            }

            found_mags[peak] = best_mag;
            found_indices[peak] = best_idx;
            found_complex[peak] = best_complex;

            // "Remove" found maximum
            local_mag[best_local_idx] = -1.0f;

            // Also remove this index from all threads
            for (uint j = 0; j < local_size; ++j) {
                if (local_idx[j] == best_idx) {
                    local_mag[j] = -1.0f;
                }
            }
        }

        // =====================================================================
        // STAGE 3: Write results with Re/Im and parabolic interpolation
        // =====================================================================

        // Bin width in Hz
        float bin_width = sample_rate / (float)nFFT;

        for (uint peak = 0; peak < max_peaks_count && peak < 16; ++peak) {
            uint out_idx = beam_idx * max_peaks_count + peak;

            MaxValue mv;
            mv.index = found_indices[peak];

            // Re and Im for all peaks
            float2 c = found_complex[peak];
            mv.real = c.x;
            mv.imag = c.y;
            mv.magnitude = found_mags[peak];

            // Phase in degrees
            float phase_rad = atan2(c.y, c.x);
            mv.phase = phase_rad * 57.2957795131f;  // 180/PI

            // Default: no interpolation
            mv.freq_offset = 0.0f;
            mv.refined_frequency = (float)mv.index * bin_width;

            // =================================================================
            // PARABOLIC INTERPOLATION: only for peak == 0!
            // =================================================================
            if (peak == 0) {
                uint center_idx = found_indices[0];

                // Boundary check
                if (center_idx > 0 && center_idx < search_range - 1) {
                    uint base_idx = beam_idx * nFFT;

                    // Read neighbor points from spectrum
                    float2 left_val = fft_output[base_idx + center_idx - 1];
                    float2 right_val = fft_output[base_idx + center_idx + 1];

                    float y_left = sqrt(left_val.x * left_val.x + left_val.y * left_val.y);
                    float y_center = found_mags[0];
                    float y_right = sqrt(right_val.x * right_val.x + right_val.y * right_val.y);

                    // Three-point parabolic interpolation
                    // offset = 0.5 * (y_left - y_right) / (y_left - 2*y_center + y_right)
                    float denom = y_left - 2.0f * y_center + y_right;

                    if (fabs(denom) > 1e-10f) {
                        float offset = 0.5f * (y_left - y_right) / denom;

                        // Clamp offset to [-0.5, +0.5]
                        offset = clamp(offset, -0.5f, 0.5f);

                        mv.freq_offset = offset;

                        // Refined frequency in Hz
                        float refined_index = (float)center_idx + offset;
                        mv.refined_frequency = refined_index * bin_width;
                    }
                }
            }

            mv.pad = 0;
            maxima_output[out_idx] = mv;
        }
    }
}
