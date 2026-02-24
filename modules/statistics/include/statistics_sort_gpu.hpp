#pragma once
/**
 * @file statistics_sort_gpu.hpp
 * @brief Declaration of GPU segmented sort (implemented in statistics_sort_gpu.hip)
 *
 * This header is included from both:
 *   - statistics_sort_gpu.hip  (HIP compiler — definition)
 *   - statistics_processor.cpp (g++ — usage)
 *
 * Functions use C++ linkage inside the statistics::gpu_sort namespace.
 */

#if ENABLE_ROCM

#include <hip/hip_runtime.h>
#include <cstddef>

namespace statistics {
namespace gpu_sort {

/**
 * @brief Query temp storage size for segmented radix sort.
 *
 * @param[out] temp_size    Required temp storage bytes
 * @param d_begin_offsets   Device ptr: begin offsets per segment [beam_count]
 * @param d_end_offsets     Device ptr: end offsets per segment   [beam_count]
 * @param total_elements    beam_count * n_point
 * @param num_segments      beam_count
 * @param stream            HIP stream
 */
hipError_t QuerySortTempSize(
    size_t&               temp_size,
    const unsigned int*   d_begin_offsets,
    const unsigned int*   d_end_offsets,
    unsigned int          total_elements,
    unsigned int          num_segments,
    hipStream_t           stream);

/**
 * @brief Execute segmented radix sort (ascending) on float magnitudes.
 *
 * Each segment (beam) is sorted independently in parallel.
 * All 256 beams are sorted in ONE GPU call.
 *
 * @param temp_storage      Pre-allocated temp buffer (from QuerySortTempSize)
 * @param temp_size         Size of temp_storage
 * @param keys_in           Device ptr: float magnitudes [total_elements]
 * @param keys_out          Device ptr: sorted output   [total_elements]
 * @param d_begin_offsets   Device ptr: begin offsets [num_segments]
 * @param d_end_offsets     Device ptr: end offsets   [num_segments]
 * @param total_elements    beam_count * n_point
 * @param num_segments      beam_count
 * @param stream            HIP stream
 */
hipError_t ExecuteSort(
    void*                 temp_storage,
    size_t                temp_size,
    const float*          keys_in,
    float*                keys_out,
    const unsigned int*   d_begin_offsets,
    const unsigned int*   d_end_offsets,
    unsigned int          total_elements,
    unsigned int          num_segments,
    hipStream_t           stream);

}  // namespace gpu_sort
}  // namespace statistics

#endif  // ENABLE_ROCM
