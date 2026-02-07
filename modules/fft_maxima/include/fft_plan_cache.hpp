#pragma once

/**
 * @file fft_plan_cache.hpp
 * @brief FFTPlanCache - Кеширование clFFT планов для повторного использования
 *
 * ============================================================================
 * ПРОБЛЕМА:
 *   В текущей реализации AntennaFFTProcMax при каждом изменении batch size
 *   (plan_num_beams_ != num_beams) FFT план пересоздаётся:
 *     - clfftDestroyPlan()
 *     - clfftCreateDefaultPlan()
 *     - clfftBakePlan()  // ДОРОГО! ~50-200ms
 *
 *   При batch processing это может происходить часто:
 *     Batch 1: 10 beams → create plan(10)
 *     Batch 2: 10 beams → reuse (OK)
 *     Batch 3: 3 beams (tail) → create plan(3), destroy plan(10)
 *     Batch 4 (next call): 10 beams → create plan(10) AGAIN!
 *
 * РЕШЕНИЕ:
 *   FFTPlanCache хранит пул планов по ключу (nFFT, batch_size).
 *   Планы создаются один раз и переиспользуются.
 *   При уничтожении кеша все планы корректно освобождаются.
 *
 * ИСПОЛЬЗОВАНИЕ:
 *   FFTPlanCache cache(context, queue);
 *
 *   // Первый раз: создаст и запомнит план
 *   auto& plan = cache.GetOrCreate(nFFT, 10);
 *
 *   // Второй раз: мгновенно вернёт кешированный
 *   auto& plan2 = cache.GetOrCreate(nFFT, 10);
 *
 *   // Другой размер: создаст новый план, старый останется в кеше
 *   auto& plan3 = cache.GetOrCreate(nFFT, 3);
 *
 *   // Снова 10: мгновенно
 *   auto& plan4 = cache.GetOrCreate(nFFT, 10);
 * ============================================================================
 *
 * @author Codo (AI Assistant)
 * @date 2026-02-07
 */

#include <clFFT.h>
#include <CL/cl.h>

#include <map>
#include <string>
#include <stdexcept>
#include <iostream>

namespace antenna_fft {

// ============================================================================
// FFTPlanKey - Unique key for a plan in the cache
// ============================================================================

/**
 * @struct FFTPlanKey
 * @brief Unique identifier for a cached FFT plan
 *
 * A plan is uniquely defined by:
 * - nFFT: FFT size (e.g., 2048, 4096)
 * - batch_size: number of batched FFTs (e.g., 10, 32)
 */
struct FFTPlanKey {
    size_t nFFT;        ///< FFT size
    size_t batch_size;  ///< Number of batched transforms

    /// Comparison operator for std::map
    bool operator<(const FFTPlanKey& other) const {
        if (nFFT != other.nFFT) return nFFT < other.nFFT;
        return batch_size < other.batch_size;
    }

    /// Equality operator
    bool operator==(const FFTPlanKey& other) const {
        return nFFT == other.nFFT && batch_size == other.batch_size;
    }
};

// ============================================================================
// FFTPlanEntry - One cached plan with its metadata
// ============================================================================

/**
 * @struct FFTPlanEntry
 * @brief A cached FFT plan with usage statistics
 */
struct FFTPlanEntry {
    clfftPlanHandle handle = 0;     ///< clFFT plan handle
    bool baked = false;             ///< Is the plan baked (ready to use)?
    size_t use_count = 0;           ///< How many times this plan was used
    size_t nFFT = 0;                ///< FFT size
    size_t batch_size = 0;          ///< Batch size
};

// ============================================================================
// FFTPlanCache - Cache of clFFT plans
// ============================================================================

/**
 * @class FFTPlanCache
 * @brief Manages a cache of clFFT plans for different configurations
 *
 * Plans are cached by (nFFT, batch_size) key.
 * Cache is NOT thread-safe (used from a single GPU thread).
 *
 * Memory management:
 * - Plans are destroyed in destructor (RAII)
 * - ClearAll() can be called to force release
 */
class FFTPlanCache {
public:
    // ========================================================================
    // Constructor / Destructor
    // ========================================================================

    /**
     * @brief Create plan cache for a specific OpenCL context
     * @param context OpenCL context (for plan creation)
     * @param queue Command queue (for plan baking)
     */
    FFTPlanCache(cl_context context, cl_command_queue queue)
        : context_(context), queue_(queue) {
    }

    /**
     * @brief Destructor - releases ALL cached plans
     */
    ~FFTPlanCache() {
        ClearAll();
    }

    // Delete copy (owns OpenCL resources)
    FFTPlanCache(const FFTPlanCache&) = delete;
    FFTPlanCache& operator=(const FFTPlanCache&) = delete;

    // Allow move
    FFTPlanCache(FFTPlanCache&& other) noexcept
        : context_(other.context_),
          queue_(other.queue_),
          cache_(std::move(other.cache_)),
          total_creates_(other.total_creates_),
          total_hits_(other.total_hits_) {
        other.context_ = nullptr;
        other.queue_ = nullptr;
    }

    // ========================================================================
    // Core API
    // ========================================================================

    /**
     * @brief Get or create a plan for the given configuration
     * @param nFFT FFT size
     * @param batch_size Number of batched transforms
     * @return Reference to the plan handle (ready to use after Bake)
     *
     * If plan exists in cache: returns immediately (cache HIT)
     * If plan doesn't exist: creates new plan, adds to cache (cache MISS)
     *
     * NOTE: The returned plan is NOT baked! Call BakeIfNeeded() or
     *       manually bake with callbacks before use.
     */
    clfftPlanHandle GetOrCreate(size_t nFFT, size_t batch_size) {
        FFTPlanKey key{nFFT, batch_size};

        auto it = cache_.find(key);
        if (it != cache_.end()) {
            // Cache HIT
            it->second.use_count++;
            total_hits_++;
            return it->second.handle;
        }

        // Cache MISS - create new plan
        FFTPlanEntry entry;
        entry.nFFT = nFFT;
        entry.batch_size = batch_size;

        // Create plan
        size_t dim = nFFT;
        clfftStatus status = clfftCreateDefaultPlan(&entry.handle, context_, CLFFT_1D, &dim);
        if (status != CLFFT_SUCCESS) {
            throw std::runtime_error(
                "[FFTPlanCache] clfftCreateDefaultPlan failed: " + std::to_string(status));
        }

        // Configure plan
        clfftSetPlanPrecision(entry.handle, CLFFT_SINGLE);
        clfftSetLayout(entry.handle, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
        clfftSetResultLocation(entry.handle, CLFFT_OUTOFPLACE);
        clfftSetPlanBatchSize(entry.handle, batch_size);

        size_t strides[1] = {1};
        size_t dist = nFFT;
        clfftSetPlanInStride(entry.handle, CLFFT_1D, strides);
        clfftSetPlanOutStride(entry.handle, CLFFT_1D, strides);
        clfftSetPlanDistance(entry.handle, dist, dist);

        entry.use_count = 1;
        entry.baked = false;

        cache_[key] = entry;
        total_creates_++;

        return entry.handle;
    }

    /**
     * @brief Check if a plan exists in cache
     * @param nFFT FFT size
     * @param batch_size Batch size
     * @return true if plan is cached
     */
    bool HasPlan(size_t nFFT, size_t batch_size) const {
        FFTPlanKey key{nFFT, batch_size};
        return cache_.find(key) != cache_.end();
    }

    /**
     * @brief Check if a cached plan is baked (ready for execution)
     * @param nFFT FFT size
     * @param batch_size Batch size
     * @return true if plan exists AND is baked
     */
    bool IsBaked(size_t nFFT, size_t batch_size) const {
        FFTPlanKey key{nFFT, batch_size};
        auto it = cache_.find(key);
        return it != cache_.end() && it->second.baked;
    }

    /**
     * @brief Mark a plan as baked (called after clfftBakePlan succeeds)
     * @param nFFT FFT size
     * @param batch_size Batch size
     */
    void MarkBaked(size_t nFFT, size_t batch_size) {
        FFTPlanKey key{nFFT, batch_size};
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            it->second.baked = true;
        }
    }

    /**
     * @brief Remove a specific plan from cache
     * @param nFFT FFT size
     * @param batch_size Batch size
     */
    void Remove(size_t nFFT, size_t batch_size) {
        FFTPlanKey key{nFFT, batch_size};
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            if (it->second.handle) {
                clfftDestroyPlan(&it->second.handle);
            }
            cache_.erase(it);
        }
    }

    /**
     * @brief Clear entire cache (release all plans)
     *
     * Called from destructor. Safe to call multiple times.
     */
    void ClearAll() {
        for (auto& [key, entry] : cache_) {
            if (entry.handle) {
                clfftDestroyPlan(&entry.handle);
                entry.handle = 0;
            }
        }
        cache_.clear();
    }

    // ========================================================================
    // Statistics
    // ========================================================================

    /**
     * @brief Get number of cached plans
     */
    size_t GetCacheSize() const { return cache_.size(); }

    /**
     * @brief Get total number of plan creations (cache misses)
     */
    size_t GetTotalCreates() const { return total_creates_; }

    /**
     * @brief Get total number of cache hits
     */
    size_t GetTotalHits() const { return total_hits_; }

    /**
     * @brief Get cache hit ratio (0.0 - 1.0)
     */
    double GetHitRatio() const {
        size_t total = total_creates_ + total_hits_;
        return total > 0 ? static_cast<double>(total_hits_) / total : 0.0;
    }

    /**
     * @brief Print cache statistics
     */
    void PrintStats() const {
        std::cout << "\n  FFTPlanCache Statistics:\n";
        std::cout << "    Cached plans: " << cache_.size() << "\n";
        std::cout << "    Total creates: " << total_creates_ << "\n";
        std::cout << "    Cache hits: " << total_hits_ << "\n";
        std::cout << "    Hit ratio: " << (GetHitRatio() * 100.0) << "%\n";

        if (!cache_.empty()) {
            std::cout << "    Plans:\n";
            for (const auto& [key, entry] : cache_) {
                std::cout << "      nFFT=" << key.nFFT
                          << " batch=" << key.batch_size
                          << " baked=" << (entry.baked ? "yes" : "no")
                          << " uses=" << entry.use_count << "\n";
            }
        }
        std::cout << "\n";
    }

private:
    // ========================================================================
    // Private members
    // ========================================================================

    cl_context context_;                          ///< OpenCL context
    cl_command_queue queue_;                       ///< Command queue

    std::map<FFTPlanKey, FFTPlanEntry> cache_;     ///< Plan cache

    size_t total_creates_ = 0;                    ///< Total plan creations
    size_t total_hits_ = 0;                       ///< Total cache hits
};

} // namespace antenna_fft
