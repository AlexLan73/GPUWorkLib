#pragma once

/**
 * @file async_service_base.hpp
 * @brief AsyncServiceBase - Base class for asynchronous background services
 *
 * ============================================================================
 * PURPOSE:
 *   Template base class for Logger, Profiler, ConsoleOutput, and future services.
 *   Provides a worker thread + message queue + observer pattern.
 *
 * ARCHITECTURE:
 *   GPU Thread 0 --> Enqueue(msg) --+
 *   GPU Thread 1 --> Enqueue(msg) --+--> [Queue] --> Worker Thread --> ProcessMessage(msg)
 *   GPU Thread N --> Enqueue(msg) --+
 *
 * GUARANTEES:
 *   - GPU threads NEVER block on output (only lock-free-ish Enqueue)
 *   - All processing happens in dedicated background thread
 *   - On Stop(): waits for all queued messages to be processed
 *   - Thread-safe: multiple producers, single consumer
 *
 * PATTERN: Producer-Consumer + Observer
 *   - Producers: GPU threads call Enqueue()
 *   - Consumer: Worker thread calls ProcessMessage() (virtual)
 *   - Observer: Worker wakes up on condition_variable notify
 *
 * USAGE:
 *   class MyService : public AsyncServiceBase<MyMessage> {
 *   protected:
 *       void ProcessMessage(const MyMessage& msg) override {
 *           // Handle message in background thread
 *       }
 *   };
 *
 *   MyService service;
 *   service.Start();
 *   service.Enqueue({...});  // Non-blocking!
 *   service.Stop();          // Waits for queue drain
 * ============================================================================
 *
 * @author Codo (AI Assistant)
 * @date 2026-02-07
 */

#include <queue>
#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <vector>
#include <string>

namespace drv_gpu_lib {

// ============================================================================
// AsyncServiceBase - Template base for background services
// ============================================================================

/**
 * @class AsyncServiceBase
 * @brief Template base class for asynchronous services with message queue
 *
 * @tparam TMessage Type of messages processed by this service
 *
 * Derived classes must implement:
 * - ProcessMessage(const TMessage& msg) - handle one message
 * - GetServiceName() - return human-readable service name
 *
 * Lifecycle:
 * 1. Construct derived class
 * 2. Call Start() to launch worker thread
 * 3. Call Enqueue() from any thread (non-blocking)
 * 4. Call Stop() to shutdown (drains queue first)
 *
 * Thread Model:
 * - Worker thread runs WorkerLoop() in background
 * - WorkerLoop() waits on condition_variable
 * - When messages arrive, wakes up and processes all pending
 * - On Stop(), processes remaining messages and joins thread
 */
template<typename TMessage>
class AsyncServiceBase {
public:
    // ========================================================================
    // Constructor / Destructor
    // ========================================================================

    /**
     * @brief Default constructor (does NOT start worker thread)
     * Call Start() to begin processing.
     */
    AsyncServiceBase() = default;

    /**
     * @brief Destructor - automatically stops worker thread
     * Waits for all queued messages to be processed.
     */
    virtual ~AsyncServiceBase() {
        Stop();
    }

    // Delete copy, allow move
    AsyncServiceBase(const AsyncServiceBase&) = delete;
    AsyncServiceBase& operator=(const AsyncServiceBase&) = delete;

    // ========================================================================
    // Lifecycle Management
    // ========================================================================

    /**
     * @brief Start the worker thread
     *
     * Launches a background thread that processes messages from the queue.
     * Safe to call multiple times (only starts once).
     *
     * @note Must be called before Enqueue() will have any effect.
     */
    void Start() {
        // Avoid double-start
        if (running_.load(std::memory_order_acquire)) {
            return;
        }

        running_.store(true, std::memory_order_release);

        worker_thread_ = std::thread([this]() {
            WorkerLoop();
        });
    }

    /**
     * @brief Stop the worker thread
     *
     * Signals the worker to stop, then waits for all queued messages
     * to be processed before joining the thread.
     *
     * Safe to call multiple times (only stops once).
     * Called automatically from destructor.
     */
    void Stop() {
        // Signal stop
        bool expected = true;
        if (!running_.compare_exchange_strong(expected, false, std::memory_order_acq_rel)) {
            return; // Already stopped or never started
        }

        // Wake up worker thread to notice the stop signal
        cv_.notify_one();

        // Wait for worker thread to finish
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }

    /**
     * @brief Check if the service is running
     * @return true if worker thread is active
     */
    bool IsRunning() const {
        return running_.load(std::memory_order_acquire);
    }

    // ========================================================================
    // Message Queue (Non-Blocking Producer API)
    // ========================================================================

    /**
     * @brief Enqueue a message for background processing
     *
     * This is the PRIMARY API for GPU threads.
     * Almost non-blocking: only acquires mutex for queue push.
     *
     * @param msg Message to process (moved into queue)
     *
     * @note If service is not running, message is silently dropped.
     *       This is intentional to avoid blocking GPU threads.
     */
    void Enqueue(TMessage msg) {
        if (!running_.load(std::memory_order_acquire)) {
            return; // Service not running, drop message
        }

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            queue_.push(std::move(msg));
        }

        // Wake up worker thread
        cv_.notify_one();
    }

    /**
     * @brief Enqueue multiple messages at once (batch)
     *
     * More efficient than calling Enqueue() multiple times
     * as it only locks once and notifies once.
     *
     * @param messages Vector of messages to enqueue
     */
    void EnqueueBatch(std::vector<TMessage> messages) {
        if (!running_.load(std::memory_order_acquire) || messages.empty()) {
            return;
        }

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            for (auto& msg : messages) {
                queue_.push(std::move(msg));
            }
        }

        cv_.notify_one();
    }

    /**
     * @brief Get current queue size (approximate, for diagnostics)
     * @return Number of pending messages
     */
    size_t GetQueueSize() const {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return queue_.size();
    }

    /**
     * @brief Get total number of messages processed since Start()
     * @return Count of processed messages
     */
    uint64_t GetProcessedCount() const {
        return processed_count_.load(std::memory_order_acquire);
    }

protected:
    // ========================================================================
    // Virtual Methods (must be implemented by derived classes)
    // ========================================================================

    /**
     * @brief Process one message from the queue
     *
     * Called by the worker thread for each message.
     * This is where derived classes implement their logic.
     *
     * IMPORTANT: This runs in the WORKER THREAD, not the GPU thread!
     * So it's safe to do I/O, file writes, console output, etc.
     *
     * @param msg The message to process
     */
    virtual void ProcessMessage(const TMessage& msg) = 0;

    /**
     * @brief Get human-readable service name (for diagnostics)
     * @return Service name (e.g., "Logger", "Profiler", "ConsoleOutput")
     */
    virtual std::string GetServiceName() const = 0;

    /**
     * @brief Called when worker thread starts (optional override)
     * Use for thread-local initialization.
     */
    virtual void OnWorkerStart() {}

    /**
     * @brief Called when worker thread stops (optional override)
     * Use for thread-local cleanup.
     */
    virtual void OnWorkerStop() {}

private:
    // ========================================================================
    // Worker Thread Implementation
    // ========================================================================

    /**
     * @brief Main worker loop (runs in background thread)
     *
     * Algorithm:
     * 1. Wait on condition_variable (sleeps when queue is empty)
     * 2. Wake up on notify (from Enqueue) or stop signal
     * 3. Drain all pending messages from queue
     * 4. Process each message via ProcessMessage()
     * 5. Repeat until Stop() is called
     * 6. On stop: drain remaining messages, then exit
     */
    void WorkerLoop() {
        // Thread-local initialization
        OnWorkerStart();

        while (true) {
            std::vector<TMessage> batch;

            {
                std::unique_lock<std::mutex> lock(queue_mutex_);

                // Wait until: (a) queue has messages, or (b) stop signal
                cv_.wait(lock, [this]() {
                    return !queue_.empty() || !running_.load(std::memory_order_acquire);
                });

                // Drain all pending messages into local batch
                // (minimizes time holding the mutex)
                while (!queue_.empty()) {
                    batch.push_back(std::move(queue_.front()));
                    queue_.pop();
                }
            }

            // Process batch outside of lock
            for (const auto& msg : batch) {
                ProcessMessage(msg);
                processed_count_.fetch_add(1, std::memory_order_relaxed);
            }

            // Check if we should stop (after processing remaining messages)
            if (!running_.load(std::memory_order_acquire)) {
                // Final drain: process any messages that arrived during processing
                std::vector<TMessage> final_batch;
                {
                    std::lock_guard<std::mutex> lock(queue_mutex_);
                    while (!queue_.empty()) {
                        final_batch.push_back(std::move(queue_.front()));
                        queue_.pop();
                    }
                }
                for (const auto& msg : final_batch) {
                    ProcessMessage(msg);
                    processed_count_.fetch_add(1, std::memory_order_relaxed);
                }
                break;
            }
        }

        // Thread-local cleanup
        OnWorkerStop();
    }

    // ========================================================================
    // Private Members
    // ========================================================================

    /// Worker thread
    std::thread worker_thread_;

    /// Message queue (FIFO)
    std::queue<TMessage> queue_;

    /// Mutex protecting the queue
    mutable std::mutex queue_mutex_;

    /// Condition variable for worker wakeup
    std::condition_variable cv_;

    /// Running flag (atomic for lock-free check in Enqueue)
    std::atomic<bool> running_{false};

    /// Counter of processed messages (for diagnostics)
    std::atomic<uint64_t> processed_count_{0};
};

} // namespace drv_gpu_lib
