#include "command_queue_pool.hpp"
#include <iostream>
#include <thread>
#include <sstream>
#include <iomanip>
#include <memory>


#include "command_queue_pool.hpp"
#include "opencl_core.hpp"
#include <iostream>
#include <sstream>
#include <thread>
#include <cstdlib>
#include <ctime>

namespace ManagerOpenCL {

// ✅ Static variable initialization
std::mutex CommandQueuePool::mutex_;
bool CommandQueuePool::initialized_ = false;
std::vector<cl_command_queue> CommandQueuePool::queues_;
std::atomic<size_t> CommandQueuePool::current_index_{0};
size_t CommandQueuePool::queue_counter_ = 0;
std::vector<size_t> CommandQueuePool::queue_usage_;

// Initialize the pool
void CommandQueuePool::Initialize(size_t num_queues) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (initialized_) {
        std::cout << "[CommandQueuePool] Already initialized\n";
        return;
    }
    
    std::cout << "[CommandQueuePool] Initializing with " 
              << (num_queues > 0 ? num_queues : std::thread::hardware_concurrency())
              << " queues\n";
    
    CommandQueuePool::CreateQueues(num_queues);
    queue_usage_.resize(queues_.size(), 0);
    queue_counter_ = 0;
    
    initialized_ = true;
    std::cout << "[CommandQueuePool] Initialized successfully\n";
}

// Cleanup all queues
void CommandQueuePool::Cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_) return;
    
    std::cout << "[CommandQueuePool] Cleaning up...\n";
    
    CommandQueuePool::ReleaseQueues();
    queues_.clear();
    queue_usage_.clear();
    queue_counter_ = 0;
    initialized_ = false;
    
    std::cout << "[CommandQueuePool] Cleaned up successfully\n";
}

// Check if initialized
bool CommandQueuePool::IsInitialized() {
    std::lock_guard<std::mutex> lock(mutex_);
    return initialized_;
}

// Get next queue (round-robin)
cl_command_queue CommandQueuePool::GetNextQueue() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_ || queues_.empty()) {
        throw std::runtime_error("[CommandQueuePool] Not initialized!");
    }
    
    size_t index = queue_counter_ % queues_.size();
    queue_counter_++;
    queue_usage_[index]++;
    
    return queues_[index];
}

// Get queue by index
cl_command_queue CommandQueuePool::GetQueue(size_t index) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_ || queues_.empty() || index >= queues_.size()) {
        throw std::runtime_error("[CommandQueuePool] Invalid queue index!");
    }
    
    queue_usage_[index]++;
    return queues_[index];
}

// Get random queue
cl_command_queue CommandQueuePool::GetRandomQueue() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_ || queues_.empty()) {
        throw std::runtime_error("[CommandQueuePool] Not initialized!");
    }
    
    size_t index = rand() % queues_.size();
    queue_usage_[index]++;
    
    return queues_[index];
}

// Finish all queues
void CommandQueuePool::FinishAll() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_) return;
    
    for (auto queue : queues_) {
        clFinish(queue);
    }
}

// Flush all queues
void CommandQueuePool::FlushAll() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_) return;
    
    for (auto queue : queues_) {
        clFlush(queue);
    }
}

// Get pool size
size_t CommandQueuePool::GetPoolSize() {
    std::lock_guard<std::mutex> lock(mutex_);
    return (initialized_) ? queues_.size() : 0;
}

// Get current queue index
size_t CommandQueuePool::GetCurrentQueueIndex() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_ || queues_.empty()) return 0;
    // Return the last used index (current_index_ points to next)
    return (current_index_ == 0) ? (queues_.size() - 1) : (current_index_ - 1);
}

// Get statistics
std::string CommandQueuePool::GetStatistics() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::stringstream ss;
    ss << "CommandQueuePool Statistics\n";
    ss << "======================================================================\n";
    ss << "Number of queues: " << queues_.size() << "\n";
    
    size_t total_ops = 0;
    for (auto ops : queue_usage_) {
        total_ops += ops;
    }
    ss << "Total operations: " << total_ops << "\n";
    ss << "Load distribution:\n";
    
    for (size_t i = 0; i < queues_.size(); ++i) {
        double percent = (total_ops > 0) 
            ? (100.0 * queue_usage_[i] / total_ops) 
            : 0.0;
        ss << "  Queue[" << i << "]: " << queue_usage_[i] << " ops (" 
           << percent << "%)\n";
    }
    ss << "======================================================================\n";
    
    return ss.str();
}

// Create N queues
void CommandQueuePool::CreateQueues(size_t num_queues) {
    if (num_queues == 0) {
        num_queues = std::thread::hardware_concurrency();
        if (num_queues == 0) num_queues = 1;
    }
    
    std::cout << "[CommandQueuePool] Creating " << num_queues << " queues\n";
    
    auto& core = OpenCLCore::GetInstance();
    cl_context context = core.GetContext();
    cl_device_id device = core.GetDevice();
    
    for (size_t i = 0; i < num_queues; ++i) {
        cl_int err = CL_SUCCESS;
        cl_command_queue queue = clCreateCommandQueue(
            context,
            device,
            CL_QUEUE_PROFILING_ENABLE,
            &err
        );
        
        if (err != CL_SUCCESS) {
            throw std::runtime_error(
                "[CommandQueuePool] Failed to create command queue: " + 
                std::to_string(err)
            );
        }
        
        queues_.push_back(queue);
//        std::cout << "[CommandQueuePool] Created queue[" << i << "]\n";
    }
}

// Release all queues
void CommandQueuePool::ReleaseQueues() {
    for (auto queue : queues_) {
        if (queue) {
            cl_int err = clReleaseCommandQueue(queue);
            if (err != CL_SUCCESS) {
                std::cerr << "[CommandQueuePool] Warning: Failed to release queue\n";
            }
        }
    }
}

}  // namespace ManagerOpenCL