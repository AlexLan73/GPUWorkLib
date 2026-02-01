#include "command_queue_pool.hpp"
#include "opencl_core.hpp"
#include "../../common/logger.hpp"
#include <iostream>

namespace drv_gpu_lib {

CommandQueuePool::CommandQueuePool()
    : context_(nullptr),
      device_(nullptr),
      initialized_(false) {
}

CommandQueuePool::~CommandQueuePool() {
    Cleanup();
}

bool CommandQueuePool::Initialize(cl_context context, cl_device_id device, size_t num_queues) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (initialized_) {
        Cleanup();
    }
    
    context_ = context;
    device_ = device;
    
    if (num_queues == 0) {
        num_queues = 2;  // Default to 2 queues
    }
    
    cl_int err;
    for (size_t i = 0; i < num_queues; ++i) {
        cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
        if (err != CL_SUCCESS) {
            DRVGPU_LOG_ERROR("CommandQueuePool", "Failed to create command queue: " + std::to_string(err));
            continue;
        }
        queues_.push_back(queue);
    }
    
    initialized_ = !queues_.empty();
    return initialized_;
}

void CommandQueuePool::Cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& queue : queues_) {
        if (queue) {
            clReleaseCommandQueue(queue);
        }
    }
    queues_.clear();
    initialized_ = false;
}

cl_command_queue CommandQueuePool::GetQueue(size_t index) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queues_.empty()) {
        return nullptr;
    }
    return queues_[index % queues_.size()];
}

size_t CommandQueuePool::GetQueueCount() const {
    return queues_.size();
}

void CommandQueuePool::Synchronize() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& queue : queues_) {
        if (queue) {
            clFinish(queue);
        }
    }
}

} // namespace ManagerOpenCL
