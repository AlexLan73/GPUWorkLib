#pragma once

/**
 * @file command_queue_pool.hpp
 * @brief Pool of OpenCL command queues
 */

#include <CL/cl.h>

#include <memory>
#include <vector>
#include <mutex>

namespace drv_gpu_lib {

class CommandQueuePool {
public:
    CommandQueuePool();
    ~CommandQueuePool();
    
    bool Initialize(cl_context context, cl_device_id device, size_t num_queues = 0);
    void Cleanup();
    
    cl_command_queue GetQueue(size_t index = 0);
    size_t GetQueueCount() const;
    
    void Synchronize();
    
private:
    std::vector<cl_command_queue> queues_;
    cl_context context_;
    cl_device_id device_;
    bool initialized_;
    mutable std::mutex mutex_;
};

} // namespace drv_gpu_lib
