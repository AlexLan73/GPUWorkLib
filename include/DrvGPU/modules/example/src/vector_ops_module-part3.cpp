// ЧАСТЬ 3: CreateKernelObjects, ReleaseKernels, LoadKernelSource

void VectorOpsModule::CreateKernelObjects() {
    DRVGPU_LOG_INFO("VectorOpsModule", "Creating kernel objects...");
    
    cl_int err;
    
    kernel_add_one_out_ = clCreateKernel(program_, "vector_add_one_out", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create kernel: vector_add_one_out");
    }
    
    kernel_add_one_inplace_ = clCreateKernel(program_, "vector_add_one_inplace", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create kernel: vector_add_one_inplace");
    }
    
    kernel_sub_one_out_ = clCreateKernel(program_, "vector_sub_one_out", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create kernel: vector_sub_one_out");
    }
    
    kernel_sub_one_inplace_ = clCreateKernel(program_, "vector_sub_one_inplace", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create kernel: vector_sub_one_inplace");
    }
    
    kernel_add_vectors_out_ = clCreateKernel(program_, "vector_add_vectors_out", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create kernel: vector_add_vectors_out");
    }
    
    kernel_add_vectors_inplace_ = clCreateKernel(program_, "vector_add_vectors_inplace", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create kernel: vector_add_vectors_inplace");
    }
    
    DRVGPU_LOG_INFO("VectorOpsModule", "All 6 kernels created ✅");
}

void VectorOpsModule::ReleaseKernels() {
    if (kernel_add_one_out_) {
        clReleaseKernel(kernel_add_one_out_);
        kernel_add_one_out_ = nullptr;
    }
    if (kernel_add_one_inplace_) {
        clReleaseKernel(kernel_add_one_inplace_);
        kernel_add_one_inplace_ = nullptr;
    }
    if (kernel_sub_one_out_) {
        clReleaseKernel(kernel_sub_one_out_);
        kernel_sub_one_out_ = nullptr;
    }
    if (kernel_sub_one_inplace_) {
        clReleaseKernel(kernel_sub_one_inplace_);
        kernel_sub_one_inplace_ = nullptr;
    }
    if (kernel_add_vectors_out_) {
        clReleaseKernel(kernel_add_vectors_out_);
        kernel_add_vectors_out_ = nullptr;
    }
    if (kernel_add_vectors_inplace_) {
        clReleaseKernel(kernel_add_vectors_inplace_);
        kernel_add_vectors_inplace_ = nullptr;
    }
    
    if (program_) {
        clReleaseProgram(program_);
        program_ = nullptr;
    }
}

std::string VectorOpsModule::LoadKernelSource(const std::string& filename) {
    std::vector<std::string> search_paths = {
        std::string(VECTOR_OPS_KERNELS_PATH) + "/" + filename,
        "modules/example/kernels/" + filename,
        "../modules/example/kernels/" + filename,
        "../../modules/example/kernels/" + filename
    };
    
    for (const auto& path : search_paths) {
        std::ifstream file(path);
        if (file.is_open()) {
            DRVGPU_LOG_DEBUG("VectorOpsModule", "Kernel loaded from: " + path);
            std::stringstream buffer;
            buffer << file.rdbuf();
            return buffer.str();
        }
    }
    
    DRVGPU_LOG_ERROR("VectorOpsModule", "Failed to load kernel: " + filename);
    DRVGPU_LOG_ERROR("VectorOpsModule", "Tried paths:");
    for (const auto& path : search_paths) {
        DRVGPU_LOG_ERROR("VectorOpsModule", "  - " + path);
    }
    
    throw std::runtime_error("VectorOpsModule: Failed to load kernel source: " + filename);
}
