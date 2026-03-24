cmake_minimum_required(VERSION 3.20)
project(GetGPU_and_Mellanox CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── ROCm / HIP (optional) ────────────────────────────────────────────────────
# Searches standard ROCm install paths.  Pass -DCMAKE_PREFIX_PATH=/opt/rocm
# if your ROCm is not in /opt/rocm or /opt/rocm-X.Y.Z.
find_package(hip QUIET
  HINTS
    /opt/rocm
    /opt/rocm-7.2.0
    /opt/rocm-7.1.0
    /opt/rocm-6.4.0
    /opt/rocm-6.3.0
    /opt/rocm-6.2.0
    /opt/rocm-6.1.0
    /opt/rocm-6.0.0
    $ENV{ROCM_PATH}
)

# ── Executable ───────────────────────────────────────────────────────────────
add_executable(get_gpu_mellanox
  main.cpp
  # gpu_mellanox_detector.hpp is header-only — no separate .cpp needed
)

target_include_directories(get_gpu_mellanox PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# ── Link HIP if found ────────────────────────────────────────────────────────
if(hip_FOUND)
  message(STATUS "[GetGPU_and_Mellanox] ROCm HIP found — ENABLE_ROCM=1, hip_id column active")
  target_compile_definitions(get_gpu_mellanox PRIVATE ENABLE_ROCM=1)
  target_link_libraries(get_gpu_mellanox PRIVATE hip::host)
else()
  message(STATUS "[GetGPU_and_Mellanox] ROCm HIP NOT found — building without HIP (hip_id will be -1)")
  message(STATUS "  To enable: cmake .. -DCMAKE_PREFIX_PATH=/opt/rocm")
endif()

# ── Install (optional, handy for server deployment) ──────────────────────────
install(TARGETS get_gpu_mellanox RUNTIME DESTINATION bin)
