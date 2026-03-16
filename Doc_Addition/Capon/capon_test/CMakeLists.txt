# CMakeList.txt: проект CMake для OpenCL_; включите исходный код и определения,
# укажите здесь логику для конкретного проекта.
#
cmake_minimum_required (VERSION 3.8)

# Включение горячей перезагрузки для компиляторов MSVC, если поддерживается.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("capon_test")

set(ArrayFire_DIR "/home/user/agapov/ArrayFire-3.10.0-Linux/share/ArrayFire/cmake")
find_package(ArrayFire CONFIG REQUIRED)
include_directories(include)


set(SOURCES
    src/capon_relief.cpp  
    src/main.cpp  
    src/mesh_grid.cpp  
    src/threshold_processing.cpp
    src/inv_schulz.cpp
)
set(HEADERS
    include/capon_relief.h
    include/mesh_grid.h
    include/threshold_processing.h
    include/inv_schulz.h
)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})


#target_link_libraries(OpenCL_ ArrayFire::afopencl Python3::Python Python3::NumPy)

target_link_libraries(${PROJECT_NAME} ArrayFire::afopencl)
#target_link_libraries(OpenCL_ ArrayFire::afcuda)
#target_link_libraries(OpenCL_ ArrayFire::afcpu)


if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
endif()




