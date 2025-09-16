cmake_minimum_required(VERSION 3.18)

project(VisionGuideNative LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wextra -frtti -fexceptions")

set(LLAMA_CPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp)

# CRITICAL: Enable CPU backend explicitly
add_definitions(-DGGML_USE_CPU)

# Enable NEON backend for ARM
if(ANDROID)
    add_definitions(-DGGML_USE_ACCEL=NEON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -funroll-loops")
endif()
#add_definitions(-DGGML_USE_ACCEL=NEON)

include_directories(
        "${LLAMA_CPP_DIR}"
        "${LLAMA_CPP_DIR}/src"
        "${LLAMA_CPP_DIR}/ggml/src"
        "${LLAMA_CPP_DIR}/ggml/include"
        "${LLAMA_CPP_DIR}/include"
        "${LLAMA_CPP_DIR}/common"
        "${LLAMA_CPP_DIR}/ggml/src/ggml-cpu"
)

################# llama library
file(GLOB LLAMA_SRC
        "${LLAMA_CPP_DIR}/src/*.cpp"
        "${LLAMA_CPP_DIR}/common/*.cpp"
)

# append build_info.cpp explicitly
list(APPEND LLAMA_SRC "${CMAKE_CURRENT_SOURCE_DIR}/build_info.cpp")

add_library(llama STATIC ${LLAMA_SRC})

target_compile_definitions(llama PRIVATE GGML_USE_CPU)
if(ANDROID)
    target_compile_definitions(llama PRIVATE GGML_USE_NEON)
endif()

# Add non-finite math flag
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(llama PRIVATE -fno-finite-math-only)
endif()


target_link_libraries(llama PRIVATE ggml-base c++_shared)

#################### ggml-base library
# Gather all CPU-related sources
file(GLOB GGML_SRC
        "${LLAMA_CPP_DIR}/ggml/src/*.c"
        "${LLAMA_CPP_DIR}/ggml/src/*.cpp"
        "${LLAMA_CPP_DIR}/ggml/src/ggml-cpu/*.c"
        "${LLAMA_CPP_DIR}/ggml/src/ggml-cpu/*.cpp"
)

# Exclude OpenCL and CANN backends (not needed on Android)
list(FILTER GGML_SRC EXCLUDE REGEX ".*/ggml-opencl/.*")
list(FILTER GGML_SRC EXCLUDE REGEX ".*/ggml-cann/.*")

# Include CPU backend source
set(CPU_BACKEND_FILE "${LLAMA_CPP_DIR}/ggml/src/ggml-cpu/ggml-cpu.c")
if(EXISTS ${CPU_BACKEND_FILE})
    list(APPEND GGML_SRC ${CPU_BACKEND_FILE})
    message(STATUS "Including CPU backend: ${CPU_BACKEND_FILE}")
else()
    message(FATAL_ERROR "Could not find CPU backend source file containing ggml_backend_cpu_reg")
endif()

set(CPU_BACKEND_FIL "${LLAMA_CPP_DIR}/ggml/src/ggml-cpu/ggml-cpu.cpp")
if(EXISTS ${CPU_BACKEND_FIL})
    list(APPEND GGML_SRC ${CPU_BACKEND_FIL})
    message(STATUS "Including CPU backend2: ${CPU_BACKEND_FIL}")
else()
    message(FATAL_ERROR "Could not find CPU backend source file2 containing ggml_backend_cpu_reg")
endif()

# Include backend registration file
set(BACKEND_REG_FILE "${LLAMA_CPP_DIR}/ggml/src/ggml-backend-reg.cpp")
if(EXISTS ${BACKEND_REG_FILE})
    list(APPEND GGML_SRC ${BACKEND_REG_FILE})
    message(STATUS "Including backend registration: ${BACKEND_REG_FILE}")
else()
    message(FATAL_ERROR "Could not find ggml-backend-reg.cpp")
endif()

# Create static library
add_library(ggml-base STATIC ${GGML_SRC})

# Definitions
target_compile_definitions(ggml-base PRIVATE GGML_USE_CPU)
if(ANDROID)
    target_compile_definitions(ggml-base PRIVATE GGML_USE_NEON)
endif()

# Add non-finite math flag
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(ggml-base PRIVATE -fno-finite-math-only)
endif()

# Link Android libraries
target_link_libraries(ggml-base PUBLIC log android c++_shared)

# JNI library
add_library(llama-jni SHARED llama_jni.cpp)
target_link_libraries(llama-jni PRIVATE ggml-base llama log android c++_shared)

