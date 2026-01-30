cmake_minimum_required(VERSION 3.14)
project(leaxer-qwen3-tts VERSION 0.2.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(LEAXER_QWEN_BUILD_TESTS "Build tests" ON)

# GPU Acceleration (requires ONNX Runtime built with these providers)
option(LEAXER_COREML "Enable CoreML Execution Provider (macOS)" OFF)
option(LEAXER_CUDA "Enable CUDA Execution Provider (NVIDIA)" OFF)
option(LEAXER_ROCM "Enable ROCm Execution Provider (AMD)" OFF)
option(LEAXER_DIRECTML "Enable DirectML Execution Provider (Windows)" OFF)

# =============================================================================
# ONNX Runtime (required)
# User must provide: -DONNXRUNTIME_DIR=/path/to/onnxruntime
# Or have it installed in standard system paths
# =============================================================================

if(ONNXRUNTIME_DIR)
    set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_DIR}/include")
    find_library(ONNXRUNTIME_LIBRARY 
        NAMES onnxruntime
        PATHS "${ONNXRUNTIME_DIR}/lib"
        NO_DEFAULT_PATH
        REQUIRED
    )
    message(STATUS "Using ONNX Runtime from: ${ONNXRUNTIME_DIR}")
else()
    # Try standard paths
    find_path(ONNXRUNTIME_INCLUDE_DIRS 
        NAMES onnxruntime_cxx_api.h onnxruntime/onnxruntime_cxx_api.h
        PATHS /usr/local/include /usr/include /opt/homebrew/include
    )
    find_library(ONNXRUNTIME_LIBRARY 
        NAMES onnxruntime
        PATHS /usr/local/lib /usr/lib /opt/homebrew/lib
    )
endif()

if(NOT ONNXRUNTIME_INCLUDE_DIRS OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "ONNX Runtime not found. Specify -DONNXRUNTIME_DIR=/path/to/onnxruntime")
endif()

message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIRS}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")

# =============================================================================
# Sources
# =============================================================================

set(LIB_SOURCES
    src/io/tokenizer.cpp
    src/io/wav_writer.cpp
    src/io/wav_reader.cpp
    src/io/mel.cpp
    src/tts_onnx.cpp
)

# =============================================================================
# Library
# =============================================================================

add_library(leaxer_qwen_lib ${LIB_SOURCES})
target_include_directories(leaxer_qwen_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${ONNXRUNTIME_INCLUDE_DIRS}
)
target_link_libraries(leaxer_qwen_lib PUBLIC ${ONNXRUNTIME_LIBRARY})
target_compile_definitions(leaxer_qwen_lib PUBLIC LEAXER_USE_ONNX=1)

# GPU providers (compile definitions only - runtime checks if available)
if(LEAXER_COREML AND APPLE)
    find_file(COREML_HEADER coreml_provider_factory.h
        PATHS ${ONNXRUNTIME_INCLUDE_DIRS} ${ONNXRUNTIME_INCLUDE_DIRS}/onnxruntime
        NO_DEFAULT_PATH
    )
    if(COREML_HEADER)
        target_compile_definitions(leaxer_qwen_lib PUBLIC LEAXER_USE_COREML=1)
        target_link_libraries(leaxer_qwen_lib PUBLIC "-framework CoreML" "-framework Foundation")
    else()
        message(WARNING "CoreML requested but coreml_provider_factory.h not found")
    endif()
endif()

if(LEAXER_CUDA)
    target_compile_definitions(leaxer_qwen_lib PUBLIC LEAXER_USE_CUDA=1)
endif()

if(LEAXER_ROCM)
    target_compile_definitions(leaxer_qwen_lib PUBLIC LEAXER_USE_ROCM=1)
endif()

if(LEAXER_DIRECTML AND WIN32)
    target_compile_definitions(leaxer_qwen_lib PUBLIC LEAXER_USE_DIRECTML=1)
endif()

# =============================================================================
# Executable
# =============================================================================

add_executable(leaxer-qwen3-tts src/main_onnx.cpp)
target_link_libraries(leaxer-qwen3-tts PRIVATE leaxer_qwen_lib)
target_compile_definitions(leaxer-qwen3-tts PRIVATE LEAXER_QWEN_VERSION="${PROJECT_VERSION}")

# =============================================================================
# Tests
# =============================================================================

if(LEAXER_QWEN_BUILD_TESTS)
    enable_testing()
    
    add_executable(test_tokenizer tests/test_tokenizer.cpp)
    target_link_libraries(test_tokenizer PRIVATE leaxer_qwen_lib)
    add_test(NAME tokenizer COMMAND test_tokenizer)

    add_executable(test_tokenizer_real tests/test_tokenizer_real.cpp)
    target_link_libraries(test_tokenizer_real PRIVATE leaxer_qwen_lib)
    add_test(NAME tokenizer_real COMMAND test_tokenizer_real)

    add_executable(test_onnx tests/test_onnx.cpp)
    target_link_libraries(test_onnx PRIVATE leaxer_qwen_lib)
    add_test(NAME onnx COMMAND test_onnx)

    add_executable(test_wav_reader tests/test_wav_reader.cpp)
    target_link_libraries(test_wav_reader PRIVATE leaxer_qwen_lib)
    add_test(NAME wav_reader COMMAND test_wav_reader)

    add_executable(test_mel tests/test_mel.cpp)
    target_link_libraries(test_mel PRIVATE leaxer_qwen_lib)
    add_test(NAME mel COMMAND test_mel)
endif()

# =============================================================================
# Install
# =============================================================================

install(TARGETS leaxer-qwen3-tts DESTINATION bin)
install(TARGETS leaxer_qwen_lib DESTINATION lib)

# =============================================================================
# Summary
# =============================================================================

message(STATUS "")
message(STATUS "leaxer-qwen3-tts v${PROJECT_VERSION}")
message(STATUS "  Tests:    ${LEAXER_QWEN_BUILD_TESTS}")
message(STATUS "  CoreML:   ${LEAXER_COREML}")
message(STATUS "  CUDA:     ${LEAXER_CUDA}")
message(STATUS "  ROCm:     ${LEAXER_ROCM}")
message(STATUS "  DirectML: ${LEAXER_DIRECTML}")
message(STATUS "")
