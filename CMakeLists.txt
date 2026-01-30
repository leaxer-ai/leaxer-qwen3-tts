cmake_minimum_required(VERSION 3.14)
project(leaxer-qwen3-tts VERSION 0.2.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(LEAXER_QWEN_BUILD_TESTS "Build tests" ON)
option(LEAXER_QWEN_STATIC "Build static library" ON)

# =============================================================================
# GPU Acceleration Options
# =============================================================================

# CoreML (Apple Silicon only)
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    option(LEAXER_COREML "Enable CoreML Execution Provider (Apple GPU/ANE)" ON)
else()
    option(LEAXER_COREML "Enable CoreML Execution Provider (Apple GPU/ANE)" OFF)
endif()

# CUDA (NVIDIA GPUs)
option(LEAXER_CUDA "Enable CUDA Execution Provider (NVIDIA GPU)" OFF)

# ROCm (AMD GPUs) 
option(LEAXER_ROCM "Enable ROCm Execution Provider (AMD GPU)" OFF)

# DirectML (Windows GPU)
if(WIN32)
    option(LEAXER_DIRECTML "Enable DirectML Execution Provider (Windows GPU)" OFF)
else()
    option(LEAXER_DIRECTML "Enable DirectML Execution Provider (Windows GPU)" OFF)
endif()

# Static linking
option(LEAXER_STATIC "Build with static linking where possible" OFF)

# =============================================================================
# ONNX Runtime (required for this branch)
# =============================================================================

# Allow user to specify ONNX Runtime paths directly (for CI/bundling)
if(ONNXRUNTIME_INCLUDE_DIRS AND ONNXRUNTIME_LIBRARY)
    set(ONNXRUNTIME_FOUND TRUE)
    message(STATUS "Using provided ONNX Runtime: ${ONNXRUNTIME_LIBRARY}")
else()
    # Try to find onnxruntime via pkg-config first
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(ONNXRUNTIME QUIET libonnxruntime)
    endif()

    if(ONNXRUNTIME_FOUND)
        message(STATUS "Found ONNX Runtime via pkg-config: ${ONNXRUNTIME_VERSION}")
    else()
        # Try homebrew path (macOS)
        if(APPLE)
            execute_process(
                COMMAND brew --prefix onnxruntime
                OUTPUT_VARIABLE ONNXRUNTIME_PREFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )
            if(ONNXRUNTIME_PREFIX)
                set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_PREFIX}/include")
                set(ONNXRUNTIME_LIBRARY_DIRS "${ONNXRUNTIME_PREFIX}/lib")
                find_library(ONNXRUNTIME_LIBRARY onnxruntime PATHS ${ONNXRUNTIME_LIBRARY_DIRS} NO_DEFAULT_PATH)
                if(ONNXRUNTIME_LIBRARY)
                    set(ONNXRUNTIME_FOUND TRUE)
                    message(STATUS "Found ONNX Runtime via Homebrew: ${ONNXRUNTIME_PREFIX}")
                endif()
            endif()
        endif()
        
        # Try standard system paths
        if(NOT ONNXRUNTIME_FOUND)
            find_path(ONNXRUNTIME_INCLUDE_DIRS onnxruntime/onnxruntime_cxx_api.h
                PATHS /usr/local/include /usr/include
            )
            find_library(ONNXRUNTIME_LIBRARY onnxruntime
                PATHS /usr/local/lib /usr/lib
            )
            if(ONNXRUNTIME_INCLUDE_DIRS AND ONNXRUNTIME_LIBRARY)
                set(ONNXRUNTIME_FOUND TRUE)
                message(STATUS "Found ONNX Runtime in system paths")
            endif()
        endif()
    endif()

    if(NOT ONNXRUNTIME_FOUND)
        message(FATAL_ERROR "ONNX Runtime not found. Install via: brew install onnxruntime")
    endif()
endif()

# =============================================================================
# Include directories
# =============================================================================

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

# =============================================================================
# Source files
# =============================================================================

# IO sources (tokenizer, wav reader/writer, mel spectrogram)
set(IO_SOURCES
    src/io/tokenizer.cpp
    src/io/wav_writer.cpp
    src/io/wav_reader.cpp
    src/io/mel.cpp
)

# TTS engine sources
set(TTS_SOURCES
    src/tts_onnx.cpp
)

# All library sources
set(LIB_SOURCES
    ${IO_SOURCES}
    ${TTS_SOURCES}
)

# =============================================================================
# Library
# =============================================================================

add_library(leaxer_qwen_lib ${LIB_SOURCES})
target_include_directories(leaxer_qwen_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

# Link ONNX Runtime
if(ONNXRUNTIME_LIBRARY)
    target_link_libraries(leaxer_qwen_lib PUBLIC ${ONNXRUNTIME_LIBRARY})
else()
    target_link_directories(leaxer_qwen_lib PUBLIC ${ONNXRUNTIME_LIBRARY_DIRS})
    target_link_libraries(leaxer_qwen_lib PUBLIC onnxruntime)
endif()

target_compile_definitions(leaxer_qwen_lib PUBLIC LEAXER_USE_ONNX=1)

# =============================================================================
# GPU Execution Provider Support
# =============================================================================

# CoreML (Apple)
if(LEAXER_COREML AND APPLE)
    # Check if CoreML provider header exists (not all ONNX Runtime builds have it)
    find_file(COREML_PROVIDER_HEADER coreml_provider_factory.h
        PATHS ${ONNXRUNTIME_INCLUDE_DIRS} ${ONNXRUNTIME_INCLUDE_DIRS}/onnxruntime
        NO_DEFAULT_PATH)
    if(COREML_PROVIDER_HEADER)
        target_compile_definitions(leaxer_qwen_lib PUBLIC LEAXER_USE_COREML=1)
        target_link_libraries(leaxer_qwen_lib PUBLIC
            "-framework CoreML"
            "-framework Foundation"
        )
        message(STATUS "CoreML provider header found: ${COREML_PROVIDER_HEADER}")
    else()
        message(STATUS "CoreML provider header NOT found - CoreML disabled")
        set(LEAXER_COREML OFF)
    endif()
endif()

# CUDA (NVIDIA)
if(LEAXER_CUDA)
    target_compile_definitions(leaxer_qwen_lib PUBLIC LEAXER_USE_CUDA=1)
    # Note: CUDA EP is loaded dynamically by ONNX Runtime if available
    # No extra linking needed - just needs onnxruntime built with CUDA
endif()

# ROCm (AMD)
if(LEAXER_ROCM)
    target_compile_definitions(leaxer_qwen_lib PUBLIC LEAXER_USE_ROCM=1)
    # Note: ROCm EP is loaded dynamically by ONNX Runtime if available
endif()

# DirectML (Windows)
if(LEAXER_DIRECTML AND WIN32)
    target_compile_definitions(leaxer_qwen_lib PUBLIC LEAXER_USE_DIRECTML=1)
endif()

# Static linking configuration
if(LEAXER_STATIC)
    if(MSVC)
        # Use static CRT on Windows
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    elseif(NOT APPLE)
        # Linux: add -static flag (doesn't work on macOS)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")
    endif()
endif()

# =============================================================================
# Main executable (ONNX-based)
# =============================================================================

add_executable(leaxer-qwen3-tts src/main_onnx.cpp)
target_link_libraries(leaxer-qwen3-tts PRIVATE leaxer_qwen_lib)

# Version info
target_compile_definitions(leaxer-qwen3-tts PRIVATE
    LEAXER_QWEN_VERSION="${PROJECT_VERSION}"
)

# =============================================================================
# Tests
# =============================================================================

if(LEAXER_QWEN_BUILD_TESTS)
    enable_testing()

    # Tokenizer test
    add_executable(test_tokenizer tests/test_tokenizer.cpp)
    target_link_libraries(test_tokenizer PRIVATE leaxer_qwen_lib)
    add_test(NAME tokenizer COMMAND test_tokenizer)

    # Tokenizer real test (against Python fixtures)
    add_executable(test_tokenizer_real tests/test_tokenizer_real.cpp)
    target_link_libraries(test_tokenizer_real PRIVATE leaxer_qwen_lib)
    add_test(NAME tokenizer_real COMMAND test_tokenizer_real)

    # ONNX test
    add_executable(test_onnx tests/test_onnx.cpp)
    target_link_libraries(test_onnx PRIVATE leaxer_qwen_lib)
    add_test(NAME onnx COMMAND test_onnx)

    # WAV reader test
    add_executable(test_wav_reader tests/test_wav_reader.cpp)
    target_link_libraries(test_wav_reader PRIVATE leaxer_qwen_lib)
    add_test(NAME wav_reader COMMAND test_wav_reader)

    # Mel spectrogram test
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
message(STATUS "========================================")
message(STATUS "  Build tests:   ${LEAXER_QWEN_BUILD_TESTS}")
message(STATUS "  Static build:  ${LEAXER_STATIC}")
message(STATUS "  ONNX Runtime:  ${ONNXRUNTIME_PREFIX}")
message(STATUS "")
message(STATUS "  GPU Acceleration:")
message(STATUS "    CoreML (Apple):   ${LEAXER_COREML}")
message(STATUS "    CUDA (NVIDIA):    ${LEAXER_CUDA}")
message(STATUS "    ROCm (AMD):       ${LEAXER_ROCM}")
message(STATUS "    DirectML (Win):   ${LEAXER_DIRECTML}")
message(STATUS "")
