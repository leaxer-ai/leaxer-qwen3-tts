cmake_minimum_required(VERSION 3.14)
project(leaxer-qwen3-tts VERSION 0.2.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(LEAXER_QWEN_BUILD_TESTS "Build tests" ON)
option(LEAXER_QWEN_STATIC "Build static library" ON)

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
# Main executable (ONNX-based)
# =============================================================================

add_executable(leaxer-qwen3-tts-onnx src/main_onnx.cpp)
target_link_libraries(leaxer-qwen3-tts-onnx PRIVATE leaxer_qwen_lib)

# Version info
target_compile_definitions(leaxer-qwen3-tts-onnx PRIVATE
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
endif()

# =============================================================================
# Install
# =============================================================================

install(TARGETS leaxer-qwen3-tts-onnx DESTINATION bin)
install(TARGETS leaxer_qwen_lib DESTINATION lib)

# =============================================================================
# Summary
# =============================================================================

message(STATUS "")
message(STATUS "leaxer-qwen3-tts v${PROJECT_VERSION} (ONNX)")
message(STATUS "  Build tests:  ${LEAXER_QWEN_BUILD_TESTS}")
message(STATUS "  ONNX Runtime: ${ONNXRUNTIME_PREFIX}")
message(STATUS "")
