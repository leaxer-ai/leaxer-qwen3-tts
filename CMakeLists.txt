cmake_minimum_required(VERSION 3.14)
project(leaxer-qwen3-tts VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(LEAXER_QWEN_BUILD_TESTS "Build tests" ON)
option(LEAXER_QWEN_STATIC "Build static library" ON)
option(LEAXER_QWEN_USE_ONNX "Build with ONNX Runtime support" ON)

# ggml submodule
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/extern/ggml/CMakeLists.txt")
    message(STATUS "ggml submodule not found, will need to initialize:")
    message(STATUS "  git submodule add https://github.com/ggerganov/ggml extern/ggml")
    message(STATUS "  git submodule update --init --recursive")
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/extern/ggml/CMakeLists.txt")
    add_subdirectory(extern/ggml)
else()
    message(WARNING "ggml not found - build will fail until submodule is initialized")
endif()

# ONNX Runtime support
if(LEAXER_QWEN_USE_ONNX)
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
        message(WARNING "ONNX Runtime not found. Install via: brew install onnxruntime")
        set(LEAXER_QWEN_USE_ONNX OFF)
    endif()
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# Add ONNX Runtime include paths if found
if(LEAXER_QWEN_USE_ONNX AND ONNXRUNTIME_FOUND)
    include_directories(${ONNXRUNTIME_INCLUDE_DIRS})
endif()

# Source files - ggml ops
set(GGML_OPS_SOURCES
    src/ggml_ops/snake_beta.cpp
    src/ggml_ops/rms_norm.cpp
    src/ggml_ops/rope.cpp
    src/ggml_ops/conv1d.cpp
)

# Source files - vocoder
set(VOCODER_SOURCES
    src/vocoder/split_rvq.cpp
    src/vocoder/causal_conv.cpp
    src/vocoder/convnext.cpp
    src/vocoder/upsample.cpp
    src/vocoder/vocoder.cpp
)

# Source files - model
set(MODEL_SOURCES
    src/model/attention.cpp
    src/model/ffn.cpp
    src/model/transformer.cpp
    src/model/transformer_cached.cpp
    src/model/code_predictor.cpp
    src/model/qwen_tts.cpp
    src/model/qwen_tts_interleaved.cpp
)

# Source files - io
set(IO_SOURCES
    src/io/gguf_loader.cpp
    src/io/tokenizer.cpp
    src/io/wav_writer.cpp
)

# Source files - onnx (conditional)
if(LEAXER_QWEN_USE_ONNX AND ONNXRUNTIME_FOUND)
    set(ONNX_SOURCES
        src/onnx/onnx_session.cpp
        src/tts_onnx.cpp
    )
else()
    set(ONNX_SOURCES "")
endif()

# All library sources
set(LIB_SOURCES
    ${GGML_OPS_SOURCES}
    ${VOCODER_SOURCES}
    ${MODEL_SOURCES}
    ${IO_SOURCES}
    ${ONNX_SOURCES}
)

# Library
add_library(leaxer_qwen_lib ${LIB_SOURCES})
target_include_directories(leaxer_qwen_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
if(TARGET ggml)
    target_link_libraries(leaxer_qwen_lib PUBLIC ggml)
endif()

# Link ONNX Runtime if available
if(LEAXER_QWEN_USE_ONNX AND ONNXRUNTIME_FOUND)
    target_include_directories(leaxer_qwen_lib PUBLIC ${ONNXRUNTIME_INCLUDE_DIRS})
    if(ONNXRUNTIME_LIBRARY)
        target_link_libraries(leaxer_qwen_lib PUBLIC ${ONNXRUNTIME_LIBRARY})
    else()
        target_link_directories(leaxer_qwen_lib PUBLIC ${ONNXRUNTIME_LIBRARY_DIRS})
        target_link_libraries(leaxer_qwen_lib PUBLIC onnxruntime)
    endif()
    target_compile_definitions(leaxer_qwen_lib PUBLIC LEAXER_USE_ONNX=1)
endif()

# Main executable (GGUF-based)
add_executable(leaxer-qwen3-tts src/leaxer_qwen.cpp)
target_link_libraries(leaxer-qwen3-tts PRIVATE leaxer_qwen_lib)

# Version info
target_compile_definitions(leaxer-qwen3-tts PRIVATE
    LEAXER_QWEN_VERSION="${PROJECT_VERSION}"
)

# ONNX-based TTS executable (if ONNX support is enabled)
if(LEAXER_QWEN_USE_ONNX AND ONNXRUNTIME_FOUND)
    add_executable(leaxer-qwen3-tts-onnx src/main_onnx.cpp)
    target_link_libraries(leaxer-qwen3-tts-onnx PRIVATE leaxer_qwen_lib)
    target_compile_definitions(leaxer-qwen3-tts-onnx PRIVATE
        LEAXER_QWEN_VERSION="${PROJECT_VERSION}"
    )
    install(TARGETS leaxer-qwen3-tts-onnx DESTINATION bin)
endif()

# Tests
if(LEAXER_QWEN_BUILD_TESTS)
    enable_testing()

    # Test utilities (header-only for now)

    # Component tests
    add_executable(test_snakebeta tests/test_snakebeta.cpp)
    target_link_libraries(test_snakebeta PRIVATE leaxer_qwen_lib)
    add_test(NAME snakebeta COMMAND test_snakebeta)

    # WAV writer test
    add_executable(test_wav tests/test_wav.cpp)
    target_link_libraries(test_wav PRIVATE leaxer_qwen_lib)
    add_test(NAME wav COMMAND test_wav)

    # RMSNorm test
    add_executable(test_rmsnorm tests/test_rmsnorm.cpp)
    target_link_libraries(test_rmsnorm PRIVATE leaxer_qwen_lib)
    add_test(NAME rmsnorm COMMAND test_rmsnorm)

    # Conv1d test
    add_executable(test_conv1d tests/test_conv1d.cpp)
    target_link_libraries(test_conv1d PRIVATE leaxer_qwen_lib)
    add_test(NAME conv1d COMMAND test_conv1d)

    # Tokenizer test
    add_executable(test_tokenizer tests/test_tokenizer.cpp)
    target_link_libraries(test_tokenizer PRIVATE leaxer_qwen_lib)
    add_test(NAME tokenizer COMMAND test_tokenizer)

    # Tokenizer real test (against Python fixtures)
    add_executable(test_tokenizer_real tests/test_tokenizer_real.cpp)
    target_link_libraries(test_tokenizer_real PRIVATE leaxer_qwen_lib)
    add_test(NAME tokenizer_real COMMAND test_tokenizer_real)

    # GGUF mapping test
    add_executable(test_gguf_mapping tests/test_gguf_mapping.cpp)
    target_link_libraries(test_gguf_mapping PRIVATE leaxer_qwen_lib)
    add_test(NAME gguf_mapping COMMAND test_gguf_mapping)

    # End-to-end test
    add_executable(test_e2e tests/test_e2e.cpp)
    target_link_libraries(test_e2e PRIVATE leaxer_qwen_lib)
    add_test(NAME e2e COMMAND test_e2e)

    # Real model test
    add_executable(test_real_model tests/test_real_model.cpp)
    target_link_libraries(test_real_model PRIVATE leaxer_qwen_lib)
    add_test(NAME real_model COMMAND test_real_model)

    # Speaker embedding test
    add_executable(test_speaker_embedding tests/test_speaker_embedding.cpp)
    target_link_libraries(test_speaker_embedding PRIVATE leaxer_qwen_lib)
    add_test(NAME speaker_embedding COMMAND test_speaker_embedding)

    # Audio quality verification test
    add_executable(test_audio_quality tests/test_audio_quality.cpp)
    target_link_libraries(test_audio_quality PRIVATE leaxer_qwen_lib)
    add_test(NAME audio_quality COMMAND test_audio_quality)

    # ONNX Runtime test (only if ONNX support is enabled)
    if(LEAXER_QWEN_USE_ONNX AND ONNXRUNTIME_FOUND)
        add_executable(test_onnx tests/test_onnx.cpp)
        target_link_libraries(test_onnx PRIVATE leaxer_qwen_lib)
        add_test(NAME onnx COMMAND test_onnx)
    endif()
endif()

# Install
install(TARGETS leaxer-qwen3-tts DESTINATION bin)
install(TARGETS leaxer_qwen_lib DESTINATION lib)
install(FILES include/leaxer_qwen.h DESTINATION include)

# Summary
message(STATUS "")
message(STATUS "leaxer-qwen3-tts v${PROJECT_VERSION}")
message(STATUS "  Build tests:  ${LEAXER_QWEN_BUILD_TESTS}")
message(STATUS "  Static lib:   ${LEAXER_QWEN_STATIC}")
if(LEAXER_QWEN_USE_ONNX AND ONNXRUNTIME_FOUND)
    message(STATUS "  ONNX Runtime: ON")
else()
    message(STATUS "  ONNX Runtime: OFF")
endif()
message(STATUS "")
