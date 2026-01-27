cmake_minimum_required(VERSION 3.14)
project(leaxer-qwen VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(LEAXER_QWEN_BUILD_TESTS "Build tests" ON)
option(LEAXER_QWEN_STATIC "Build static library" ON)

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

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

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
    src/model/code_predictor.cpp
    src/model/qwen_tts.cpp
)

# Source files - io
set(IO_SOURCES
    src/io/gguf_loader.cpp
    src/io/tokenizer.cpp
    src/io/wav_writer.cpp
)

# All library sources
set(LIB_SOURCES
    ${GGML_OPS_SOURCES}
    ${VOCODER_SOURCES}
    ${MODEL_SOURCES}
    ${IO_SOURCES}
)

# Library
add_library(leaxer_qwen_lib ${LIB_SOURCES})
target_include_directories(leaxer_qwen_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
if(TARGET ggml)
    target_link_libraries(leaxer_qwen_lib PUBLIC ggml)
endif()

# Main executable
add_executable(leaxer-qwen src/leaxer_qwen.cpp)
target_link_libraries(leaxer-qwen PRIVATE leaxer_qwen_lib)

# Version info
target_compile_definitions(leaxer-qwen PRIVATE
    LEAXER_QWEN_VERSION="${PROJECT_VERSION}"
)

# Tests
if(LEAXER_QWEN_BUILD_TESTS)
    enable_testing()

    # Test utilities (header-only for now)
    # Tests will be added as components are implemented

    # Placeholder test
    add_executable(test_placeholder tests/test_placeholder.cpp)
    target_link_libraries(test_placeholder PRIVATE leaxer_qwen_lib)
    add_test(NAME placeholder COMMAND test_placeholder)

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

    # End-to-end test
    add_executable(test_e2e tests/test_e2e.cpp)
    target_link_libraries(test_e2e PRIVATE leaxer_qwen_lib)
    add_test(NAME e2e COMMAND test_e2e)
endif()

# Install
install(TARGETS leaxer-qwen DESTINATION bin)
install(TARGETS leaxer_qwen_lib DESTINATION lib)
install(FILES include/leaxer_qwen.h DESTINATION include)

# Summary
message(STATUS "")
message(STATUS "leaxer-qwen v${PROJECT_VERSION}")
message(STATUS "  Build tests: ${LEAXER_QWEN_BUILD_TESTS}")
message(STATUS "  Static lib:  ${LEAXER_QWEN_STATIC}")
message(STATUS "")
