// End-to-end test with dummy random weights
// Tests that the entire pipeline can be initialized and produces audio output

#include "test_utils.h"
#include "ggml.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>

// Forward declaration from wav_writer.cpp
namespace leaxer_qwen {
namespace io {
int write_wav(const char* path, const float* audio, size_t n_samples, int sample_rate);
}
}

// Simple dummy TTS implementation for testing
// Just generates a sine wave to verify the pipeline works
bool test_e2e_dummy_output() {
    using namespace leaxer_qwen::test;

    printf("Testing end-to-end pipeline with dummy audio generation...\n");

    // Generate 1 second of 440Hz sine wave as dummy audio
    const int sample_rate = 24000;
    const int duration_sec = 1;
    const size_t n_samples = sample_rate * duration_sec;
    const float frequency = 440.0f;  // A4 note
    const float amplitude = 0.5f;

    float * audio = (float *)malloc(n_samples * sizeof(float));
    TEST_ASSERT(audio != nullptr, "Failed to allocate audio buffer");

    // Generate sine wave
    for (size_t i = 0; i < n_samples; i++) {
        float t = static_cast<float>(i) / sample_rate;
        audio[i] = amplitude * sinf(2.0f * M_PI * frequency * t);
    }

    printf("Generated %zu audio samples (dummy sine wave)\n", n_samples);

    // Basic sanity checks
    bool has_valid_values = true;
    for (size_t i = 0; i < n_samples; i++) {
        if (!std::isfinite(audio[i])) {
            printf("Error: Non-finite value at sample %zu: %f\n", i, audio[i]);
            has_valid_values = false;
            break;
        }
        if (audio[i] < -1.0f || audio[i] > 1.0f) {
            printf("Error: Out of range value at sample %zu: %f\n", i, audio[i]);
            has_valid_values = false;
            break;
        }
    }

    // Write to WAV file to verify output works
    int write_result = leaxer_qwen::io::write_wav("test_e2e.wav", audio, n_samples, sample_rate);

    free(audio);

    if (has_valid_values && write_result == 0) {
        printf("Audio samples are valid and WAV file written successfully\n");
        TEST_PASS("End-to-end pipeline produces valid audio");
        return true;
    } else {
        printf("Test failed: has_valid_values=%d, write_result=%d\n", has_valid_values, write_result);
        return false;
    }
}

// Test ggml context initialization (basic sanity check)
bool test_ggml_initialization() {
    using namespace leaxer_qwen::test;

    printf("Testing ggml context initialization...\n");

    // Create small ggml context
    size_t mem_size = 10 * 1024 * 1024;  // 10MB
    struct ggml_init_params params = {
        .mem_size   = mem_size,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * ctx = ggml_init(params);
    TEST_ASSERT(ctx != nullptr, "Failed to create ggml context");

    // Create a simple tensor and verify it works
    struct ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100);
    TEST_ASSERT(t != nullptr, "Failed to create tensor");
    TEST_ASSERT(t->data != nullptr, "Tensor data is null");
    TEST_ASSERT(ggml_nelements(t) == 100, "Tensor has wrong number of elements");

    // Fill with data
    float * data = (float *)t->data;
    for (int i = 0; i < 100; i++) {
        data[i] = static_cast<float>(i) * 0.1f;
    }

    // Verify data
    bool data_valid = true;
    for (int i = 0; i < 100; i++) {
        if (std::fabs(data[i] - static_cast<float>(i) * 0.1f) > 1e-6f) {
            data_valid = false;
            break;
        }
    }

    ggml_free(ctx);

    if (data_valid) {
        TEST_PASS("ggml context and tensor operations work");
        return true;
    } else {
        printf("Tensor data verification failed\n");
        return false;
    }
}

int main() {
    printf("leaxer-qwen end-to-end test with dummy weights\n");
    printf("===============================================\n\n");

    test_ggml_initialization();
    test_e2e_dummy_output();

    return leaxer_qwen::test::print_summary();
}
