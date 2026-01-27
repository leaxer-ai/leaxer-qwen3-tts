// Test SnakeBeta activation
// Tests ggml tensor implementation against oracle fixtures

#include "test_utils.h"
#include "ggml.h"
#include "common.h"
#include <cmath>
#include <cstdio>

// Forward declarations
namespace leaxer_qwen {
namespace ops {
float snake_beta_scalar(float x, float alpha, float beta);
void snake_beta_inplace(
    struct ggml_tensor * dst,
    const struct ggml_tensor * x,
    const struct ggml_tensor * alpha_logscale,
    const struct ggml_tensor * beta_logscale);
}
}

using namespace leaxer_qwen::test;
using namespace leaxer_qwen;

int main() {
    printf("Testing SnakeBeta activation...\n\n");

    // Load fixtures
    auto input = load_fixture_f32("snakebeta_input.bin");
    auto alpha_logscale = load_fixture_f32("snakebeta_alpha_logscale.bin");
    auto beta_logscale = load_fixture_f32("snakebeta_beta_logscale.bin");
    auto expected = load_fixture_f32("snakebeta_output.bin");

    // Validate shapes
    // Input: [1, 64, 100] = 6400 floats (batch=1, channels=64, seq_len=100)
    // Alpha/Beta: [64] floats
    size_t batch = 1;
    size_t channels = 64;
    size_t seq_len = 100;
    size_t n_samples = batch * channels * seq_len;

    if (input.size() != n_samples) {
        printf("[FAIL] Input size mismatch: got %zu, expected %zu\n",
               input.size(), n_samples);
        return 1;
    }
    if (alpha_logscale.size() != channels) {
        printf("[FAIL] Alpha size mismatch: got %zu, expected %zu\n",
               alpha_logscale.size(), channels);
        return 1;
    }
    if (beta_logscale.size() != channels) {
        printf("[FAIL] Beta size mismatch: got %zu, expected %zu\n",
               beta_logscale.size(), channels);
        return 1;
    }
    if (expected.size() != n_samples) {
        printf("[FAIL] Output size mismatch: got %zu, expected %zu\n",
               expected.size(), n_samples);
        return 1;
    }

    printf("Shape validation passed\n");
    printf("  batch=%zu, channels=%zu, seq_len=%zu\n\n", batch, channels, seq_len);

    // Create ggml context
    size_t mem_size = 100 * 1024 * 1024;  // 100MB
    struct ggml_context * ctx = create_ggml_context(mem_size);
    if (!ctx) {
        printf("[FAIL] Failed to create ggml context\n");
        return 1;
    }

    // Create tensors
    // ggml tensor layout: [seq_len, channels, batch]
    struct ggml_tensor * x_tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, seq_len, channels, batch);
    struct ggml_tensor * alpha_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
    struct ggml_tensor * beta_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
    struct ggml_tensor * output_tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, seq_len, channels, batch);

    // Copy data to tensors
    // Note: ggml layout is [seq_len, channels, batch]
    // Our input data is [batch, channels, seq_len]
    // Need to reorder during copy
    float * x_data = (float *)x_tensor->data;
    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t t = 0; t < seq_len; t++) {
                size_t src_idx = b * channels * seq_len + c * seq_len + t;
                size_t dst_idx = b * channels * seq_len + c * seq_len + t;
                x_data[dst_idx] = input[src_idx];
            }
        }
    }

    float * alpha_data = (float *)alpha_tensor->data;
    float * beta_data = (float *)beta_tensor->data;
    for (size_t c = 0; c < channels; c++) {
        alpha_data[c] = alpha_logscale[c];
        beta_data[c] = beta_logscale[c];
    }

    // Apply SnakeBeta using inplace function
    ops::snake_beta_inplace(output_tensor, x_tensor, alpha_tensor, beta_tensor);

    // Extract results
    std::vector<float> output(n_samples);
    float * output_data = (float *)output_tensor->data;
    for (size_t i = 0; i < n_samples; i++) {
        output[i] = output_data[i];
    }

    // Compare with expected output
    bool passed = assert_tensor_close(
        output,
        expected,
        TOL_TIGHT,
        "SnakeBeta activation (ggml)"
    );

    // Cleanup
    free_ggml_context(ctx);

    return print_summary();
}
