// Test SnakeBeta activation
// Tests scalar implementation against oracle fixtures

#include "test_utils.h"
#include <cmath>
#include <cstdio>

// Forward declaration of scalar implementation
namespace leaxer_qwen {
namespace ops {
float snake_beta_scalar(float x, float alpha, float beta);
}
}

using namespace leaxer_qwen::test;

int main() {
    printf("Testing SnakeBeta activation...\n\n");

    // Load fixtures
    auto input = load_fixture_f32("snakebeta_input.bin");
    auto alpha_logscale = load_fixture_f32("snakebeta_alpha_logscale.bin");
    auto beta_logscale = load_fixture_f32("snakebeta_beta_logscale.bin");
    auto expected = load_fixture_f32("snakebeta_output.bin");

    // Validate shapes
    // Input: [1, 64, 100] = 6400 floats
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

    // Compute SnakeBeta activation
    // Formula: y = x + (1/beta) * sin²(alpha * x)
    // Alpha and beta are stored as log-scale values
    std::vector<float> output(n_samples);

    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            // Convert from log-scale
            float alpha = std::exp(alpha_logscale[c]);
            float beta = std::exp(beta_logscale[c]);

            for (size_t t = 0; t < seq_len; t++) {
                size_t idx = b * channels * seq_len + c * seq_len + t;
                float x = input[idx];

                // Apply SnakeBeta
                output[idx] = leaxer_qwen::ops::snake_beta_scalar(x, alpha, beta);
            }
        }
    }

    // Compare with expected output
    bool passed = assert_tensor_close(
        output,
        expected,
        TOL_TIGHT,
        "SnakeBeta activation"
    );

    return print_summary();
}
