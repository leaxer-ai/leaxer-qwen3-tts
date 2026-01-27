// Test RMSNorm operation
// Tests ggml tensor implementation against oracle fixtures

#include "test_utils.h"
#include "ggml.h"
#include "common.h"
#include <cmath>
#include <cstdio>

// Forward declarations
namespace leaxer_qwen {
namespace ops {
void rms_norm_inplace(
    struct ggml_tensor * dst,
    const struct ggml_tensor * x,
    const struct ggml_tensor * weight,
    float eps);
}
}

using namespace leaxer_qwen::test;
using namespace leaxer_qwen;

int main() {
    printf("Testing RMSNorm operation...\n\n");

    // Load fixtures
    auto input = load_fixture_f32("rmsnorm_input.bin");
    auto weight = load_fixture_f32("rmsnorm_weight.bin");
    auto expected = load_fixture_f32("rmsnorm_output.bin");

    // Validate shapes
    // Input: [1, 32, 1024] = 32768 floats (batch=1, seq_len=32, hidden_dim=1024)
    // Weight: [1024] floats
    size_t batch = 1;
    size_t seq_len = 32;
    size_t hidden_dim = 1024;
    size_t n_samples = batch * seq_len * hidden_dim;

    if (input.size() != n_samples) {
        printf("[FAIL] Input size mismatch: got %zu, expected %zu\n",
               input.size(), n_samples);
        return 1;
    }
    if (weight.size() != hidden_dim) {
        printf("[FAIL] Weight size mismatch: got %zu, expected %zu\n",
               weight.size(), hidden_dim);
        return 1;
    }
    if (expected.size() != n_samples) {
        printf("[FAIL] Output size mismatch: got %zu, expected %zu\n",
               expected.size(), n_samples);
        return 1;
    }

    printf("Shape validation passed\n");
    printf("  batch=%zu, seq_len=%zu, hidden_dim=%zu\n\n", batch, seq_len, hidden_dim);

    // Create ggml context
    size_t mem_size = 100 * 1024 * 1024;  // 100MB
    struct ggml_context * ctx = create_ggml_context(mem_size);
    if (!ctx) {
        printf("[FAIL] Failed to create ggml context\n");
        return 1;
    }

    // Create tensors
    // ggml tensor layout: [hidden_dim, seq_len, batch]
    struct ggml_tensor * x_tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden_dim, seq_len, batch);
    struct ggml_tensor * weight_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_dim);
    struct ggml_tensor * output_tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden_dim, seq_len, batch);

    // Copy data to tensors
    float * x_data = (float *)x_tensor->data;
    for (size_t i = 0; i < n_samples; i++) {
        x_data[i] = input[i];
    }

    float * weight_data = (float *)weight_tensor->data;
    for (size_t i = 0; i < hidden_dim; i++) {
        weight_data[i] = weight[i];
    }

    // Apply RMSNorm using inplace function
    ops::rms_norm_inplace(output_tensor, x_tensor, weight_tensor, 1e-6f);

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
        "RMSNorm operation (ggml)"
    );

    // Cleanup
    free_ggml_context(ctx);

    return print_summary();
}
