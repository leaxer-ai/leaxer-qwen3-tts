// Test Causal Conv1d
// Tests causal 1D convolution against oracle fixtures

#include "test_utils.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "common.h"
#include <cmath>
#include <cstdio>

// Forward declarations
namespace leaxer_qwen {
namespace ops {
struct ggml_tensor * conv1d_causal(
    struct ggml_context * ctx,
    struct ggml_tensor * weight,
    struct ggml_tensor * input,
    int kernel_size,
    int stride,
    int dilation);
}
}

using namespace leaxer_qwen::test;
using namespace leaxer_qwen;

int main() {
    printf("Testing Causal Conv1d...\n\n");

    // Load fixtures
    auto input = load_fixture_f32("conv1d_input.bin");
    auto weight = load_fixture_f32("conv1d_weight.bin");
    auto expected = load_fixture_f32("conv1d_output.bin");

    // Validate shapes
    // Input: [1, 64, 100]
    // Weight: [128, 64, 3]
    // Output: [1, 128, 100]
    size_t batch = 1;
    size_t in_channels = 64;
    size_t out_channels = 128;
    size_t seq_len = 100;
    size_t kernel_size = 3;

    size_t input_size = batch * in_channels * seq_len;
    size_t weight_size = out_channels * in_channels * kernel_size;
    size_t output_size = batch * out_channels * seq_len;

    if (input.size() != input_size) {
        printf("[FAIL] Input size mismatch: got %zu, expected %zu\n",
               input.size(), input_size);
        return 1;
    }
    if (weight.size() != weight_size) {
        printf("[FAIL] Weight size mismatch: got %zu, expected %zu\n",
               weight.size(), weight_size);
        return 1;
    }
    if (expected.size() != output_size) {
        printf("[FAIL] Output size mismatch: got %zu, expected %zu\n",
               expected.size(), output_size);
        return 1;
    }

    printf("Shape validation passed\n");
    printf("  batch=%zu, in_channels=%zu, out_channels=%zu, seq_len=%zu, kernel_size=%zu\n\n",
           batch, in_channels, out_channels, seq_len, kernel_size);

    // Create ggml context
    size_t mem_size = 100 * 1024 * 1024;  // 100MB
    struct ggml_context * ctx = create_ggml_context(mem_size);
    if (!ctx) {
        printf("[FAIL] Failed to create ggml context\n");
        return 1;
    }

    // Create tensors
    // ggml layout for conv1d (same as PyTorch row-major when batch=1):
    //   - kernel (weight): [kernel_size, in_channels, out_channels] (GGML_TYPE_F16)
    //   - input: [seq_len, in_channels, batch] (GGML_TYPE_F32)
    //   - output: [seq_len, out_channels, batch]
    struct ggml_tensor * weight_tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F16,
                                                             kernel_size, in_channels, out_channels);
    struct ggml_tensor * input_tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                                            seq_len, in_channels, batch);

    // Copy weight data and convert to F16
    ggml_fp16_t * weight_data = (ggml_fp16_t *)weight_tensor->data;
    ggml_fp32_to_fp16_row(weight.data(), weight_data, weight_size);

    // Copy input data
    float * input_data = (float *)input_tensor->data;
    memcpy(input_data, input.data(), input_size * sizeof(float));

    // Apply causal conv1d
    struct ggml_tensor * output_tensor = ops::conv1d_causal(
        ctx, weight_tensor, input_tensor, kernel_size, 1, 1
    );

    // Build and compute graph using backend
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output_tensor);

    // Create backend and allocator
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(allocr, graph);

    // Compute
    ggml_backend_graph_compute(backend, graph);

    // Cleanup backend
    ggml_gallocr_free(allocr);
    ggml_backend_free(backend);

    // Extract results
    // Both PyTorch and ggml store in row-major order
    // Since batch=1, the layouts are equivalent
    std::vector<float> output(output_size);
    float * output_data = (float *)output_tensor->data;
    memcpy(output.data(), output_data, output_size * sizeof(float));

    // Compare with expected output
    // Use very relaxed tolerance due to F16 precision in convolution
    // Conv operations accumulate errors, so we need higher tolerance
    bool passed = assert_tensor_close(
        output,
        expected,
        0.01f,  // 1% error tolerance
        "Causal Conv1d (ggml)"
    );

    // Cleanup
    free_ggml_context(ctx);

    return print_summary();
}
