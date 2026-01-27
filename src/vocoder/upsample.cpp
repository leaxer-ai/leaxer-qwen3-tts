// Transposed Convolution Upsampling for Vocoder
// Upsample rates: 8, 5, 4, 3 (total: 480x per two stages, 1920x total)

#include "ggml.h"
#include "ggml-cpu.h"
#include "common.h"
#include <cstring>

namespace leaxer_qwen {
namespace vocoder {

// Upsample configuration
constexpr int UPSAMPLE_RATES[] = {8, 5, 4, 3};
constexpr int NUM_UPSAMPLE_STAGES = 4;
constexpr int TOTAL_UPSAMPLE = 8 * 5 * 4 * 3;  // 480

// Forward declaration of SnakeBeta from ops
namespace ops {
void snake_beta_inplace(
    struct ggml_tensor * dst,
    const struct ggml_tensor * x,
    const struct ggml_tensor * alpha_logscale,
    const struct ggml_tensor * beta_logscale);
}

// Transposed Conv1d for upsampling
// Input: [seq_len, in_channels, batch] (ggml layout)
// Weight: [kernel_size, in_channels, out_channels] (ggml layout)
// Output: [seq_len * stride, out_channels, batch]
struct ggml_tensor * conv1d_transpose(
    struct ggml_context * ctx,
    struct ggml_tensor * weight,  // [kernel_size, in_channels, out_channels]
    struct ggml_tensor * input,   // [seq_len, in_channels, batch]
    int kernel_size,
    int stride,
    int padding = 0,
    int dilation = 1
) {
    // Use ggml_conv_transpose_1d
    // Parameters: kernel, data, stride, padding, dilation
    struct ggml_tensor * output = ggml_conv_transpose_1d(ctx, weight, input, stride, padding, dilation);

    return output;
}

// Single upsample stage: transposed conv + SnakeBeta activation
// Input: [seq_len, in_channels, batch]
// Weight: [kernel_size, in_channels, out_channels]
// Alpha/Beta: [out_channels] - per-channel SnakeBeta parameters
// Output: [seq_len * stride, out_channels, batch]
void upsample_stage(
    struct ggml_tensor * dst,
    const struct ggml_tensor * input,
    const struct ggml_tensor * weight,
    const struct ggml_tensor * alpha_logscale,
    const struct ggml_tensor * beta_logscale,
    int kernel_size,
    int stride,
    int padding = 0
) {
    GGML_ASSERT(input->type == GGML_TYPE_F32);
    GGML_ASSERT(weight->type == GGML_TYPE_F32);
    GGML_ASSERT(alpha_logscale->type == GGML_TYPE_F32);
    GGML_ASSERT(beta_logscale->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t seq_len_in = input->ne[0];
    const int64_t in_channels = input->ne[1];
    const int64_t batch = input->ne[2];
    const int64_t out_channels = weight->ne[2];

    // Expected output dimensions: [seq_len * stride, out_channels, batch]
    const int64_t seq_len_out = seq_len_in * stride;
    GGML_ASSERT(dst->ne[0] == seq_len_out);
    GGML_ASSERT(dst->ne[1] == out_channels);
    GGML_ASSERT(dst->ne[2] == batch);

    // Create temporary context for intermediate tensor
    size_t mem_size = 100 * 1024 * 1024;  // 100MB
    struct ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * temp_ctx = ggml_init(params);

    // Create mutable copies of input tensors for ggml operations
    struct ggml_tensor * input_copy = ggml_new_tensor_3d(temp_ctx, GGML_TYPE_F32, seq_len_in, in_channels, batch);
    struct ggml_tensor * weight_copy = ggml_new_tensor_3d(temp_ctx, GGML_TYPE_F32, kernel_size, in_channels, out_channels);

    // Copy data
    memcpy(input_copy->data, input->data, ggml_nbytes(input));
    memcpy(weight_copy->data, weight->data, ggml_nbytes(weight));

    // Step 1: Apply transposed convolution
    struct ggml_tensor * conv_out = conv1d_transpose(
        temp_ctx,
        weight_copy,
        input_copy,
        kernel_size,
        stride,
        padding,
        1  // dilation
    );

    // Build and compute graph for conv operation
    struct ggml_cgraph * graph = ggml_new_graph(temp_ctx);
    ggml_build_forward_expand(graph, conv_out);
    ggml_graph_compute_with_ctx(temp_ctx, graph, 1);  // 1 thread

    // Step 2: Apply SnakeBeta activation in-place to dst
    // First copy conv output to dst
    memcpy(dst->data, conv_out->data, ggml_nbytes(dst));

    // Apply SnakeBeta activation in-place
    ops::snake_beta_inplace(dst, dst, alpha_logscale, beta_logscale);

    // Cleanup
    ggml_free(temp_ctx);
}

} // namespace vocoder
} // namespace leaxer_qwen
