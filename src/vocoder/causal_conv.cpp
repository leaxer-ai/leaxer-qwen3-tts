// Causal Convolution Network for Vocoder
// Projects RVQ output to transformer input dimension

#include "ggml.h"
#include "common.h"
#include <cstring>

namespace leaxer_qwen {
namespace vocoder {

// Forward declaration from conv1d.cpp
namespace ops {
struct ggml_tensor * conv1d_causal(
    struct ggml_context * ctx,
    struct ggml_tensor * weight,
    struct ggml_tensor * input,
    int kernel_size,
    int stride,
    int dilation
);
} // namespace ops

// Causal ConvNet projection layer
// Projects RVQ output from codebook_dim to hidden_dim
// Input: latent [codebook_dim, seq_len] - RVQ reconstructed features
// Weight: [kernel_size, in_channels, out_channels]
// Bias: [out_channels]
// Output: [out_channels, seq_len] - projected features
void causal_conv_project(
    struct ggml_tensor * dst,
    const struct ggml_tensor * latent,
    const struct ggml_tensor * weight,
    const struct ggml_tensor * bias
) {
    GGML_ASSERT(latent->type == GGML_TYPE_F32);
    GGML_ASSERT(weight->type == GGML_TYPE_F32);
    GGML_ASSERT(bias->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t codebook_dim = latent->ne[0];
    const int64_t seq_len = latent->ne[1];
    const int64_t kernel_size = weight->ne[0];
    const int64_t in_channels = weight->ne[1];
    const int64_t out_channels = weight->ne[2];

    GGML_ASSERT(codebook_dim == in_channels);
    GGML_ASSERT(bias->ne[0] == out_channels);
    GGML_ASSERT(dst->ne[0] == out_channels);
    GGML_ASSERT(dst->ne[1] == seq_len);

    // Create temporary context for computation graph
    size_t mem_size = 100 * 1024 * 1024;  // 100MB
    struct ggml_context * temp_ctx = create_ggml_context(mem_size);
    GGML_ASSERT(temp_ctx != nullptr);

    // Reshape latent to [seq_len, in_channels, 1] for conv1d
    struct ggml_tensor * input = ggml_new_tensor_3d(temp_ctx, GGML_TYPE_F32, seq_len, in_channels, 1);
    memcpy(input->data, latent->data, ggml_nbytes(latent));

    // Apply causal conv1d
    // Output: [seq_len, out_channels, 1]
    struct ggml_tensor * conv_out = ops::conv1d_causal(
        temp_ctx,
        (struct ggml_tensor *)weight,
        input,
        (int)kernel_size,
        1,  // stride
        1   // dilation
    );

    // Add bias
    // Broadcast bias across seq_len dimension
    const float * bias_data = (const float *)bias->data;
    float * conv_data = (float *)conv_out->data;

    for (int64_t t = 0; t < seq_len; t++) {
        for (int64_t c = 0; c < out_channels; c++) {
            conv_data[t * out_channels + c] += bias_data[c];
        }
    }

    // Reshape output from [seq_len, out_channels, 1] to [out_channels, seq_len]
    float * dst_data = (float *)dst->data;
    for (int64_t t = 0; t < seq_len; t++) {
        for (int64_t c = 0; c < out_channels; c++) {
            dst_data[c * seq_len + t] = conv_data[t * out_channels + c];
        }
    }

    // Cleanup
    free_ggml_context(temp_ctx);
}

} // namespace vocoder
} // namespace leaxer_qwen
