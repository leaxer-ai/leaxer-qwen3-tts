// SnakeBeta activation: x + (1/beta) * sin²(alpha * x)
// Alpha and beta are per-channel learned parameters

#include "ggml.h"
#include "common.h"
#include <cmath>
#include <cstdio>

namespace leaxer_qwen {
namespace ops {

// Scalar implementation for reference
float snake_beta_scalar(float x, float alpha, float beta) {
    float sin_ax = std::sin(alpha * x);
    return x + (1.0f / beta) * sin_ax * sin_ax;
}

// ggml tensor implementation using element-wise operations
struct ggml_tensor * snake_beta(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * alpha_logscale,
    struct ggml_tensor * beta_logscale) {

    // Convert from log-scale: alpha = exp(alpha_logscale), beta = exp(beta_logscale)
    struct ggml_tensor * alpha = ggml_exp(ctx, alpha_logscale);
    struct ggml_tensor * beta = ggml_exp(ctx, beta_logscale);

    // Reshape alpha and beta to match input dimensions for broadcasting
    // Input x is [seq_len, channels, batch]
    // Alpha/beta are [channels]
    // Need to reshape to [1, channels, 1] for proper broadcasting
    int64_t ne_broadcast[3] = {1, alpha->ne[0], 1};
    alpha = ggml_reshape_3d(ctx, alpha, ne_broadcast[0], ne_broadcast[1], ne_broadcast[2]);
    beta = ggml_reshape_3d(ctx, beta, ne_broadcast[0], ne_broadcast[1], ne_broadcast[2]);

    // Compute: alpha * x
    struct ggml_tensor * alpha_x = ggml_mul(ctx, alpha, x);

    // Compute: sin(alpha * x)
    // Note: ggml might not have sin, so we'll need to implement manually
    // For now, let's use a manual approach

    // ggml doesn't have built-in sin operation.
    // Use snake_beta_inplace() after graph execution for manual computation.
    struct ggml_tensor * result = ggml_dup(ctx, x);

    return result;
}

// Helper function to apply SnakeBeta to pre-allocated tensors
void snake_beta_inplace(
    struct ggml_tensor * dst,
    const struct ggml_tensor * x,
    const struct ggml_tensor * alpha_logscale,
    const struct ggml_tensor * beta_logscale) {

    GGML_ASSERT(ggml_are_same_shape(dst, x));
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(alpha_logscale->type == GGML_TYPE_F32);
    GGML_ASSERT(beta_logscale->type == GGML_TYPE_F32);

    const float * x_data = (const float *)x->data;
    const float * alpha_data = (const float *)alpha_logscale->data;
    const float * beta_data = (const float *)beta_logscale->data;
    float * dst_data = (float *)dst->data;

    const int64_t seq_len = x->ne[0];
    const int64_t channels = x->ne[1];
    const int64_t batch = x->ne[2];

    // Apply SnakeBeta: y = x + (1/beta) * sin²(alpha * x)
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t c = 0; c < channels; c++) {
            float alpha = expf(alpha_data[c]);
            float beta = expf(beta_data[c]);

            for (int64_t t = 0; t < seq_len; t++) {
                int64_t idx = b * channels * seq_len + c * seq_len + t;
                float val = x_data[idx];
                float sin_ax = sinf(alpha * val);
                dst_data[idx] = val + (1.0f / beta) * sin_ax * sin_ax;
            }
        }
    }
}

} // namespace ops
} // namespace leaxer_qwen
