// RMS Normalization (Qwen/LLaMA style)
// y = x * rsqrt(mean(x²) + eps) * weight

#include "ggml.h"
#include "common.h"
#include <cmath>
#include <cstdio>

namespace leaxer_qwen {
namespace ops {

// RMS normalization using ggml built-in operation
// Input: x with shape [hidden_dim, seq_len, batch]
// Weight: weight with shape [hidden_dim]
// Output: normalized tensor with same shape as input
struct ggml_tensor * rms_norm(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * weight,
    float eps = 1e-6f) {

    // Use ggml's built-in RMS normalization
    struct ggml_tensor * normalized = ggml_rms_norm(ctx, x, eps);

    // Scale by weight (element-wise multiplication)
    // Weight needs to be broadcast across the normalized tensor
    struct ggml_tensor * output = ggml_mul(ctx, normalized, weight);

    return output;
}

// In-place version for pre-allocated output tensor
void rms_norm_inplace(
    struct ggml_tensor * dst,
    const struct ggml_tensor * x,
    const struct ggml_tensor * weight,
    float eps = 1e-6f) {

    GGML_ASSERT(ggml_are_same_shape(dst, x));
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weight->type == GGML_TYPE_F32);

    const float * x_data = (const float *)x->data;
    const float * weight_data = (const float *)weight->data;
    float * dst_data = (float *)dst->data;

    const int64_t hidden_dim = x->ne[0];
    const int64_t seq_len = x->ne[1];
    const int64_t batch = x->ne[2];

    // Apply RMS normalization: y = x * rsqrt(mean(x²) + eps) * weight
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t s = 0; s < seq_len; s++) {
            // Compute RMS for this sequence position
            float sum_sq = 0.0f;
            for (int64_t h = 0; h < hidden_dim; h++) {
                int64_t idx = b * seq_len * hidden_dim + s * hidden_dim + h;
                float val = x_data[idx];
                sum_sq += val * val;
            }

            float rms = sqrtf(sum_sq / hidden_dim + eps);
            float scale = 1.0f / rms;

            // Normalize and scale by weight
            for (int64_t h = 0; h < hidden_dim; h++) {
                int64_t idx = b * seq_len * hidden_dim + s * hidden_dim + h;
                dst_data[idx] = x_data[idx] * scale * weight_data[h];
            }
        }
    }
}

} // namespace ops
} // namespace leaxer_qwen
