// Rotary Position Embeddings (RoPE)
// Qwen3-TTS uses multimodal RoPE with temporal/height/width dimensions
// For TTS, height and width are 1, so it reduces to temporal-only

#include "ggml.h"
#include "common.h"

namespace leaxer_qwen {
namespace ops {

// 1D RoPE for temporal dimension (TTS use case)
// Input: x with shape [head_dim, seq_len, num_heads, batch]
// pos: position tensor with shape [seq_len]
// n_dims: number of dimensions to apply RoPE to (typically head_dim)
// freq_base: base frequency for RoPE (default: 10000)
// Output: tensor with RoPE applied
struct ggml_tensor * rope_1d(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * pos,
    int n_dims,
    int mode = 0,
    float freq_base = 10000.0f) {

    // Use ggml's built-in RoPE with standard parameters
    // For 1D temporal RoPE, we use the basic ggml_rope function
    return ggml_rope(ctx, x, pos, n_dims, mode);
}

// Extended version with more control over RoPE parameters
struct ggml_tensor * rope_1d_ext(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * pos,
    int n_dims,
    int mode = 0,
    int n_ctx_orig = 0,
    float freq_base = 10000.0f,
    float freq_scale = 1.0f,
    float ext_factor = 0.0f,
    float attn_factor = 1.0f,
    float beta_fast = 32.0f,
    float beta_slow = 1.0f) {

    // Dummy freq_factors tensor (not used for basic 1D RoPE)
    struct ggml_tensor * freq_factors = nullptr;

    // Use ggml's extended RoPE for more control
    return ggml_rope_ext(
        ctx, x, pos, freq_factors,
        n_dims, mode, n_ctx_orig,
        freq_base, freq_scale,
        ext_factor, attn_factor,
        beta_fast, beta_slow
    );
}

} // namespace ops
} // namespace leaxer_qwen
