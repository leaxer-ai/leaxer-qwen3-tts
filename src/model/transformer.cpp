// Transformer Block
// RMSNorm → GQA → Residual → RMSNorm → FFN → Residual

#include "ggml.h"
#include "common.h"

namespace leaxer_qwen {
namespace model {

// Forward declarations from attention.cpp
struct ggml_tensor * gqa_q_proj(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * q_weight);

struct ggml_tensor * gqa_kv_proj(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * kv_weight);

struct ggml_tensor * attention_scores(
    struct ggml_context * ctx,
    struct ggml_tensor * Q,
    struct ggml_tensor * K);

struct ggml_tensor * attention_output(
    struct ggml_context * ctx,
    struct ggml_tensor * scores,
    struct ggml_tensor * V,
    struct ggml_tensor * o_weight);

// Forward declaration from ffn.cpp
struct ggml_tensor * swiglu_ffn(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * w1,
    struct ggml_tensor * w2,
    struct ggml_tensor * w3);

} // namespace model

namespace ops {

// Forward declaration from rms_norm.cpp
struct ggml_tensor * rms_norm(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * weight,
    float eps);

} // namespace ops

namespace model {

// Qwen3-TTS transformer configuration
// 1.7B model: 20 layers, hidden_size=1024
// 0.6B model: 12 layers, hidden_size=896

// Transformer block with pre-normalization
// Architecture: RMSNorm → Attention (with Q/K norms) → Add → RMSNorm → FFN → Add
// Input: x with shape [hidden_dim, seq_len, batch]
// Weights:
//   - attn_norm_weight: [hidden_dim] for pre-attention RMSNorm
//   - q_weight: [hidden_dim, num_heads * head_dim] for query projection
//   - k_weight: [hidden_dim, num_kv_heads * head_dim] for key projection
//   - v_weight: [hidden_dim, num_kv_heads * head_dim] for value projection
//   - o_weight: [hidden_dim, num_heads * head_dim] for output projection
//   - q_norm_weight: [head_dim] for per-head Q normalization (nullptr = skip)
//   - k_norm_weight: [head_dim] for per-head K normalization (nullptr = skip)
//   - ffn_norm_weight: [hidden_dim] for pre-FFN RMSNorm
//   - w1, w2, w3: FFN weights
// Output: [hidden_dim, seq_len, batch]
struct ggml_tensor * transformer_block(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * attn_norm_weight,
    struct ggml_tensor * q_weight,
    struct ggml_tensor * k_weight,
    struct ggml_tensor * v_weight,
    struct ggml_tensor * o_weight,
    struct ggml_tensor * q_norm_weight,
    struct ggml_tensor * k_norm_weight,
    struct ggml_tensor * ffn_norm_weight,
    struct ggml_tensor * ffn_w1,
    struct ggml_tensor * ffn_w2,
    struct ggml_tensor * ffn_w3) {

    // First residual branch: Attention
    // RMSNorm → Attention → Add
    struct ggml_tensor * normed = ops::rms_norm(ctx, x, attn_norm_weight, 1e-6f);

    // Compute attention (GQA - Grouped Query Attention)
    // Q, K, V projections
    struct ggml_tensor * Q = gqa_q_proj(ctx, normed, q_weight);
    struct ggml_tensor * K = gqa_kv_proj(ctx, normed, k_weight);
    struct ggml_tensor * V = gqa_kv_proj(ctx, normed, v_weight);

    // Reshape Q, K, V to separate head dimension
    // Q: [num_heads * head_dim, seq_len] -> [head_dim, seq_len, num_heads]
    // K, V: [num_kv_heads * head_dim, seq_len] -> [head_dim, seq_len, num_kv_heads]
    int seq_len = Q->ne[1];
    int q_dim = Q->ne[0];  // num_heads * head_dim
    int kv_dim = K->ne[0];  // num_kv_heads * head_dim

    // For GQA: 16 query heads, 8 KV heads, head_dim = 128
    // These should be parameters, but hardcode for now to match model config
    const int num_heads = 16;
    const int num_kv_heads = 8;  // GQA: 8 KV heads shared across 16 Q heads
    const int head_dim = q_dim / num_heads;  // 2048/16 = 128
    const int kv_head_dim = kv_dim / num_kv_heads;  // 1024/8 = 128
    const float rope_freq_base = 1000000.0f;  // Qwen3-TTS uses 1M, NOT 10K!

    // First convert to 4D to separate heads, then permute
    // Q: [num_heads * head_dim, seq_len] = [head_dim, num_heads, seq_len, 1]  (view)
    // Then permute to [head_dim, seq_len, num_heads, 1]
    Q = ggml_reshape_4d(ctx, Q, head_dim, num_heads, seq_len, 1);
    Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));  // [head_dim, seq_len, num_heads, 1]
    Q = ggml_reshape_3d(ctx, Q, head_dim, seq_len, num_heads);  // Remove singleton dim

    // K and V use kv_head_dim which should equal head_dim for this model
    K = ggml_reshape_4d(ctx, K, kv_head_dim, num_kv_heads, seq_len, 1);
    K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
    K = ggml_reshape_3d(ctx, K, kv_head_dim, seq_len, num_kv_heads);

    V = ggml_reshape_4d(ctx, V, kv_head_dim, num_kv_heads, seq_len, 1);
    V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));
    V = ggml_reshape_3d(ctx, V, kv_head_dim, seq_len, num_kv_heads);

    // Apply Q/K normalization (RMSNorm per head, applied to head_dim)
    // This is a key component of Qwen3 attention - normalizes Q and K before computing scores
    if (q_norm_weight != nullptr) {
        // Q: [head_dim, seq_len, num_heads] - norm applied to head_dim dimension
        Q = ops::rms_norm(ctx, Q, q_norm_weight, 1e-6f);
    }
    if (k_norm_weight != nullptr) {
        // K: [head_dim, seq_len, num_kv_heads] - norm applied to head_dim dimension
        K = ops::rms_norm(ctx, K, k_norm_weight, 1e-6f);
    }

    // Apply RoPE (Rotary Position Embeddings) to Q and K
    // CRITICAL: Qwen3 applies RoPE AFTER Q/K normalization
    // Position tensor needs to be created for the sequence
    // For simplicity, we create positions 0, 1, 2, ..., seq_len-1
    {
        // Create position tensor [seq_len]
        struct ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len);
        int32_t * pos_data = (int32_t *)pos->data;
        for (int i = 0; i < seq_len; i++) {
            pos_data[i] = i;
        }

        // Apply RoPE with freq_base=1000000 (Qwen3-TTS setting - NOT 10000!)
        // ggml_rope_ext params: ctx, a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
        // mode=0 for standard RoPE (Qwen3 uses interleaved M-RoPE but for TTS all 3 dims are same, so equivalent to standard)

        // Permute Q: [head_dim, seq_len, num_heads] -> [head_dim, num_heads, seq_len, 1]
        Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
        Q = ggml_rope_ext(ctx, Q, pos, nullptr, head_dim, 0,
                          0, rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));  // Back to [head_dim, seq_len, num_heads]

        // Permute K: [head_dim, seq_len, num_kv_heads] -> [head_dim, num_kv_heads, seq_len, 1]
        K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
        K = ggml_rope_ext(ctx, K, pos, nullptr, head_dim, 0,
                          0, rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));  // Back to [head_dim, seq_len, num_kv_heads]
    }

    // GQA: Expand K and V heads to match Q heads
    // K/V have 8 heads, Q has 16 heads, so repeat K/V heads 2x
    // This is done by repeating along the heads dimension (dim 2)
    // Use Q tensor as the shape reference since it already has the target shape
    const int heads_ratio = num_heads / num_kv_heads;  // 16/8 = 2
    if (heads_ratio > 1) {
        // K: [head_dim, seq_len, num_kv_heads] -> [head_dim, seq_len, num_heads]
        // Use Q as shape reference (Q has shape [head_dim, seq_len, num_heads])
        K = ggml_repeat(ctx, K, Q);
        V = ggml_repeat(ctx, V, Q);
    }

    // Compute attention scores
    struct ggml_tensor * scores = attention_scores(ctx, Q, K);

    // Compute attention output
    struct ggml_tensor * attn_out = attention_output(ctx, scores, V, o_weight);

    // Residual connection
    struct ggml_tensor * x_residual = ggml_add(ctx, x, attn_out);

    // Second residual branch: FFN
    // RMSNorm → FFN → Add
    struct ggml_tensor * ffn_normed = ops::rms_norm(ctx, x_residual, ffn_norm_weight, 1e-6f);

    // Apply SwiGLU FFN
    struct ggml_tensor * ffn_out = swiglu_ffn(ctx, ffn_normed, ffn_w1, ffn_w2, ffn_w3);

    // Residual connection
    struct ggml_tensor * output = ggml_add(ctx, x_residual, ffn_out);

    return output;
}

} // namespace model
} // namespace leaxer_qwen
