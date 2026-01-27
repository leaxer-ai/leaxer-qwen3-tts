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
// Architecture: RMSNorm → Attention → Add → RMSNorm → FFN → Add
// Input: x with shape [hidden_dim, seq_len, batch]
// Weights:
//   - attn_norm_weight: [hidden_dim] for pre-attention RMSNorm
//   - q_weight: [hidden_dim, num_heads * head_dim] for query projection
//   - k_weight: [hidden_dim, num_kv_heads * head_dim] for key projection
//   - v_weight: [hidden_dim, num_kv_heads * head_dim] for value projection
//   - o_weight: [hidden_dim, num_heads * head_dim] for output projection
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
    struct ggml_tensor * ffn_norm_weight,
    struct ggml_tensor * ffn_w1,
    struct ggml_tensor * ffn_w2,
    struct ggml_tensor * ffn_w3) {

    // First residual branch: Attention
    // RMSNorm → Attention → Add
    struct ggml_tensor * normed = ops::rms_norm(ctx, x, attn_norm_weight, 1e-6f);

    // Compute attention (simplified - full GQA implementation would be more complex)
    // For now, we'll implement a basic attention path
    // Q, K, V projections
    struct ggml_tensor * Q = gqa_q_proj(ctx, normed, q_weight);
    struct ggml_tensor * K = gqa_kv_proj(ctx, normed, k_weight);
    struct ggml_tensor * V = gqa_kv_proj(ctx, normed, v_weight);

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
