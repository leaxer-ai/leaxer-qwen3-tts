// Grouped Query Attention (GQA)
// 16 query heads, 2 key-value heads
// Used in main Qwen3 transformer

#include "ggml.h"
#include "common.h"
#include <cmath>

namespace leaxer_qwen {

// Forward declaration from rope.cpp
namespace ops {
struct ggml_tensor * rope_1d(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * pos,
    int n_dims,
    int mode,
    float freq_base);
}

namespace model {

// Attention configuration (from model config.json)
// hidden_size=1024, num_attention_heads=16, num_key_value_heads=8
// head_dim = hidden_size / num_attention_heads = 1024/16 = 64 for Q
// but Q proj has output 2048 = 16 * 128, so head_dim=128
constexpr int NUM_HEADS = 16;
constexpr int NUM_KV_HEADS = 8;   // GQA: 8 KV heads shared across 16 Q heads
constexpr int HEAD_DIM = 128;     // 2048/16 = 128 (Q), 1024/8 = 128 (KV)

// Q projection for GQA
// Projects input to query vectors for all attention heads
// Input: x with shape [hidden_dim, seq_len, batch]
// Weight: q_weight with shape [hidden_dim, num_heads * head_dim]
// Output: queries with shape [num_heads * head_dim, seq_len, batch]
struct ggml_tensor * gqa_q_proj(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * q_weight) {

    // Matrix multiplication: q_weight^T @ x
    // ggml_mul_mat expects: (weight is transposed internally)
    // q_weight: [hidden_dim, num_heads * head_dim]
    // x: [hidden_dim, seq_len, batch]
    // output: [num_heads * head_dim, seq_len, batch]
    struct ggml_tensor * queries = ggml_mul_mat(ctx, q_weight, x);

    return queries;
}

// K/V projection for GQA
// Projects input to key/value vectors for KV heads (shared across query head groups)
// Input: x with shape [hidden_dim, seq_len, batch]
// Weight: kv_weight with shape [hidden_dim, num_kv_heads * head_dim]
// Output: keys/values with shape [num_kv_heads * head_dim, seq_len, batch]
struct ggml_tensor * gqa_kv_proj(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * kv_weight) {

    // Matrix multiplication: kv_weight^T @ x
    // ggml_mul_mat expects: (weight is transposed internally)
    // kv_weight: [hidden_dim, num_kv_heads * head_dim]
    // x: [hidden_dim, seq_len, batch]
    // output: [num_kv_heads * head_dim, seq_len, batch]
    struct ggml_tensor * kv = ggml_mul_mat(ctx, kv_weight, x);

    return kv;
}

// Compute attention scores: Q*K^T / sqrt(d) with causal mask
// Input: Q with shape [head_dim, seq_len, num_heads * batch]
// Input: K with shape [head_dim, seq_len, num_heads * batch]
// Output: scores with shape [seq_len, seq_len, num_heads * batch]
// Applies causal mask (upper triangular set to -inf)
// Note: RoPE (Rotary Position Embeddings) should be applied to Q and K before this function
// In production, RoPE would be applied with proper position indices from KV cache
struct ggml_tensor * attention_scores(
    struct ggml_context * ctx,
    struct ggml_tensor * Q,
    struct ggml_tensor * K) {

    // Get dimensions
    // Q, K: [head_dim, seq_len, num_heads * batch]
    int head_dim = Q->ne[0];
    int seq_len = Q->ne[1];
    int num_heads_batch = Q->ne[2];

    // Note: RoPE is applied elsewhere in the pipeline with proper position tracking
    // For Qwen3-TTS, RoPE encoding is critical for position awareness
    // In a full implementation, ggml_rope would be called here with position tensor
    // that tracks the actual position of each token in the sequence

    // Compute Q * K^T
    // We need to transpose K: [head_dim, seq_len] -> [seq_len, head_dim]
    // Then multiply: [head_dim, seq_len] @ [seq_len, head_dim]^T = [seq_len, seq_len]
    // ggml_mul_mat(ctx, a, b) computes a^T @ b, so we need: K^T @ Q
    // which gives us [seq_len, head_dim] @ [head_dim, seq_len] = [seq_len, seq_len]

    // For each head separately, we need:
    // Q: [head_dim, seq_len] @ K^T: [seq_len, head_dim] = [seq_len, seq_len]
    // In ggml terms: ggml_mul_mat(K, Q) where K and Q are per-head slices

    // Actually, ggml_mul_mat with 3d tensors will batch across the 3rd dimension
    // So we can just do: ggml_mul_mat(ctx, K, Q)
    // This computes K^T @ Q for each slice along dim 2
    // Result: [seq_len, seq_len, num_heads * batch]
    struct ggml_tensor * scores = ggml_mul_mat(ctx, K, Q);

    // Scale by 1/sqrt(head_dim)
    float scale = 1.0f / sqrtf((float)head_dim);
    scores = ggml_scale(ctx, scores, scale);

    // Apply causal mask
    // Set upper triangular part (j > i) to -inf
    scores = ggml_diag_mask_inf(ctx, scores, 0);

    return scores;
}

// Attention output: softmax(scores) * V + output projection
// Input: scores with shape [seq_len, seq_len, num_heads * batch] (pre-softmax)
// Input: V with shape [head_dim, seq_len, num_heads * batch]
// Input: o_weight with shape [hidden_dim, num_heads * head_dim] (output projection)
// Output: output with shape [hidden_dim, seq_len, batch]
struct ggml_tensor * attention_output(
    struct ggml_context * ctx,
    struct ggml_tensor * scores,
    struct ggml_tensor * V,
    struct ggml_tensor * o_weight) {

    // Get dimensions
    // scores: [seq_len, seq_len, num_heads * batch]
    // V: [head_dim, seq_len, num_heads * batch]
    int seq_len = scores->ne[0];
    int num_heads_batch = scores->ne[2];
    int head_dim = V->ne[0];

    // Apply softmax to scores
    // Softmax along the key dimension (dim 1, which has size seq_len)
    struct ggml_tensor * attn_weights = ggml_soft_max(ctx, scores);

    // Compute context = attn_weights @ V
    // ggml_mul_mat(a, b) computes a^T @ b, requiring ne0 to match.
    // attn_weights: [seq, seq], V: [head_dim, seq] - ne0 doesn't match.
    // Solution: transpose V to [seq, head_dim], multiply, transpose back.
    struct ggml_tensor * V_T = ggml_cont(ctx, ggml_permute(ctx, V, 1, 0, 2, 3));

    // attn_weights @ V_T: [seq, seq] @ [seq, head_dim] -> [seq, head_dim]
    struct ggml_tensor * context = ggml_mul_mat(ctx, attn_weights, V_T);

    // Transpose back to [head_dim, seq_len, num_heads*batch]
    context = ggml_cont(ctx, ggml_permute(ctx, context, 1, 0, 2, 3));

    // Reshape to concatenate heads: [num_heads*head_dim, seq_len, batch]
    // Assuming batch=1 for simplicity
    context = ggml_reshape_3d(ctx, context,
                              head_dim * num_heads_batch,
                              seq_len,
                              1);

    // Apply output projection: o_weight^T @ context
    struct ggml_tensor * output = ggml_mul_mat(ctx, o_weight, context);

    return output;
}

} // namespace model
} // namespace leaxer_qwen
