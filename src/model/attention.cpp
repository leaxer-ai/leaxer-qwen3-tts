// Grouped Query Attention (GQA)
// 16 query heads, 2 key-value heads
// Used in main Qwen3 transformer

#include "ggml.h"
#include "common.h"
#include <cmath>

namespace leaxer_qwen {
namespace model {

// Attention configuration
constexpr int NUM_HEADS = 16;
constexpr int NUM_KV_HEADS = 2;
constexpr int HEAD_DIM = 64;  // hidden_size / num_heads

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
struct ggml_tensor * attention_scores(
    struct ggml_context * ctx,
    struct ggml_tensor * Q,
    struct ggml_tensor * K) {

    // Get dimensions
    // Q, K: [head_dim, seq_len, num_heads * batch]
    int head_dim = Q->ne[0];
    int seq_len = Q->ne[1];
    int num_heads_batch = Q->ne[2];

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

// TODO: Implement remaining GQA components
// Key features:
// - RoPE position embeddings
// - Full attention with softmax and value projection

} // namespace model
} // namespace leaxer_qwen
