// Transformer Block with KV Caching
// Enables efficient autoregressive generation by caching key/value tensors

#include "ggml.h"
#include "common.h"
#include "kv_cache.h"
#include <cmath>
#include <cstring>

namespace leaxer_qwen {
namespace model {

// Forward declarations
struct ggml_tensor * gqa_q_proj(struct ggml_context * ctx, struct ggml_tensor * x, struct ggml_tensor * q_weight);
struct ggml_tensor * gqa_kv_proj(struct ggml_context * ctx, struct ggml_tensor * x, struct ggml_tensor * kv_weight);

} // namespace model

namespace ops {
struct ggml_tensor * rms_norm(struct ggml_context * ctx, struct ggml_tensor * x, struct ggml_tensor * weight, float eps);
} // namespace ops

namespace model {

// Forward declaration from ffn.cpp
struct ggml_tensor * swiglu_ffn(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * w1,
    struct ggml_tensor * w2,
    struct ggml_tensor * w3);

// Configuration
constexpr int NUM_HEADS = 16;
constexpr int NUM_KV_HEADS = 8;
constexpr int HEAD_DIM = 128;

// Compute attention scores with cached K/V
// Q: [head_dim, q_len, num_heads] - queries for new token(s)
// K: [head_dim, kv_len, num_kv_heads] - keys (from cache + new)
// V: [head_dim, kv_len, num_kv_heads] - values (from cache + new)
// Returns: attention output [head_dim * num_heads, q_len]
static struct ggml_tensor * cached_attention(
    struct ggml_context * ctx,
    struct ggml_tensor * Q,
    struct ggml_tensor * K,
    struct ggml_tensor * V,
    struct ggml_tensor * o_weight,
    int num_heads,
    int num_kv_heads) {

    int head_dim = Q->ne[0];
    int q_len = Q->ne[1];
    int kv_len = K->ne[1];

    // GQA: Expand K and V heads to match Q heads
    int heads_ratio = num_heads / num_kv_heads;
    if (heads_ratio > 1) {
        // Create target shape tensor with Q's head count
        struct ggml_tensor * target = ggml_new_tensor_3d(ctx, K->type, head_dim, kv_len, num_heads);
        K = ggml_repeat(ctx, K, target);
        V = ggml_repeat(ctx, V, target);
    }

    // Compute attention scores: Q @ K^T / sqrt(d)
    // Q: [head_dim, q_len, num_heads]
    // K: [head_dim, kv_len, num_heads]
    // scores = K^T @ Q = [kv_len, q_len, num_heads]
    struct ggml_tensor * scores = ggml_mul_mat(ctx, K, Q);

    // Scale
    float scale = 1.0f / sqrtf((float)head_dim);
    scores = ggml_scale(ctx, scores, scale);

    // Causal mask - only mask future tokens relative to query position
    // For cached generation, query is at position (kv_len - q_len) to (kv_len - 1)
    // We need to mask positions > current query position
    scores = ggml_diag_mask_inf(ctx, scores, kv_len - q_len);

    // Softmax
    struct ggml_tensor * attn_weights = ggml_soft_max(ctx, scores);

    // Compute context: attn_weights @ V
    // attn_weights: [kv_len, q_len, num_heads]
    // V: [head_dim, kv_len, num_heads]
    // V_T: [kv_len, head_dim, num_heads]
    struct ggml_tensor * V_T = ggml_cont(ctx, ggml_permute(ctx, V, 1, 0, 2, 3));

    // context = attn_weights @ V_T = [kv_len, q_len] @ [kv_len, head_dim] = [q_len, head_dim]
    // But we need [head_dim, q_len], so transpose after
    struct ggml_tensor * context = ggml_mul_mat(ctx, attn_weights, V_T);
    context = ggml_cont(ctx, ggml_permute(ctx, context, 1, 0, 2, 3));

    // Reshape to concatenate heads: [num_heads * head_dim, q_len]
    context = ggml_reshape_3d(ctx, context, head_dim * num_heads, q_len, 1);

    // Output projection
    struct ggml_tensor * output = ggml_mul_mat(ctx, o_weight, context);

    return output;
}

// Transformer block with KV caching
// For prefill: process all tokens, store K/V in cache
// For decode: process only new token(s), use cached K/V
//
// Parameters:
//   x: input tensor [hidden_dim, seq_len, 1]
//   layer_idx: layer index for KV cache
//   kv_cache: KV cache (can be nullptr for non-cached mode)
//   start_pos: position of first token in x (0 for prefill, cache_len for decode)
//   q_norm_weight: Q normalization weight (can be nullptr to skip)
//   k_norm_weight: K normalization weight (can be nullptr to skip)
struct ggml_tensor * transformer_block_cached(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * attn_norm_weight,
    struct ggml_tensor * q_weight,
    struct ggml_tensor * k_weight,
    struct ggml_tensor * v_weight,
    struct ggml_tensor * o_weight,
    struct ggml_tensor * q_norm_weight,      // Q normalization (Qwen3-specific)
    struct ggml_tensor * k_norm_weight,      // K normalization (Qwen3-specific)
    struct ggml_tensor * ffn_norm_weight,
    struct ggml_tensor * ffn_w1,
    struct ggml_tensor * ffn_w2,
    struct ggml_tensor * ffn_w3,
    int layer_idx,
    KVCache * kv_cache,
    int start_pos) {

    int seq_len = x->ne[1];
    const int num_heads = NUM_HEADS;
    const int num_kv_heads = NUM_KV_HEADS;
    const int head_dim = HEAD_DIM;

    // Pre-attention RMSNorm
    struct ggml_tensor * normed = ops::rms_norm(ctx, x, attn_norm_weight, 1e-6f);

    // Compute Q, K, V for current tokens
    struct ggml_tensor * Q = gqa_q_proj(ctx, normed, q_weight);  // [num_heads * head_dim, seq_len]
    struct ggml_tensor * K_new = gqa_kv_proj(ctx, normed, k_weight);  // [num_kv_heads * head_dim, seq_len]
    struct ggml_tensor * V_new = gqa_kv_proj(ctx, normed, v_weight);

    // Reshape to separate heads
    // Q: [head_dim, seq_len, num_heads]
    Q = ggml_reshape_4d(ctx, Q, head_dim, num_heads, seq_len, 1);
    Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
    Q = ggml_reshape_3d(ctx, Q, head_dim, seq_len, num_heads);

    // K_new, V_new: [head_dim, seq_len, num_kv_heads]
    K_new = ggml_reshape_4d(ctx, K_new, head_dim, num_kv_heads, seq_len, 1);
    K_new = ggml_cont(ctx, ggml_permute(ctx, K_new, 0, 2, 1, 3));
    K_new = ggml_reshape_3d(ctx, K_new, head_dim, seq_len, num_kv_heads);

    V_new = ggml_reshape_4d(ctx, V_new, head_dim, num_kv_heads, seq_len, 1);
    V_new = ggml_cont(ctx, ggml_permute(ctx, V_new, 0, 2, 1, 3));
    V_new = ggml_reshape_3d(ctx, V_new, head_dim, seq_len, num_kv_heads);

    // Apply Q/K normalization (RMSNorm per head) - Qwen3 specific
    // CRITICAL: Qwen3 applies QK normalization BEFORE RoPE
    if (q_norm_weight != nullptr) {
        Q = ops::rms_norm(ctx, Q, q_norm_weight, 1e-6f);
    }
    if (k_norm_weight != nullptr) {
        K_new = ops::rms_norm(ctx, K_new, k_norm_weight, 1e-6f);
    }

    // Apply RoPE (Rotary Position Embeddings) to Q and K_new
    // CRITICAL: RoPE must be applied AFTER Q/K norm and BEFORE caching K
    // Position for each token: start_pos, start_pos+1, ..., start_pos+seq_len-1
    {
        // Create position tensor for this sequence segment
        struct ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len);
        int32_t * pos_data = (int32_t *)pos->data;
        for (int i = 0; i < seq_len; i++) {
            pos_data[i] = start_pos + i;
        }

        // Apply RoPE with freq_base=10000 (standard Qwen3 setting)
        // Permute Q: [head_dim, seq_len, num_heads] -> [head_dim, num_heads, seq_len, 1] for ggml_rope
        Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
        Q = ggml_rope(ctx, Q, pos, head_dim, 0);  // Apply RoPE to all head_dim dimensions
        Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));  // Back to [head_dim, seq_len, num_heads]

        // Apply RoPE to K_new (before caching)
        K_new = ggml_cont(ctx, ggml_permute(ctx, K_new, 0, 2, 1, 3));
        K_new = ggml_rope(ctx, K_new, pos, head_dim, 0);
        K_new = ggml_cont(ctx, ggml_permute(ctx, K_new, 0, 2, 1, 3));  // Back to [head_dim, seq_len, num_kv_heads]
    }

    // Get full K, V (cached + new)
    struct ggml_tensor * K_full = K_new;
    struct ggml_tensor * V_full = V_new;

    if (kv_cache && kv_cache->get_seq_len(layer_idx) > 0) {
        // Get cached K, V
        struct ggml_tensor * K_cached = kv_cache->get_k_tensor(ctx, layer_idx);
        struct ggml_tensor * V_cached = kv_cache->get_v_tensor(ctx, layer_idx);

        // Concatenate: [head_dim, cached_len + seq_len, num_kv_heads]
        K_full = ggml_concat(ctx, K_cached, K_new, 1);
        V_full = ggml_concat(ctx, V_cached, V_new, 1);
    }

    // Compute attention with full K, V
    struct ggml_tensor * attn_out = cached_attention(ctx, Q, K_full, V_full, o_weight, num_heads, num_kv_heads);

    // Residual connection
    struct ggml_tensor * x_residual = ggml_add(ctx, x, attn_out);

    // Pre-FFN RMSNorm
    struct ggml_tensor * ffn_normed = ops::rms_norm(ctx, x_residual, ffn_norm_weight, 1e-6f);

    // SwiGLU FFN
    struct ggml_tensor * ffn_out = swiglu_ffn(ctx, ffn_normed, ffn_w1, ffn_w2, ffn_w3);

    // Residual connection
    struct ggml_tensor * output = ggml_add(ctx, x_residual, ffn_out);

    return output;
}

// Update KV cache after forward pass
// Must be called after computing the graph to copy K, V data to cache
void update_kv_cache(
    KVCache * kv_cache,
    int layer_idx,
    struct ggml_tensor * K_new,
    struct ggml_tensor * V_new) {

    if (!kv_cache || !K_new || !V_new) return;

    int head_dim = K_new->ne[0];
    int seq_len = K_new->ne[1];
    int num_kv_heads = K_new->ne[2];

    // Copy K, V data to cache
    kv_cache->append(layer_idx, (float *)K_new->data, (float *)V_new->data, seq_len);
}

} // namespace model
} // namespace leaxer_qwen
