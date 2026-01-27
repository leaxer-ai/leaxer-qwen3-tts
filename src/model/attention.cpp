// Grouped Query Attention (GQA)
// 16 query heads, 2 key-value heads
// Used in main Qwen3 transformer

#include "ggml.h"
#include "common.h"

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

// TODO: Implement remaining GQA components
// Key features:
// - K/V projections (shared across query head groups)
// - RoPE position embeddings
// - Attention computation with causal masking

} // namespace model
} // namespace leaxer_qwen
