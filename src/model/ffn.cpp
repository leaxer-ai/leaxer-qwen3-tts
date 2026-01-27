// SwiGLU Feed-Forward Network (Qwen style)
// FFN(x) = SiLU(W1 @ x) * (W2 @ x) @ W3

#include "ggml.h"
#include "common.h"

namespace leaxer_qwen {
namespace model {

// SwiGLU FFN
// Implements: SiLU(W1 @ x) * (W2 @ x) then W3 projection
// Input: x with shape [hidden_dim, seq_len, batch]
// W1: gate projection [hidden_dim, intermediate_dim]
// W2: up projection [hidden_dim, intermediate_dim]
// W3: down projection [intermediate_dim, hidden_dim]
// Output: [hidden_dim, seq_len, batch]
struct ggml_tensor * swiglu_ffn(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * w1,
    struct ggml_tensor * w2,
    struct ggml_tensor * w3) {

    // Gate projection: W1 @ x
    // w1: [hidden_dim, intermediate_dim]
    // x: [hidden_dim, seq_len, batch]
    // gate: [intermediate_dim, seq_len, batch]
    struct ggml_tensor * gate = ggml_mul_mat(ctx, w1, x);

    // Apply SiLU activation: x * sigmoid(x)
    gate = ggml_silu(ctx, gate);

    // Up projection: W2 @ x
    // w2: [hidden_dim, intermediate_dim]
    // x: [hidden_dim, seq_len, batch]
    // up: [intermediate_dim, seq_len, batch]
    struct ggml_tensor * up = ggml_mul_mat(ctx, w2, x);

    // Element-wise multiply: SiLU(W1 @ x) * (W2 @ x)
    struct ggml_tensor * gated = ggml_mul(ctx, gate, up);

    // Down projection: W3 @ gated
    // w3: [intermediate_dim, hidden_dim]
    // gated: [intermediate_dim, seq_len, batch]
    // output: [hidden_dim, seq_len, batch]
    struct ggml_tensor * output = ggml_mul_mat(ctx, w3, gated);

    return output;
}

} // namespace model
} // namespace leaxer_qwen
