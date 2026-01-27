// Full Qwen3-TTS Model
// Integrates: Tokenizer → LLM → Code Predictor → Vocoder

#include "ggml.h"
#include "common.h"

namespace leaxer_qwen {

// Forward declarations from ops
namespace ops {
struct ggml_tensor * rms_norm(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * weight,
    float eps);
}

// Forward declarations from model
namespace model {
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
    struct ggml_tensor * ffn_w3);
}

namespace model {

// Model variants
// Qwen3-TTS-12Hz-1.7B: 20 layers, 1024 hidden
// Qwen3-TTS-12Hz-0.6B: 12 layers, 896 hidden

// Special token IDs
constexpr int IM_START_TOKEN_ID = 151644;
constexpr int IM_END_TOKEN_ID = 151645;
constexpr int TTS_PAD_TOKEN_ID = 151671;
constexpr int TTS_BOS_TOKEN_ID = 151672;
constexpr int TTS_EOS_TOKEN_ID = 151673;
constexpr int CODEC_PAD_ID = 4196;
constexpr int CODEC_BOS_ID = 4197;
constexpr int CODEC_EOS_ID = 4198;

// LLM Forward Pass
// Architecture: Embedding → N Transformer Blocks → Final Norm → Output Projection
// Input: token_ids with shape [seq_len]
// Weights:
//   - embed_weight: [vocab_size, hidden_dim] embedding matrix
//   - layer_X_*: weights for each transformer layer (X = 0 to n_layers-1)
//   - norm_weight: [hidden_dim] final RMSNorm weight
//   - lm_head_weight: [vocab_size, hidden_dim] output projection
// Output: [vocab_size, seq_len] logits for next token prediction
struct ggml_tensor * llm_forward(
    struct ggml_context * ctx,
    struct ggml_tensor * token_ids,
    struct ggml_tensor * embed_weight,
    struct ggml_tensor ** layer_weights,  // Array of pointers to layer weights
    int n_layers,
    struct ggml_tensor * norm_weight,
    struct ggml_tensor * lm_head_weight) {

    // Step 1: Token Embedding
    // token_ids: [seq_len]
    // embed_weight: [vocab_size, hidden_dim]
    // Output: [hidden_dim, seq_len]
    struct ggml_tensor * embedded = ggml_get_rows(ctx, embed_weight, token_ids);

    // Step 2: Pass through N transformer blocks
    struct ggml_tensor * hidden = embedded;
    for (int i = 0; i < n_layers; i++) {
        // Each layer expects 10 weight tensors in order:
        // 0: attn_norm_weight
        // 1: q_weight
        // 2: k_weight
        // 3: v_weight
        // 4: o_weight
        // 5: ffn_norm_weight
        // 6: ffn_w1
        // 7: ffn_w2
        // 8: ffn_w3
        // 9: (reserved for future use)
        struct ggml_tensor ** layer_w = &layer_weights[i * 10];

        hidden = transformer_block(
            ctx,
            hidden,
            layer_w[0],  // attn_norm_weight
            layer_w[1],  // q_weight
            layer_w[2],  // k_weight
            layer_w[3],  // v_weight
            layer_w[4],  // o_weight
            layer_w[5],  // ffn_norm_weight
            layer_w[6],  // ffn_w1
            layer_w[7],  // ffn_w2
            layer_w[8]   // ffn_w3
        );
    }

    // Step 3: Final RMSNorm
    // hidden: [hidden_dim, seq_len]
    // norm_weight: [hidden_dim]
    struct ggml_tensor * normalized = ops::rms_norm(ctx, hidden, norm_weight, 1e-6f);

    // Step 4: Output projection to vocabulary
    // normalized: [hidden_dim, seq_len]
    // lm_head_weight: [vocab_size, hidden_dim]
    // Output: [vocab_size, seq_len]
    struct ggml_tensor * logits = ggml_mul_mat(ctx, lm_head_weight, normalized);

    return logits;
}

// TODO: Implement full TTS model
// Pipeline:
// 1. Format prompt: "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
// 2. Tokenize text
// 3. Run through LLM to generate codec tokens
// 4. Refine with code predictor
// 5. Decode with vocoder

} // namespace model
} // namespace leaxer_qwen
