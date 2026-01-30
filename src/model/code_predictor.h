// Code Predictor Header
// 5-layer transformer that generates codebooks 1-15 autoregressively
// given talker hidden state and codebook 0 token

#ifndef LEAXER_QWEN_CODE_PREDICTOR_H
#define LEAXER_QWEN_CODE_PREDICTOR_H

#include "ggml.h"

namespace leaxer_qwen {
namespace model {

// Code predictor configuration constants
constexpr int CODE_PRED_NUM_LAYERS = 5;
constexpr int CODE_PRED_NUM_HEADS = 16;
constexpr int CODE_PRED_NUM_KV_HEADS = 8;
constexpr int CODE_PRED_NUM_CODEBOOKS = 16;          // Total codebooks (0-15)
constexpr int CODE_PRED_VOCAB_SIZE = 2048;           // Per-codebook vocabulary

// Code Predictor Forward Pass (FULL version with talker hidden states)
//
// INPUT FORMAT (per Python reference):
//   input_embeds = concat([talker_hidden_state, codebook_0_embed], dim=1)
//   Shape: [batch=1, seq_len=2, hidden=1024]
//
// AUTOREGRESSIVE GENERATION:
//   For each audio frame, generate codebooks 1-15:
//     1. Run transformer on accumulated input
//     2. Apply lm_head[i-1] to last position
//     3. Sample token for codebook i
//     4. Embed with codec_embedding[i-1]
//     5. Append to input for next iteration
//
// Parameters:
//   ctx:                   ggml context for output tensor allocation
//   codebook_0_tokens:     [seq_len] int32 - First codebook tokens from talker
//   codec_embeddings:      Array of 15 embedding tables for codebooks 1-15
//                          codec_embeddings[i] embeds tokens for codebook i+1
//   layer_weights:         Weights for 5 transformer layers (9 weights per layer)
//                          Order per layer: attn_norm, q, k, v, o, ffn_norm, w1, w2, w3
//   output_norm_weight:    Final RMS norm weight [hidden_dim]
//   output_heads:          15 output projection heads
//                          output_heads[i] predicts codebook i+1
//   hidden_dim:            Hidden dimension (1024)
//   seq_len:               Number of audio frames to process
//   talker_hidden_states:  [seq_len, hidden_dim] - Hidden states from talker
//                          CRITICAL: This is the hidden state BEFORE lm_head projection
//                          If NULL, uses degraded mode (zeros) which produces poor results
//   talker_codec_embedding: Talker's codec embedding table [vocab=3072, dim=1024]
//                          Used to embed codebook 0 tokens
//                          If NULL, falls back to code predictor's first embedding
//
// Returns:
//   [seq_len, 16] int32 tensor with all codebook tokens (0-15)
//   Codebook 0 values are copied from input
//   Codebooks 1-15 are generated autoregressively
struct ggml_tensor * code_predictor_forward(
    struct ggml_context * ctx,
    struct ggml_tensor * codebook_0_tokens,     // [seq_len] int32
    struct ggml_tensor ** codec_embeddings,     // 15 embedding tables
    struct ggml_tensor ** layer_weights,        // 5×9 layer weights
    struct ggml_tensor * output_norm_weight,    // [hidden_dim]
    struct ggml_tensor ** output_heads,         // 15 output heads
    int hidden_dim,
    int seq_len,
    struct ggml_tensor * talker_hidden_states,  // [seq_len, hidden_dim] - from talker
    struct ggml_tensor * talker_codec_embedding // [vocab, hidden_dim] - talker's cb embedding
);

// Code Predictor Forward Pass (LEGACY backward-compatible version)
//
// WARNING: This version does NOT receive talker hidden states!
// It uses zeros for the talker hidden state portion of the input,
// which WILL produce poor quality results.
//
// This overload exists only for backward compatibility.
// Prefer the full version above that takes talker_hidden_states.
struct ggml_tensor * code_predictor_forward(
    struct ggml_context * ctx,
    struct ggml_tensor * semantic_codes,        // [seq_len] int32 - codebook 0 tokens
    struct ggml_tensor ** codec_embeddings,     // 15 embedding tables
    struct ggml_tensor ** layer_weights,        // 5×9 layer weights
    struct ggml_tensor * output_norm_weight,    // [hidden_dim]
    struct ggml_tensor ** output_heads,         // 15 output heads
    int hidden_dim,
    int seq_len);

} // namespace model
} // namespace leaxer_qwen

#endif // LEAXER_QWEN_CODE_PREDICTOR_H
