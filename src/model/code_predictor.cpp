// Code Predictor Model
// 5-layer transformer with 16 output heads (one per codebook)
// Refines codec token predictions across codebook hierarchy

#include "ggml.h"
#include "common.h"
#include <cstring>

namespace leaxer_qwen {

// Forward declarations from other modules
namespace ops {
struct ggml_tensor * rms_norm(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * weight,
    float eps);
}

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

// Code predictor configuration
constexpr int CODE_PRED_LAYERS = 5;
constexpr int CODE_PRED_HEADS = 16;
constexpr int CODE_PRED_KV_HEADS = 8;
constexpr int NUM_CODEBOOKS = 16;
constexpr int CODEBOOK_VOCAB = 2048;

// Code predictor transformer layer
// Same architecture as main transformer but for code refinement
struct ggml_tensor * code_pred_transformer_layer(
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

    // Pre-normalization for attention
    struct ggml_tensor * normed = ops::rms_norm(ctx, x, attn_norm_weight, 1e-6f);

    // Attention projections
    struct ggml_tensor * Q = gqa_q_proj(ctx, normed, q_weight);
    struct ggml_tensor * K = gqa_kv_proj(ctx, normed, k_weight);
    struct ggml_tensor * V = gqa_kv_proj(ctx, normed, v_weight);

    // Reshape for multi-head attention
    int seq_len = Q->ne[1];
    int q_dim = Q->ne[0];
    const int num_heads = CODE_PRED_HEADS;
    const int num_kv_heads = CODE_PRED_KV_HEADS;
    const int head_dim = q_dim / num_heads;

    Q = ggml_reshape_4d(ctx, Q, head_dim, num_heads, seq_len, 1);
    Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
    Q = ggml_reshape_3d(ctx, Q, head_dim, seq_len, num_heads);

    K = ggml_reshape_4d(ctx, K, head_dim, num_kv_heads, seq_len, 1);
    K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
    K = ggml_reshape_3d(ctx, K, head_dim, seq_len, num_kv_heads);

    V = ggml_reshape_4d(ctx, V, head_dim, num_kv_heads, seq_len, 1);
    V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));
    V = ggml_reshape_3d(ctx, V, head_dim, seq_len, num_kv_heads);

    // Compute attention
    struct ggml_tensor * scores = attention_scores(ctx, Q, K);
    struct ggml_tensor * attn_out = attention_output(ctx, scores, V, o_weight);

    // First residual connection
    struct ggml_tensor * x_residual = ggml_add(ctx, x, attn_out);

    // Pre-normalization for FFN
    struct ggml_tensor * ffn_normed = ops::rms_norm(ctx, x_residual, ffn_norm_weight, 1e-6f);

    // Apply FFN
    struct ggml_tensor * ffn_out = swiglu_ffn(ctx, ffn_normed, ffn_w1, ffn_w2, ffn_w3);

    // Second residual connection
    struct ggml_tensor * output = ggml_add(ctx, x_residual, ffn_out);

    return output;
}

// Code predictor forward pass
// Takes semantic codes from LLM and generates 16 codebook tokens per timestep
// Uses autoregressive generation: each codebook conditions on previous codebooks
// Input: semantic_codes [seq_len] - semantic code indices
// Codec embeddings: 16 embedding tables, each [CODEBOOK_VOCAB, hidden_dim]
// Output: [NUM_CODEBOOKS, seq_len] int32 - codebook indices
struct ggml_tensor * code_predictor_forward(
    struct ggml_context * ctx,
    struct ggml_tensor * semantic_codes,
    struct ggml_tensor ** codec_embeddings,  // Array of 16 embedding tables
    struct ggml_tensor ** layer_weights,     // Weights for 5 transformer layers
    struct ggml_tensor * output_norm_weight,
    struct ggml_tensor ** output_heads,      // 16 output projection heads
    int hidden_dim,
    int seq_len) {

    // Step 1: Embed semantic codes using first codebook embedding
    // semantic_codes: [seq_len] with int32 indices
    // codec_embeddings[0]: [CODEBOOK_VOCAB, hidden_dim]
    // Output: [hidden_dim, seq_len]
    struct ggml_tensor * hidden = ggml_get_rows(ctx, codec_embeddings[0], semantic_codes);

    // Store all codebook predictions
    struct ggml_tensor * codebook_predictions[NUM_CODEBOOKS];

    // Codebook 0 is the semantic code from LLM
    codebook_predictions[0] = semantic_codes;

    // Step 2: Autoregressive generation for codebooks 1-15
    // Each codebook prediction conditions on all previous codebook embeddings
    for (int cb = 1; cb < NUM_CODEBOOKS; cb++) {
        // Process through 5 transformer layers
        struct ggml_tensor * layer_output = hidden;
        for (int layer = 0; layer < CODE_PRED_LAYERS; layer++) {
            int base_idx = layer * 9;
            layer_output = code_pred_transformer_layer(
                ctx,
                layer_output,
                layer_weights[base_idx + 0],  // attn_norm_weight
                layer_weights[base_idx + 1],  // q_weight
                layer_weights[base_idx + 2],  // k_weight
                layer_weights[base_idx + 3],  // v_weight
                layer_weights[base_idx + 4],  // o_weight
                layer_weights[base_idx + 5],  // ffn_norm_weight
                layer_weights[base_idx + 6],  // ffn_w1
                layer_weights[base_idx + 7],  // ffn_w2
                layer_weights[base_idx + 8]   // ffn_w3
            );
        }

        // Final normalization before projection
        struct ggml_tensor * normed = ops::rms_norm(ctx, layer_output, output_norm_weight, 1e-6f);

        // Project to vocabulary space for this codebook
        // output_heads[cb]: [hidden_dim, CODEBOOK_VOCAB]
        // normed: [hidden_dim, seq_len]
        // logits: [CODEBOOK_VOCAB, seq_len]
        struct ggml_tensor * logits = ggml_mul_mat(ctx, output_heads[cb], normed);

        // Get argmax along vocab dimension (dim 0)
        // Result: [seq_len] int32 with codebook indices
        codebook_predictions[cb] = ggml_argmax(ctx, logits);

        // Add this codebook's embedding to hidden state for next iteration
        // This is the autoregressive step: next codebook conditions on this one
        struct ggml_tensor * cb_embedding = ggml_get_rows(ctx, codec_embeddings[cb], codebook_predictions[cb]);
        hidden = ggml_add(ctx, hidden, cb_embedding);
    }

    // Step 3: Stack all codebook predictions into [NUM_CODEBOOKS, seq_len]
    struct ggml_tensor * reshaped[NUM_CODEBOOKS];
    for (int cb = 0; cb < NUM_CODEBOOKS; cb++) {
        reshaped[cb] = ggml_reshape_2d(ctx, codebook_predictions[cb], seq_len, 1);
    }

    // Concatenate along dimension 1 to form [seq_len, NUM_CODEBOOKS]
    struct ggml_tensor * output = reshaped[0];
    for (int cb = 1; cb < NUM_CODEBOOKS; cb++) {
        output = ggml_concat(ctx, output, reshaped[cb], 1);
    }

    // Transpose to get [NUM_CODEBOOKS, seq_len]
    output = ggml_cont(ctx, ggml_permute(ctx, output, 1, 0, 2, 3));

    return output;
}

} // namespace model
} // namespace leaxer_qwen
