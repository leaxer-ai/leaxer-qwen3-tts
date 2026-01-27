// Code Predictor Model
// 5-layer transformer with 15 output heads (one per acoustic codebook)
// Refines codec token predictions across codebook hierarchy

#include "ggml.h"
#include "ggml-cpu.h"
#include "common.h"
#include <cstdio>
#include <cstdlib>
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

    // GQA: expand K and V to match Q's head count (8 KV heads -> 16 Q heads)
    if (num_kv_heads < num_heads) {
        K = ggml_repeat(ctx, K, Q);
        V = ggml_repeat(ctx, V, Q);
    }

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

// Code predictor forward pass - SIMPLIFIED VERSION
// For now, skip the transformer and just copy semantic codes to all codebooks
// This allows testing the vocoder pipeline while we figure out the memory-efficient code predictor
// TODO: Implement proper code predictor with chunked attention or flash attention
struct ggml_tensor * code_predictor_forward(
    struct ggml_context * ctx,
    struct ggml_tensor * semantic_codes,
    struct ggml_tensor ** codec_embeddings,  // Array of 15 embedding tables (indices 0-14)
    struct ggml_tensor ** layer_weights,     // Weights for 5 transformer layers
    struct ggml_tensor * output_norm_weight,
    struct ggml_tensor ** output_heads,      // 15 output projection heads (codebooks 1-15)
    int hidden_dim,
    int seq_len) {

    (void)codec_embeddings;
    (void)layer_weights;
    (void)output_norm_weight;
    (void)output_heads;
    (void)hidden_dim;

    printf("  [STUB] Using simplified code predictor (semantic codes only)\n");
    fflush(stdout);

    // Allocate output buffer for all codebook predictions
    int32_t * all_predictions = (int32_t *)malloc(NUM_CODEBOOKS * seq_len * sizeof(int32_t));
    if (!all_predictions) {
        fprintf(stderr, "Error: failed to allocate code predictor output buffer\n");
        return nullptr;
    }

    // Copy semantic codes to all 16 codebooks (as placeholder)
    // This won't produce good audio but will test the vocoder pipeline
    // Memory layout: [t0_cb0, t0_cb1, ..., t0_cb15, t1_cb0, t1_cb1, ...]
    // This matches vocoder expectation: codes[t * NUM_CODEBOOKS + cb]
    const int32_t * semantic_data = (const int32_t *)semantic_codes->data;
    for (int t = 0; t < seq_len; t++) {
        for (int cb = 0; cb < NUM_CODEBOOKS; cb++) {
            all_predictions[t * NUM_CODEBOOKS + cb] = semantic_data[t];
        }
    }

    // Create output tensor
    // Shape: [NUM_CODEBOOKS, seq_len] where ne[0]=NUM_CODEBOOKS is the fast dimension
    // This matches memory layout: data[t * NUM_CODEBOOKS + cb]
    struct ggml_tensor * output = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, NUM_CODEBOOKS, seq_len);
    memcpy(output->data, all_predictions, NUM_CODEBOOKS * seq_len * sizeof(int32_t));
    free(all_predictions);

    return output;
}

} // namespace model
} // namespace leaxer_qwen
