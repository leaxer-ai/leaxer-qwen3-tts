// Code Predictor Model
// 5-layer transformer with 15 output heads (one per acoustic codebook)
// Refines codec token predictions across codebook hierarchy

#include "ggml.h"
#include "ggml-cpu.h"
#include "common.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

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

// Helper: argmax over an array
static int argmax(const float * data, int n) {
    int best = 0;
    float best_val = data[0];
    for (int i = 1; i < n; i++) {
        if (data[i] > best_val) {
            best_val = data[i];
            best = i;
        }
    }
    return best;
}

// RMS Norm helper - direct CPU implementation
static void rms_norm_cpu(float * out, const float * x, const float * weight,
                         int seq_len, int hidden_dim, float eps) {
    for (int t = 0; t < seq_len; t++) {
        // Compute sum of squares
        float ss = 0.0f;
        for (int d = 0; d < hidden_dim; d++) {
            float v = x[t * hidden_dim + d];
            ss += v * v;
        }
        float rms = sqrtf(ss / hidden_dim + eps);

        // Normalize and scale
        for (int d = 0; d < hidden_dim; d++) {
            out[t * hidden_dim + d] = (x[t * hidden_dim + d] / rms) * weight[d];
        }
    }
}

// Execute a single transformer layer using ggml
// Returns true on success, output is written to hidden_out
static bool run_transformer_layer(
    float * hidden_out,
    const float * hidden_in,
    int seq_len,
    int hidden_dim,
    struct ggml_tensor * attn_norm,
    struct ggml_tensor * q_weight,
    struct ggml_tensor * k_weight,
    struct ggml_tensor * v_weight,
    struct ggml_tensor * o_weight,
    struct ggml_tensor * ffn_norm,
    struct ggml_tensor * ffn_w1,
    struct ggml_tensor * ffn_w2,
    struct ggml_tensor * ffn_w3) {

    // Calculate memory needed for single layer
    // Attention: Q,K,V projections + scores + output (O(seq_len^2))
    // FFN: gate, up, down projections
    // For seq_len=2000+, attention alone needs ~300MB
    size_t compute_mem = (size_t)1024 * 1024 * 1024;  // 1GB per layer

    struct ggml_init_params params = {
        .mem_size   = compute_mem,
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };

    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }

    // Create input tensor
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, seq_len);
    memcpy(x->data, hidden_in, seq_len * hidden_dim * sizeof(float));

    // Run transformer layer
    struct ggml_tensor * out = code_pred_transformer_layer(
        ctx, x,
        attn_norm, q_weight, k_weight, v_weight, o_weight,
        ffn_norm, ffn_w1, ffn_w2, ffn_w3
    );

    // Build and execute graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    int n_threads = 4;
    struct ggml_cplan plan = ggml_graph_plan(gf, n_threads, nullptr);
    if (plan.work_size > 0) {
        plan.work_data = (uint8_t *)malloc(plan.work_size);
    }
    ggml_graph_compute(gf, &plan);
    if (plan.work_data) {
        free(plan.work_data);
    }

    // Copy output
    memcpy(hidden_out, out->data, seq_len * hidden_dim * sizeof(float));

    ggml_free(ctx);
    return true;
}

// Code predictor forward pass - WITH TRANSFORMER LAYERS
// Processes transformer layers one at a time to manage memory
//
// Algorithm:
//   - Codebook 0: Semantic codes from input (from main LLM)
//   - For codebook cb (1 to 15):
//     1. Sum embeddings from codebooks 0..cb-1
//     2. Run through 5 transformer layers (one layer at a time)
//     3. Apply final RMS norm
//     4. Project with output head for codebook cb
//     5. Argmax to get codes
struct ggml_tensor * code_predictor_forward(
    struct ggml_context * ctx,
    struct ggml_tensor * semantic_codes,
    struct ggml_tensor ** codec_embeddings,  // Array of 16 embedding tables
    struct ggml_tensor ** layer_weights,     // Weights for 5 transformer layers (9 weights per layer)
    struct ggml_tensor * output_norm_weight,
    struct ggml_tensor ** output_heads,      // 15 output projection heads (codebooks 1-15)
    int hidden_dim,
    int seq_len) {

    // For long sequences, transformer needs too much memory (O(seq_len^2) attention)
    // Use transformer for seq_len <= 512, simplified version for longer
    constexpr int MAX_TRANSFORMER_SEQ = 512;
    bool use_transformer = (seq_len <= MAX_TRANSFORMER_SEQ);

    printf("  Running code predictor (%d codebooks, seq_len=%d, transformer=%s)\n",
           NUM_CODEBOOKS, seq_len, use_transformer ? "yes" : "no (seq too long)");
    fflush(stdout);

    // Validate inputs
    if (!semantic_codes || !codec_embeddings || !layer_weights ||
        !output_norm_weight || !output_heads) {
        fprintf(stderr, "Error: null input to code_predictor_forward\n");
        return nullptr;
    }

    // Allocate output buffer for all codebook predictions
    int32_t * all_codes = (int32_t *)malloc(NUM_CODEBOOKS * seq_len * sizeof(int32_t));
    if (!all_codes) {
        fprintf(stderr, "Error: failed to allocate code predictor output buffer\n");
        return nullptr;
    }

    // Copy semantic codes (codebook 0) from input
    const int32_t * semantic_data = (const int32_t *)semantic_codes->data;
    for (int t = 0; t < seq_len; t++) {
        all_codes[t * NUM_CODEBOOKS + 0] = semantic_data[t];
    }

    // Allocate hidden state buffers [seq_len, hidden_dim]
    float * hidden = (float *)calloc(seq_len * hidden_dim, sizeof(float));
    float * hidden_tmp = (float *)malloc(seq_len * hidden_dim * sizeof(float));
    float * logits = (float *)malloc(CODEBOOK_VOCAB * sizeof(float));

    if (!hidden || !hidden_tmp || !logits) {
        fprintf(stderr, "Error: failed to allocate buffers\n");
        free(all_codes);
        free(hidden);
        free(hidden_tmp);
        free(logits);
        return nullptr;
    }

    const float * norm_weight = (const float *)output_norm_weight->data;

    // Generate codebooks 1-15 autoregressively
    for (int cb = 1; cb < NUM_CODEBOOKS; cb++) {
        // Step 1: Aggregate embeddings from codebooks 0..cb-1
        memset(hidden, 0, seq_len * hidden_dim * sizeof(float));

        for (int prev_cb = 0; prev_cb < cb; prev_cb++) {
            struct ggml_tensor * emb_table = codec_embeddings[prev_cb];
            if (!emb_table) continue;

            const float * emb_data = (const float *)emb_table->data;
            int vocab_size = emb_table->ne[0];

            for (int t = 0; t < seq_len; t++) {
                int32_t idx = all_codes[t * NUM_CODEBOOKS + prev_cb];
                if (idx < 0) idx = 0;
                if (idx >= vocab_size) idx = vocab_size - 1;

                // Determine embedding layout and lookup
                if (emb_table->ne[1] == hidden_dim) {
                    // [vocab, hidden_dim] layout
                    for (int d = 0; d < hidden_dim; d++) {
                        hidden[t * hidden_dim + d] += emb_data[idx * hidden_dim + d];
                    }
                } else {
                    // [hidden_dim, vocab] layout
                    for (int d = 0; d < hidden_dim; d++) {
                        hidden[t * hidden_dim + d] += emb_data[d * vocab_size + idx];
                    }
                }
            }
        }

        // Step 2: Run through 5 transformer layers (one at a time)
        // Skip for long sequences due to O(seq_len^2) memory requirements
        if (use_transformer) {
            for (int layer = 0; layer < CODE_PRED_LAYERS; layer++) {
                int base = layer * 9;
                bool ok = run_transformer_layer(
                    hidden_tmp, hidden, seq_len, hidden_dim,
                    layer_weights[base + 0],  // attn_norm
                    layer_weights[base + 1],  // q_weight
                    layer_weights[base + 2],  // k_weight
                    layer_weights[base + 3],  // v_weight
                    layer_weights[base + 4],  // o_weight
                    layer_weights[base + 5],  // ffn_norm
                    layer_weights[base + 6],  // ffn_w1 (gate)
                    layer_weights[base + 7],  // ffn_w2 (up)
                    layer_weights[base + 8]   // ffn_w3 (down)
                );

                if (!ok) {
                    fprintf(stderr, "Warning: transformer layer %d failed for codebook %d\n", layer, cb);
                    break;
                }

                // Swap buffers for next layer
                float * tmp = hidden;
                hidden = hidden_tmp;
                hidden_tmp = tmp;
            }
        }

        // Step 3: Apply final RMS norm
        const float * norm_input = use_transformer ? hidden : hidden;
        rms_norm_cpu(hidden_tmp, norm_input, norm_weight, seq_len, hidden_dim, 1e-6f);

        // Step 4: Apply output head and argmax
        struct ggml_tensor * head = output_heads[cb - 1];
        if (!head) {
            for (int t = 0; t < seq_len; t++) {
                all_codes[t * NUM_CODEBOOKS + cb] = 0;
            }
            continue;
        }

        const float * head_data = (const float *)head->data;
        int head_out_dim = head->ne[0];
        int head_in_dim = head->ne[1];

        for (int t = 0; t < seq_len; t++) {
            memset(logits, 0, CODEBOOK_VOCAB * sizeof(float));

            for (int d = 0; d < head_in_dim && d < hidden_dim; d++) {
                float h = hidden_tmp[t * hidden_dim + d];
                for (int v = 0; v < head_out_dim && v < CODEBOOK_VOCAB; v++) {
                    logits[v] += h * head_data[v + d * head_out_dim];
                }
            }

            all_codes[t * NUM_CODEBOOKS + cb] = argmax(logits, head_out_dim);
        }

        // Progress indicator
        if (cb % 3 == 0 || cb == NUM_CODEBOOKS - 1) {
            printf("    Codebook %d/%d complete\n", cb, NUM_CODEBOOKS - 1);
            fflush(stdout);
        }
    }

    // Debug: print code distribution summary
    printf("  Code distribution summary:\n");
    for (int cb = 0; cb < NUM_CODEBOOKS; cb += 4) {
        int min_code = CODEBOOK_VOCAB, max_code = 0;
        long sum = 0;
        for (int t = 0; t < seq_len; t++) {
            int code = all_codes[t * NUM_CODEBOOKS + cb];
            if (code < min_code) min_code = code;
            if (code > max_code) max_code = code;
            sum += code;
        }
        printf("    CB%2d: min=%4d, max=%4d, mean=%.1f\n",
               cb, min_code, max_code, (float)sum / seq_len);
    }
    fflush(stdout);

    free(logits);
    free(hidden_tmp);
    free(hidden);

    struct ggml_tensor * output = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, NUM_CODEBOOKS, seq_len);
    memcpy(output->data, all_codes, NUM_CODEBOOKS * seq_len * sizeof(int32_t));
    free(all_codes);

    printf("  Code predictor complete.\n");
    fflush(stdout);

    return output;
}

} // namespace model
} // namespace leaxer_qwen
