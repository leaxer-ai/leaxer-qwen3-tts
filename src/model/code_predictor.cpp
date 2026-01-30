// Code Predictor Model
// 5-layer transformer with 15 output heads (one per acoustic codebook)
// Generates codebooks 1-15 autoregressively given talker hidden state and codebook 0
//
// CORRECT INPUT FORMAT (per Python reference):
//   input_embeds = concat([talker_hidden_state, codebook_0_embed], dim=1)
//   Shape: [batch=1, seq_len=2, hidden=1024]
//
// AUTOREGRESSIVE GENERATION:
//   For each codebook i (1 to 15):
//     1. Run transformer forward on accumulated input
//     2. Apply lm_head[i-1] to last position hidden state
//     3. Sample token for codebook i
//     4. Embed with codec_embedding[i-1]
//     5. Append embedding to input for next iteration

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
constexpr int NUM_CODEBOOKS = 16;          // Total codebooks (0-15)
constexpr int NUM_PREDICTED_CODEBOOKS = 15; // Code predictor generates codebooks 1-15
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

// Lookup embedding for a single token from an embedding table
// Handles different tensor layouts [vocab, hidden] or [hidden, vocab]
static void lookup_embedding(float * out, struct ggml_tensor * emb_table, int token_id, int hidden_dim) {
    const float * emb_data = (const float *)emb_table->data;
    int vocab_size = emb_table->ne[0];
    int emb_dim = emb_table->ne[1];
    
    // Clamp token to valid range
    if (token_id < 0) token_id = 0;
    
    if (emb_dim == hidden_dim) {
        // Layout: [vocab, hidden_dim]
        if (token_id >= vocab_size) token_id = vocab_size - 1;
        for (int d = 0; d < hidden_dim; d++) {
            out[d] = emb_data[token_id * hidden_dim + d];
        }
    } else {
        // Layout: [hidden_dim, vocab] - transposed
        if (token_id >= emb_dim) token_id = emb_dim - 1;
        for (int d = 0; d < hidden_dim; d++) {
            out[d] = emb_data[d * emb_dim + token_id];
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
    // For short sequences (code predictor uses seq_len=2..17), this is manageable
    size_t compute_mem = (size_t)256 * 1024 * 1024;  // 256MB per layer

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

// Run all 5 transformer layers on the input
// Input shape: [seq_len, hidden_dim]
// Output shape: [seq_len, hidden_dim]
static bool run_transformer_stack(
    float * output,
    const float * input,
    int seq_len,
    int hidden_dim,
    struct ggml_tensor ** layer_weights) {
    
    float * hidden = (float *)malloc(seq_len * hidden_dim * sizeof(float));
    float * hidden_tmp = (float *)malloc(seq_len * hidden_dim * sizeof(float));
    
    if (!hidden || !hidden_tmp) {
        free(hidden);
        free(hidden_tmp);
        return false;
    }
    
    // Copy input to hidden
    memcpy(hidden, input, seq_len * hidden_dim * sizeof(float));
    
    // Run through 5 layers
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
            fprintf(stderr, "Warning: transformer layer %d failed\n", layer);
            free(hidden);
            free(hidden_tmp);
            return false;
        }

        // Swap buffers for next layer
        float * tmp = hidden;
        hidden = hidden_tmp;
        hidden_tmp = tmp;
    }
    
    // Copy result to output
    memcpy(output, hidden, seq_len * hidden_dim * sizeof(float));
    
    free(hidden);
    free(hidden_tmp);
    return true;
}

// Apply output head (linear projection) to get logits for a codebook
// hidden_in: [hidden_dim] - the hidden state at the last position
// output: logits [vocab_size]
static void apply_output_head(float * logits, const float * hidden_in, 
                               struct ggml_tensor * head, int hidden_dim) {
    const float * head_data = (const float *)head->data;
    // GGUF shape [hidden_dim, vocab_size] = [1024, 2048]
    // ne[0] = hidden_dim (input), ne[1] = vocab_size (output)
    int head_in_dim = head->ne[0];
    int head_out_dim = head->ne[1];
    
    memset(logits, 0, head_out_dim * sizeof(float));
    
    // Linear: output[v] = sum_d(hidden[d] * weight[d, v])
    // GGUF layout: weight[d, v] = head_data[d + v * head_in_dim]
    for (int d = 0; d < head_in_dim && d < hidden_dim; d++) {
        float h = hidden_in[d];
        for (int v = 0; v < head_out_dim; v++) {
            logits[v] += h * head_data[d + v * head_in_dim];
        }
    }
}

// Code predictor forward pass - FIXED IMPLEMENTATION
//
// CORRECT ALGORITHM (per Python reference):
//   For each audio frame t:
//     1. Construct initial input: concat([talker_hidden[t], cb0_embed[t]], dim=1)
//        Shape: [2, 1024]
//     2. Autoregressive generation of codebooks 1-15:
//        For codebook i (1 to 15):
//          a. Run transformer on accumulated input
//          b. Apply RMS norm to last position
//          c. Apply lm_head[i-1] to get logits
//          d. Sample token (argmax for now)
//          e. Embed token with codec_embedding[i-1]
//          f. Append embedding to input for next iteration
//
// Parameters:
//   talker_hidden_states: [seq_len, hidden_dim] - Hidden states from talker for each frame
//                         If NULL, use codebook 0 embedding only (backward compat)
//   codebook_0_tokens:    [seq_len] int32 - First codebook tokens from talker
//   codec_embeddings:     Array of 15 embedding tables for codebooks 1-15
//                         codec_embeddings[i] embeds tokens for codebook i+1
//   layer_weights:        Weights for 5 transformer layers (9 weights per layer)
//   output_norm_weight:   Final RMS norm weight
//   output_heads:         15 output projection heads (lm_head[i] predicts codebook i+1)
//
// Output: [seq_len, 16] int32 tensor with all codebook tokens
struct ggml_tensor * code_predictor_forward(
    struct ggml_context * ctx,
    struct ggml_tensor * codebook_0_tokens,     // [seq_len] - codebook 0 tokens
    struct ggml_tensor ** codec_embeddings,     // 15 embedding tables (cb 1-15)
    struct ggml_tensor ** layer_weights,        // 5 layers × 9 weights
    struct ggml_tensor * output_norm_weight,
    struct ggml_tensor ** output_heads,         // 15 output heads
    int hidden_dim,
    int seq_len,
    struct ggml_tensor * talker_hidden_states,  // [seq_len, hidden_dim] - from talker (optional)
    struct ggml_tensor * talker_codec_embedding // Talker's codec embedding for cb0 lookup
) {
    printf("  Running code predictor (FIXED: correct input format)\n");
    printf("    seq_len=%d, hidden_dim=%d\n", seq_len, hidden_dim);
    printf("    talker_hidden_states=%s, talker_codec_embedding=%s\n",
           talker_hidden_states ? "provided" : "NULL",
           talker_codec_embedding ? "provided" : "NULL");
    fflush(stdout);

    // Validate inputs
    if (!codebook_0_tokens || !codec_embeddings || !layer_weights ||
        !output_norm_weight || !output_heads) {
        fprintf(stderr, "Error: null input to code_predictor_forward\n");
        return nullptr;
    }

    const int32_t * cb0_tokens = (const int32_t *)codebook_0_tokens->data;
    const float * norm_weight = (const float *)output_norm_weight->data;
    
    // Check if we have talker hidden states for proper input construction
    const float * talker_hidden_data = nullptr;
    if (talker_hidden_states) {
        talker_hidden_data = (const float *)talker_hidden_states->data;
    }

    // Allocate output buffer for all codebook predictions [seq_len, 16]
    int32_t * all_codes = (int32_t *)malloc(NUM_CODEBOOKS * seq_len * sizeof(int32_t));
    if (!all_codes) {
        fprintf(stderr, "Error: failed to allocate code predictor output buffer\n");
        return nullptr;
    }

    // Copy codebook 0 tokens from input
    for (int t = 0; t < seq_len; t++) {
        all_codes[t * NUM_CODEBOOKS + 0] = cb0_tokens[t];
    }

    // Allocate working buffers
    // Max sequence length during autoregression: 2 (initial) + 15 (generated) = 17
    const int MAX_AR_SEQ = 17;
    float * input_embeds = (float *)malloc(MAX_AR_SEQ * hidden_dim * sizeof(float));
    float * transformer_out = (float *)malloc(MAX_AR_SEQ * hidden_dim * sizeof(float));
    float * normed_hidden = (float *)malloc(hidden_dim * sizeof(float));
    float * logits = (float *)malloc(CODEBOOK_VOCAB * sizeof(float));
    float * cb0_embed = (float *)malloc(hidden_dim * sizeof(float));

    if (!input_embeds || !transformer_out || !normed_hidden || !logits || !cb0_embed) {
        fprintf(stderr, "Error: failed to allocate working buffers\n");
        free(all_codes);
        free(input_embeds);
        free(transformer_out);
        free(normed_hidden);
        free(logits);
        free(cb0_embed);
        return nullptr;
    }

    // Process each audio frame
    for (int t = 0; t < seq_len; t++) {
        int cb0_token = cb0_tokens[t];
        
        // Step 1: Construct initial input = concat([talker_hidden[t], cb0_embed], dim=1)
        // Result shape: [2, hidden_dim]
        
        // Position 0: talker hidden state (or zeros if not provided)
        if (talker_hidden_data) {
            memcpy(&input_embeds[0], &talker_hidden_data[t * hidden_dim], 
                   hidden_dim * sizeof(float));
        } else {
            // Fallback: use zeros (this is WRONG but maintains backward compat)
            memset(&input_embeds[0], 0, hidden_dim * sizeof(float));
        }
        
        // Position 1: codebook 0 embedding
        // Use talker's codec embedding table if provided, otherwise use first code predictor embedding
        if (talker_codec_embedding) {
            lookup_embedding(cb0_embed, talker_codec_embedding, cb0_token, hidden_dim);
        } else if (codec_embeddings[0]) {
            // Fallback: use code predictor's first embedding table
            lookup_embedding(cb0_embed, codec_embeddings[0], cb0_token, hidden_dim);
        } else {
            memset(cb0_embed, 0, hidden_dim * sizeof(float));
        }
        memcpy(&input_embeds[1 * hidden_dim], cb0_embed, hidden_dim * sizeof(float));
        
        int current_seq_len = 2;  // Initial: [talker_hidden, cb0_embed]

        // Step 2: Autoregressive generation of codebooks 1-15
        for (int cb = 1; cb < NUM_CODEBOOKS; cb++) {
            // Run transformer stack
            bool ok = run_transformer_stack(
                transformer_out, input_embeds, 
                current_seq_len, hidden_dim, layer_weights);
            
            if (!ok) {
                fprintf(stderr, "Warning: transformer failed at frame %d, codebook %d\n", t, cb);
                all_codes[t * NUM_CODEBOOKS + cb] = 0;
                continue;
            }
            
            // Apply RMS norm to the last position's hidden state
            const float * last_hidden = &transformer_out[(current_seq_len - 1) * hidden_dim];
            float ss = 0.0f;
            for (int d = 0; d < hidden_dim; d++) {
                ss += last_hidden[d] * last_hidden[d];
            }
            float rms = sqrtf(ss / hidden_dim + 1e-6f);
            for (int d = 0; d < hidden_dim; d++) {
                normed_hidden[d] = (last_hidden[d] / rms) * norm_weight[d];
            }
            
            // Apply output head for this codebook (lm_head[cb-1] predicts codebook cb)
            struct ggml_tensor * head = output_heads[cb - 1];
            if (!head) {
                all_codes[t * NUM_CODEBOOKS + cb] = 0;
                continue;
            }
            apply_output_head(logits, normed_hidden, head, hidden_dim);
            
            // Sample token (argmax)
            int head_out_dim = head->ne[1];
            int token = argmax(logits, head_out_dim < CODEBOOK_VOCAB ? head_out_dim : CODEBOOK_VOCAB);
            all_codes[t * NUM_CODEBOOKS + cb] = token;
            
            // Embed the sampled token using codec_embedding[cb-1]
            // codec_embedding[i] is for codebook i+1, so codec_embedding[cb-1] is for codebook cb
            if (cb < NUM_CODEBOOKS && codec_embeddings[cb - 1]) {
                float * new_embed = &input_embeds[current_seq_len * hidden_dim];
                lookup_embedding(new_embed, codec_embeddings[cb - 1], token, hidden_dim);
                current_seq_len++;
            }
        }

        // Progress indicator
        if ((t + 1) % 20 == 0 || t == seq_len - 1) {
            printf("    Frame %d/%d complete\n", t + 1, seq_len);
            fflush(stdout);
        }
    }

    // Free working buffers
    free(input_embeds);
    free(transformer_out);
    free(normed_hidden);
    free(logits);
    free(cb0_embed);

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

    // Create output tensor
    struct ggml_tensor * output = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, NUM_CODEBOOKS, seq_len);
    memcpy(output->data, all_codes, NUM_CODEBOOKS * seq_len * sizeof(int32_t));
    free(all_codes);

    printf("  Code predictor complete.\n");
    fflush(stdout);

    return output;
}

// Legacy wrapper for backward compatibility
// This version doesn't have talker hidden states - uses degraded mode
struct ggml_tensor * code_predictor_forward(
    struct ggml_context * ctx,
    struct ggml_tensor * semantic_codes,
    struct ggml_tensor ** codec_embeddings,
    struct ggml_tensor ** layer_weights,
    struct ggml_tensor * output_norm_weight,
    struct ggml_tensor ** output_heads,
    int hidden_dim,
    int seq_len) {
    
    printf("  WARNING: code_predictor_forward called without talker hidden states!\n");
    printf("           Using degraded mode (zeros for talker hidden). This WILL produce poor results.\n");
    fflush(stdout);
    
    // Call the full implementation with NULL for talker hidden states
    return code_predictor_forward(
        ctx,
        semantic_codes,  // codebook_0_tokens
        codec_embeddings,
        layer_weights,
        output_norm_weight,
        output_heads,
        hidden_dim,
        seq_len,
        nullptr,  // talker_hidden_states = NULL
        nullptr   // talker_codec_embedding = NULL
    );
}

} // namespace model
} // namespace leaxer_qwen
