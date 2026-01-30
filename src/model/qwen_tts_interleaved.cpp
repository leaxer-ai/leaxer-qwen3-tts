// Interleaved Talker + Code Predictor Generation
// This is the CORRECT implementation following Python reference
//
// ARCHITECTURE (from Python):
// FOR EACH frame:
//   1. Talker forward → sample codebook 0 token, capture hidden state
//   2. Code_predictor(talker_hidden, cb0_embed) → codebooks 1-15
//   3. Get embeddings for ALL 16 codebook tokens
//   4. Sum ALL 16 embeddings + trailing_text_embed
//   5. Feed sum as next input to talker
//   6. Repeat until EOS

#include "ggml.h"
#include "ggml-cpu.h"
#include "common.h"
#include "kv_cache.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>

namespace leaxer_qwen {

// Forward declaration from ops namespace
namespace ops {
struct ggml_tensor * rms_norm(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * weight,
    float eps);
}

namespace model {

// Forward declaration of cached transformer block
struct ggml_tensor * transformer_block_cached(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * attn_norm_weight,
    struct ggml_tensor * q_weight,
    struct ggml_tensor * k_weight,
    struct ggml_tensor * v_weight,
    struct ggml_tensor * o_weight,
    struct ggml_tensor * q_norm_weight,
    struct ggml_tensor * k_norm_weight,
    struct ggml_tensor * ffn_norm_weight,
    struct ggml_tensor * ffn_w1,
    struct ggml_tensor * ffn_w2,
    struct ggml_tensor * ffn_w3,
    int layer_idx,
    KVCache * kv_cache,
    int start_pos);

// Special token IDs
constexpr int TTS_PAD_TOKEN_ID = 151671;
constexpr int CODEC_PAD_ID = 2148;
constexpr int CODEC_EOS_ID = 2150;

// Model dimensions
constexpr int TEXT_EMBED_DIM = 2048;
constexpr int HIDDEN_DIM = 1024;
constexpr int CODEC_VOCAB_SIZE = 3072;
constexpr int NUM_CODEBOOKS = 16;
constexpr int NUM_PREDICTED_CODEBOOKS = 15;
constexpr int CODEBOOK_VOCAB = 2048;

// Token suppression range (suppress special tokens except EOS)
constexpr int CODEC_SUPPRESS_START = 2048;

// ============================================================================
// Text Projection MLP: 2048 → 2048 → SiLU → 1024
// ============================================================================
static struct ggml_tensor * text_projection(
    struct ggml_context * ctx,
    struct ggml_tensor * input,
    struct ggml_tensor * fc1_weight,
    struct ggml_tensor * fc1_bias,
    struct ggml_tensor * fc2_weight,
    struct ggml_tensor * fc2_bias)
{
    struct ggml_tensor * proj = ggml_mul_mat(ctx, fc1_weight, input);
    struct ggml_tensor * bias1 = fc1_bias;
    if (fc1_bias->type == GGML_TYPE_F16) {
        bias1 = ggml_cast(ctx, fc1_bias, GGML_TYPE_F32);
    }
    proj = ggml_add(ctx, proj, bias1);
    proj = ggml_silu(ctx, proj);
    proj = ggml_mul_mat(ctx, fc2_weight, proj);
    struct ggml_tensor * bias2 = fc2_bias;
    if (fc2_bias->type == GGML_TYPE_F16) {
        bias2 = ggml_cast(ctx, fc2_bias, GGML_TYPE_F32);
    }
    proj = ggml_add(ctx, proj, bias2);
    return proj;
}

// ============================================================================
// Token Sampling with suppression
// ============================================================================
static int sample_token(
    const float * logits,
    int vocab_size,
    float temperature,
    int top_k,
    float top_p,
    uint64_t * rng_state,
    int eos_token_id = -1)
{
    // Create modified logits with suppressed tokens
    float * mod_logits = new float[vocab_size];
    memcpy(mod_logits, logits, vocab_size * sizeof(float));

    // Suppress tokens in range [2048, vocab_size) except EOS
    for (int i = CODEC_SUPPRESS_START; i < vocab_size; i++) {
        if (i != eos_token_id) {
            mod_logits[i] = -1e10f;
        }
    }

    // Greedy sampling
    if (temperature <= 0.0f) {
        int max_idx = 0;
        float max_val = mod_logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (mod_logits[i] > max_val) {
                max_val = mod_logits[i];
                max_idx = i;
            }
        }
        delete[] mod_logits;
        return max_idx;
    }

    // Temperature + softmax
    struct TokenProb { int id; float prob; };
    TokenProb * candidates = new TokenProb[vocab_size];

    float max_logit = mod_logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (mod_logits[i] > max_logit) max_logit = mod_logits[i];
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        candidates[i].id = i;
        candidates[i].prob = expf((mod_logits[i] - max_logit) / temperature);
        sum_exp += candidates[i].prob;
    }
    delete[] mod_logits;

    for (int i = 0; i < vocab_size; i++) {
        candidates[i].prob /= sum_exp;
    }

    // Sort by probability (descending)
    for (int i = 0; i < vocab_size - 1; i++) {
        for (int j = i + 1; j < vocab_size; j++) {
            if (candidates[j].prob > candidates[i].prob) {
                TokenProb tmp = candidates[i];
                candidates[i] = candidates[j];
                candidates[j] = tmp;
            }
        }
    }

    // Apply top-k
    int n_candidates = vocab_size;
    if (top_k > 0 && top_k < vocab_size) {
        n_candidates = top_k;
    }

    // Apply top-p
    if (top_p < 1.0f) {
        float cumsum = 0.0f;
        int nucleus_size = 0;
        for (int i = 0; i < n_candidates; i++) {
            cumsum += candidates[i].prob;
            nucleus_size++;
            if (cumsum >= top_p) break;
        }
        n_candidates = nucleus_size;
    }

    // Renormalize
    float prob_sum = 0.0f;
    for (int i = 0; i < n_candidates; i++) {
        prob_sum += candidates[i].prob;
    }
    for (int i = 0; i < n_candidates; i++) {
        candidates[i].prob /= prob_sum;
    }

    // Sample
    *rng_state = (*rng_state * 1664525ULL + 1013904223ULL) & 0xFFFFFFFFULL;
    float rand_val = (float)(*rng_state) / (float)0x100000000ULL;

    float cumsum = 0.0f;
    for (int i = 0; i < n_candidates; i++) {
        cumsum += candidates[i].prob;
        if (rand_val < cumsum) {
            int result = candidates[i].id;
            delete[] candidates;
            return result;
        }
    }

    int result = candidates[n_candidates - 1].id;
    delete[] candidates;
    return result;
}

// ============================================================================
// Code Predictor Single Frame
// Generates codebooks 1-15 for ONE frame
// ============================================================================
// Forward declarations for code predictor helpers
static void lookup_embedding(float * out, struct ggml_tensor * emb_table, int token_id, int hidden_dim);
static bool run_code_pred_transformer_stack(
    float * output,
    const float * input,
    int seq_len,
    int hidden_dim,
    struct ggml_tensor ** layer_weights);
static void apply_code_pred_output_head(float * logits, const float * hidden_in, 
                                         struct ggml_tensor * head, int hidden_dim);
static int argmax_suppress_codebook(const float * data, int n);

// Single-frame code predictor
// Input: talker_hidden [HIDDEN_DIM], cb0_token (int)
// Output: cb_tokens [NUM_CODEBOOKS] (cb0 is passed in, this fills 1-15)
static bool code_predictor_single_frame(
    int * cb_tokens,                              // OUTPUT: all 16 codebook tokens (cb0 pre-filled)
    const float * talker_hidden,                  // [HIDDEN_DIM] - talker's hidden state for this frame
    int cb0_token,                                // Codebook 0 token (already sampled)
    struct ggml_tensor * talker_codec_embedding,  // Talker's codec embedding for cb0 lookup
    struct ggml_tensor ** codec_embeddings,       // 15 embedding tables for cb 1-15
    struct ggml_tensor ** layer_weights,          // 5 layers × 11 weights
    struct ggml_tensor * output_norm_weight,      // Final RMS norm
    struct ggml_tensor ** output_heads)           // 15 output heads
{
    const float * norm_weight = (const float *)output_norm_weight->data;
    
    // Fill in cb0
    cb_tokens[0] = cb0_token;
    
    // Allocate working buffers
    const int MAX_AR_SEQ = 17;  // Initial 2 + up to 15 generated
    float * input_embeds = (float *)malloc(MAX_AR_SEQ * HIDDEN_DIM * sizeof(float));
    float * transformer_out = (float *)malloc(MAX_AR_SEQ * HIDDEN_DIM * sizeof(float));
    float * normed_hidden = (float *)malloc(HIDDEN_DIM * sizeof(float));
    float * logits = (float *)malloc(CODEBOOK_VOCAB * sizeof(float));
    float * cb0_embed = (float *)malloc(HIDDEN_DIM * sizeof(float));
    
    if (!input_embeds || !transformer_out || !normed_hidden || !logits || !cb0_embed) {
        free(input_embeds); free(transformer_out); free(normed_hidden); free(logits); free(cb0_embed);
        return false;
    }
    
    // Step 1: Construct initial input = concat([talker_hidden, cb0_embed])
    // Position 0: talker hidden state
    memcpy(&input_embeds[0], talker_hidden, HIDDEN_DIM * sizeof(float));
    
    // Position 1: codebook 0 embedding
    if (talker_codec_embedding) {
        lookup_embedding(cb0_embed, talker_codec_embedding, cb0_token, HIDDEN_DIM);
    } else {
        memset(cb0_embed, 0, HIDDEN_DIM * sizeof(float));
    }
    memcpy(&input_embeds[HIDDEN_DIM], cb0_embed, HIDDEN_DIM * sizeof(float));
    
    int current_seq_len = 2;
    
    // Step 2: Autoregressive generation of codebooks 1-15
    for (int cb = 1; cb < NUM_CODEBOOKS; cb++) {
        // Run transformer stack
        bool ok = run_code_pred_transformer_stack(
            transformer_out, input_embeds, 
            current_seq_len, HIDDEN_DIM, layer_weights);
        
        if (!ok) {
            cb_tokens[cb] = 0;
            continue;
        }
        
        // RMS norm on last position
        const float * last_hidden = &transformer_out[(current_seq_len - 1) * HIDDEN_DIM];
        float ss = 0.0f;
        for (int d = 0; d < HIDDEN_DIM; d++) {
            ss += last_hidden[d] * last_hidden[d];
        }
        float rms = sqrtf(ss / HIDDEN_DIM + 1e-6f);
        for (int d = 0; d < HIDDEN_DIM; d++) {
            normed_hidden[d] = (last_hidden[d] / rms) * norm_weight[d];
        }
        
        // Apply output head for this codebook
        struct ggml_tensor * head = output_heads[cb - 1];
        if (!head) {
            cb_tokens[cb] = 0;
            continue;
        }
        apply_code_pred_output_head(logits, normed_hidden, head, HIDDEN_DIM);
        
        // Sample token (argmax with suppression)
        int head_out_dim = head->ne[1];
        int token = argmax_suppress_codebook(logits, head_out_dim < CODEBOOK_VOCAB ? head_out_dim : CODEBOOK_VOCAB);
        cb_tokens[cb] = token;
        
        // Embed and append for next iteration
        if (cb < NUM_CODEBOOKS && codec_embeddings[cb - 1]) {
            float * new_embed = &input_embeds[current_seq_len * HIDDEN_DIM];
            lookup_embedding(new_embed, codec_embeddings[cb - 1], token, HIDDEN_DIM);
            current_seq_len++;
        }
    }
    
    free(input_embeds);
    free(transformer_out);
    free(normed_hidden);
    free(logits);
    free(cb0_embed);
    
    return true;
}

// ============================================================================
// Embedding lookup helper
// ============================================================================
static void lookup_embedding(float * out, struct ggml_tensor * emb_table, int token_id, int hidden_dim) {
    const float * emb_data = (const float *)emb_table->data;
    int vocab_size = emb_table->ne[0];
    int emb_dim = emb_table->ne[1];
    
    if (token_id < 0) token_id = 0;
    
    if (emb_dim == hidden_dim) {
        // Layout: [vocab, hidden_dim]
        if (token_id >= vocab_size) token_id = vocab_size - 1;
        for (int d = 0; d < hidden_dim; d++) {
            out[d] = emb_data[token_id * hidden_dim + d];
        }
    } else {
        // Layout: [hidden_dim, vocab]
        if (token_id >= emb_dim) token_id = emb_dim - 1;
        for (int d = 0; d < hidden_dim; d++) {
            out[d] = emb_data[d * emb_dim + token_id];
        }
    }
}

// ============================================================================
// Sum embeddings for all 16 codebooks
// ============================================================================
static void sum_codebook_embeddings(
    float * sum_out,                              // OUTPUT: [HIDDEN_DIM]
    const int * cb_tokens,                        // [NUM_CODEBOOKS]
    struct ggml_tensor * talker_codec_embedding,  // For cb0
    struct ggml_tensor ** codec_embeddings)       // 15 tables for cb 1-15
{
    float * embed_buf = (float *)malloc(HIDDEN_DIM * sizeof(float));
    
    // Initialize sum to zero
    memset(sum_out, 0, HIDDEN_DIM * sizeof(float));
    
    // Add codebook 0 embedding
    lookup_embedding(embed_buf, talker_codec_embedding, cb_tokens[0], HIDDEN_DIM);
    for (int d = 0; d < HIDDEN_DIM; d++) {
        sum_out[d] += embed_buf[d];
    }
    
    // Add codebooks 1-15 embeddings
    for (int cb = 1; cb < NUM_CODEBOOKS; cb++) {
        if (codec_embeddings[cb - 1]) {
            lookup_embedding(embed_buf, codec_embeddings[cb - 1], cb_tokens[cb], HIDDEN_DIM);
            for (int d = 0; d < HIDDEN_DIM; d++) {
                sum_out[d] += embed_buf[d];
            }
        }
    }
    
    free(embed_buf);
}

// ============================================================================
// Forward declarations for code predictor transformer
// ============================================================================
static int argmax_suppress_codebook(const float * data, int n) {
    int best = 0;
    float best_val = -1e10f;
    int effective_n = (n > 2047 + 1) ? 2048 : n;
    for (int i = 0; i < effective_n; i++) {
        if (data[i] > best_val) {
            best_val = data[i];
            best = i;
        }
    }
    return best;
}

static void apply_code_pred_output_head(float * logits, const float * hidden_in, 
                                         struct ggml_tensor * head, int hidden_dim) {
    const float * head_data = (const float *)head->data;
    int head_in_dim = head->ne[0];
    int head_out_dim = head->ne[1];
    
    memset(logits, 0, head_out_dim * sizeof(float));
    
    for (int d = 0; d < head_in_dim && d < hidden_dim; d++) {
        float h = hidden_in[d];
        for (int v = 0; v < head_out_dim; v++) {
            logits[v] += h * head_data[d + v * head_in_dim];
        }
    }
}

// Code predictor transformer layer (uses external implementation)
extern struct ggml_tensor * code_pred_transformer_layer(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * attn_norm_weight,
    struct ggml_tensor * q_weight,
    struct ggml_tensor * k_weight,
    struct ggml_tensor * v_weight,
    struct ggml_tensor * o_weight,
    struct ggml_tensor * q_norm_weight,
    struct ggml_tensor * k_norm_weight,
    struct ggml_tensor * ffn_norm_weight,
    struct ggml_tensor * ffn_w1,
    struct ggml_tensor * ffn_w2,
    struct ggml_tensor * ffn_w3,
    int start_pos);

static bool run_code_pred_layer(
    float * hidden_out,
    const float * hidden_in,
    int seq_len,
    int hidden_dim,
    struct ggml_tensor * attn_norm,
    struct ggml_tensor * q_weight,
    struct ggml_tensor * k_weight,
    struct ggml_tensor * v_weight,
    struct ggml_tensor * o_weight,
    struct ggml_tensor * q_norm,
    struct ggml_tensor * k_norm,
    struct ggml_tensor * ffn_norm,
    struct ggml_tensor * ffn_w1,
    struct ggml_tensor * ffn_w2,
    struct ggml_tensor * ffn_w3,
    int start_pos = 0)
{
    size_t compute_mem = 256ULL * 1024 * 1024;
    struct ggml_init_params params = {
        .mem_size = compute_mem,
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;
    
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, seq_len);
    memcpy(x->data, hidden_in, seq_len * hidden_dim * sizeof(float));
    
    struct ggml_tensor * out = code_pred_transformer_layer(
        ctx, x, attn_norm, q_weight, k_weight, v_weight, o_weight,
        q_norm, k_norm, ffn_norm, ffn_w1, ffn_w2, ffn_w3, start_pos);
    
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    
    int n_threads = 4;
    struct ggml_cplan plan = ggml_graph_plan(gf, n_threads, nullptr);
    if (plan.work_size > 0) {
        plan.work_data = (uint8_t *)malloc(plan.work_size);
    }
    ggml_graph_compute(gf, &plan);
    if (plan.work_data) free(plan.work_data);
    
    memcpy(hidden_out, out->data, seq_len * hidden_dim * sizeof(float));
    ggml_free(ctx);
    return true;
}

static bool run_code_pred_transformer_stack(
    float * output,
    const float * input,
    int seq_len,
    int hidden_dim,
    struct ggml_tensor ** layer_weights)
{
    float * hidden = (float *)malloc(seq_len * hidden_dim * sizeof(float));
    float * hidden_tmp = (float *)malloc(seq_len * hidden_dim * sizeof(float));
    
    if (!hidden || !hidden_tmp) {
        free(hidden); free(hidden_tmp);
        return false;
    }
    
    memcpy(hidden, input, seq_len * hidden_dim * sizeof(float));
    
    // 5 layers, 11 weights per layer
    for (int layer = 0; layer < 5; layer++) {
        int base = layer * 11;
        bool ok = run_code_pred_layer(
            hidden_tmp, hidden, seq_len, hidden_dim,
            layer_weights[base + 0],
            layer_weights[base + 1],
            layer_weights[base + 2],
            layer_weights[base + 3],
            layer_weights[base + 4],
            layer_weights[base + 5],
            layer_weights[base + 6],
            layer_weights[base + 7],
            layer_weights[base + 8],
            layer_weights[base + 9],
            layer_weights[base + 10],
            0);
        
        if (!ok) {
            free(hidden); free(hidden_tmp);
            return false;
        }
        
        float * tmp = hidden;
        hidden = hidden_tmp;
        hidden_tmp = tmp;
    }
    
    memcpy(output, hidden, seq_len * hidden_dim * sizeof(float));
    free(hidden);
    free(hidden_tmp);
    return true;
}

// ============================================================================
// MAIN FUNCTION: Interleaved Generation
// ============================================================================
// This is the CORRECT implementation that interleaves talker and code predictor
//
// Returns: [n_frames, NUM_CODEBOOKS] int32 tensor with all codebook tokens
//          n_frames is written to n_frames_out
// ============================================================================
int generate_interleaved(
    // Text tokens and codec tokens for prefill
    const int * text_tokens,
    const int * codec_tokens,      // Parallel codec tokens for prefill
    int prefill_len,
    // Text embedding and projection
    struct ggml_tensor * text_embed_weight,
    struct ggml_tensor * text_proj_fc1_weight,
    struct ggml_tensor * text_proj_fc1_bias,
    struct ggml_tensor * text_proj_fc2_weight,
    struct ggml_tensor * text_proj_fc2_bias,
    // Talker weights
    struct ggml_tensor * talker_codec_embedding,
    struct ggml_tensor ** talker_layer_weights,
    int n_talker_layers,
    struct ggml_tensor * talker_norm_weight,
    struct ggml_tensor * talker_lm_head_weight,
    // Code predictor weights
    struct ggml_tensor ** codec_embeddings,          // 15 tables for cb 1-15
    struct ggml_tensor ** code_pred_layer_weights,
    struct ggml_tensor * code_pred_norm_weight,
    struct ggml_tensor ** code_pred_output_heads,
    // TTS pad embedding for trailing (pre-computed, 1024-dim)
    const float * tts_pad_embed,
    // Generation parameters
    int max_frames,
    float temperature,
    int top_k,
    float top_p,
    uint64_t * rng_state,
    // Output
    int32_t * all_codes_out,    // OUTPUT: [max_frames, NUM_CODEBOOKS]
    int * n_frames_out)         // OUTPUT: number of frames generated
{
    printf("  Interleaved generation (correct flow)...\n");
    
    // KV cache config
    const int num_kv_heads = 8;
    const int head_dim = 128;
    const int max_seq_len = prefill_len + max_frames + 128;
    
    KVCache * kv_cache = KVCache::create(n_talker_layers, num_kv_heads, head_dim, max_seq_len);
    if (!kv_cache) {
        fprintf(stderr, "Failed to create KV cache\n");
        *n_frames_out = 0;
        return -1;
    }
    
    int current_pos = 0;
    int n_frames = 0;
    
    // Working buffers
    float * combined_embed = (float *)malloc(HIDDEN_DIM * sizeof(float));
    float * hidden_state = (float *)malloc(HIDDEN_DIM * sizeof(float));
    float * codebook_sum = (float *)malloc(HIDDEN_DIM * sizeof(float));
    int cb_tokens[NUM_CODEBOOKS];
    
    if (!combined_embed || !hidden_state || !codebook_sum) {
        free(combined_embed); free(hidden_state); free(codebook_sum);
        KVCache::destroy(kv_cache);
        *n_frames_out = 0;
        return -1;
    }
    
    // ========================================================================
    // Phase 1: PREFILL
    // ========================================================================
    printf("    Prefill (%d tokens)...\n", prefill_len);
    {
        size_t prefill_mem = 512ULL * 1024 * 1024;
        struct ggml_init_params params = {
            .mem_size = prefill_mem,
            .mem_buffer = nullptr,
            .no_alloc = false,
        };
        struct ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            free(combined_embed); free(hidden_state); free(codebook_sum);
            KVCache::destroy(kv_cache);
            *n_frames_out = 0;
            return -1;
        }
        
        // Create text token tensor
        struct ggml_tensor * text_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, prefill_len);
        memcpy(text_ids->data, text_tokens, prefill_len * sizeof(int32_t));
        
        // Create codec token tensor
        struct ggml_tensor * codec_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, prefill_len);
        int32_t * codec_data = (int32_t *)codec_ids->data;
        if (codec_tokens) {
            memcpy(codec_data, codec_tokens, prefill_len * sizeof(int32_t));
        } else {
            for (int i = 0; i < prefill_len; i++) {
                codec_data[i] = CODEC_PAD_ID;
            }
        }
        
        // Text embedding + projection
        struct ggml_tensor * text_embedded = ggml_get_rows(ctx, text_embed_weight, text_ids);
        struct ggml_tensor * text_proj = text_projection(
            ctx, text_embedded,
            text_proj_fc1_weight, text_proj_fc1_bias,
            text_proj_fc2_weight, text_proj_fc2_bias);
        
        // Codec embedding
        struct ggml_tensor * codec_embedded = ggml_get_rows(ctx, talker_codec_embedding, codec_ids);
        
        // Combined = text_proj + codec_embed (element-wise SUM!)
        struct ggml_tensor * combined = ggml_add(ctx, text_proj, codec_embedded);
        
        // Forward through transformer with caching
        struct ggml_tensor * x = combined;
        for (int i = 0; i < n_talker_layers; i++) {
            struct ggml_tensor ** layer_w = &talker_layer_weights[i * 11];
            x = transformer_block_cached(
                ctx, x,
                layer_w[0], layer_w[1], layer_w[2], layer_w[3], layer_w[4],
                layer_w[5], layer_w[6], layer_w[7], layer_w[8], layer_w[9], layer_w[10],
                i, kv_cache, 0);
        }
        
        // Final RMS norm
        struct ggml_tensor * normalized = leaxer_qwen::ops::rms_norm(ctx, x, talker_norm_weight, 1e-6f);
        
        // Output logits
        struct ggml_tensor * logits = ggml_mul_mat(ctx, talker_lm_head_weight, normalized);
        
        // Build and execute
        struct ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, logits);
        ggml_build_forward_expand(graph, normalized);
        ggml_graph_compute_with_ctx(ctx, graph, 4);
        
        // Sample first codec token
        int vocab_size = logits->ne[0];
        float * logits_data = (float *)logits->data;
        float * last_logits = logits_data + vocab_size * (prefill_len - 1);
        
        int first_cb0 = sample_token(last_logits, vocab_size, temperature, top_k, top_p, rng_state, CODEC_EOS_ID);
        
        // Capture hidden state for code predictor
        float * norm_data = (float *)normalized->data;
        memcpy(hidden_state, &norm_data[(prefill_len - 1) * HIDDEN_DIM], HIDDEN_DIM * sizeof(float));
        
        ggml_free(ctx);
        current_pos = prefill_len;
        
        // Check for EOS
        if (first_cb0 == CODEC_EOS_ID) {
            printf("    EOS at first token\n");
            free(combined_embed); free(hidden_state); free(codebook_sum);
            KVCache::destroy(kv_cache);
            *n_frames_out = 0;
            return 0;
        }
        
        // Generate codebooks 1-15 for first frame
        cb_tokens[0] = first_cb0;
        code_predictor_single_frame(
            cb_tokens, hidden_state, first_cb0,
            talker_codec_embedding, codec_embeddings,
            code_pred_layer_weights, code_pred_norm_weight, code_pred_output_heads);
        
        // Store first frame
        for (int cb = 0; cb < NUM_CODEBOOKS; cb++) {
            all_codes_out[n_frames * NUM_CODEBOOKS + cb] = cb_tokens[cb];
        }
        n_frames++;
        
        // Compute sum of all 16 codebook embeddings for next input
        sum_codebook_embeddings(codebook_sum, cb_tokens, talker_codec_embedding, codec_embeddings);
        
        // Add tts_pad_embed (we're past the text now)
        for (int d = 0; d < HIDDEN_DIM; d++) {
            codebook_sum[d] += tts_pad_embed[d];
        }
    }
    
    // ========================================================================
    // Phase 2: DECODE (Interleaved)
    // ========================================================================
    printf("    Decoding (interleaved)...\n");
    
    while (n_frames < max_frames) {
        size_t decode_mem = 256ULL * 1024 * 1024;
        struct ggml_init_params params = {
            .mem_size = decode_mem,
            .mem_buffer = nullptr,
            .no_alloc = false,
        };
        struct ggml_context * ctx = ggml_init(params);
        if (!ctx) break;
        
        // Input is the codebook sum from previous frame (already has tts_pad added)
        struct ggml_tensor * input_embed = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HIDDEN_DIM, 1);
        memcpy(input_embed->data, codebook_sum, HIDDEN_DIM * sizeof(float));
        
        // Forward through transformer
        struct ggml_tensor * x = input_embed;
        for (int i = 0; i < n_talker_layers; i++) {
            struct ggml_tensor ** layer_w = &talker_layer_weights[i * 11];
            x = transformer_block_cached(
                ctx, x,
                layer_w[0], layer_w[1], layer_w[2], layer_w[3], layer_w[4],
                layer_w[5], layer_w[6], layer_w[7], layer_w[8], layer_w[9], layer_w[10],
                i, kv_cache, current_pos);
        }
        
        // Final RMS norm
        struct ggml_tensor * normalized = leaxer_qwen::ops::rms_norm(ctx, x, talker_norm_weight, 1e-6f);
        
        // Output logits
        struct ggml_tensor * logits = ggml_mul_mat(ctx, talker_lm_head_weight, normalized);
        
        // Build and execute
        struct ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, logits);
        ggml_build_forward_expand(graph, normalized);
        ggml_graph_compute_with_ctx(ctx, graph, 4);
        
        // Sample codebook 0 token
        int vocab_size = logits->ne[0];
        float * logits_data = (float *)logits->data;
        int cb0_token = sample_token(logits_data, vocab_size, temperature, top_k, top_p, rng_state, CODEC_EOS_ID);
        
        // Capture hidden state
        memcpy(hidden_state, normalized->data, HIDDEN_DIM * sizeof(float));
        
        ggml_free(ctx);
        current_pos++;
        
        // Check for EOS
        if (cb0_token == CODEC_EOS_ID) {
            printf("    EOS at frame %d\n", n_frames);
            break;
        }
        
        // Generate codebooks 1-15
        cb_tokens[0] = cb0_token;
        code_predictor_single_frame(
            cb_tokens, hidden_state, cb0_token,
            talker_codec_embedding, codec_embeddings,
            code_pred_layer_weights, code_pred_norm_weight, code_pred_output_heads);
        
        // Store frame
        for (int cb = 0; cb < NUM_CODEBOOKS; cb++) {
            all_codes_out[n_frames * NUM_CODEBOOKS + cb] = cb_tokens[cb];
        }
        n_frames++;
        
        // Compute sum for next input
        sum_codebook_embeddings(codebook_sum, cb_tokens, talker_codec_embedding, codec_embeddings);
        for (int d = 0; d < HIDDEN_DIM; d++) {
            codebook_sum[d] += tts_pad_embed[d];
        }
        
        // Progress
        if (n_frames % 50 == 0) {
            printf("      %d frames...\n", n_frames);
        }
    }
    
    free(combined_embed);
    free(hidden_state);
    free(codebook_sum);
    KVCache::destroy(kv_cache);
    
    *n_frames_out = n_frames;
    printf("    Generated %d frames.\n", n_frames);
    return 0;
}

} // namespace model
} // namespace leaxer_qwen
