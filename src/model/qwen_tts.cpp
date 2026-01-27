// Full Qwen3-TTS Model
// Integrates: Tokenizer → LLM → Code Predictor → Vocoder

#include "ggml.h"
#include "ggml-cpu.h"
#include "common.h"
#include "kv_cache.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>

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

// Talker Forward Pass (Qwen3-TTS LLM)
// Architecture: Embedding → Text Projection → 28 Transformer Blocks (with RoPE) → Final Norm → Output Projection
// Input: token_ids with shape [seq_len]
// Weights:
//   - embed_weight: [vocab_size, embedding_dim] embedding matrix
//   - text_proj_fc1_weight, text_proj_fc1_bias: first layer of text projection (embedding_dim → embedding_dim)
//   - text_proj_fc2_weight, text_proj_fc2_bias: second layer of text projection (embedding_dim → hidden_dim)
//   - layer_X_*: weights for each transformer layer (X = 0 to n_layers-1)
//   - norm_weight: [hidden_dim] final RMSNorm weight
//   - lm_head_weight: [vocab_size, hidden_dim] output projection (semantic codebook)
// Output: [vocab_size, seq_len] logits for semantic token prediction
// Note: Uses RoPE (Rotary Position Embeddings) for position encoding
struct ggml_tensor * talker_forward(
    struct ggml_context * ctx,
    struct ggml_tensor * token_ids,
    struct ggml_tensor * embed_weight,
    struct ggml_tensor * text_proj_fc1_weight,
    struct ggml_tensor * text_proj_fc1_bias,
    struct ggml_tensor * text_proj_fc2_weight,
    struct ggml_tensor * text_proj_fc2_bias,
    struct ggml_tensor ** layer_weights,  // Array of pointers to layer weights
    int n_layers,
    struct ggml_tensor * norm_weight,
    struct ggml_tensor * lm_head_weight) {

    // Step 1: Token Embedding
    // token_ids: [seq_len]
    // embed_weight: [vocab_size, embedding_dim=2048]
    // Output: [embedding_dim, seq_len]
    struct ggml_tensor * embedded = ggml_get_rows(ctx, embed_weight, token_ids);

    // Step 2: Text Projection (embedding_dim=2048 → hidden_dim=1024)
    // Flow: input(2048) → fc1 → SiLU → fc2 → output(1024)
    // fc1: linear
    struct ggml_tensor * proj = ggml_mul_mat(ctx, text_proj_fc1_weight, embedded);
    // Cast bias to F32 if needed (weights may be F16)
    struct ggml_tensor * fc1_bias = text_proj_fc1_bias;
    if (text_proj_fc1_bias->type == GGML_TYPE_F16) {
        fc1_bias = ggml_cast(ctx, text_proj_fc1_bias, GGML_TYPE_F32);
    }
    proj = ggml_add(ctx, proj, fc1_bias);
    // SiLU activation
    proj = ggml_silu(ctx, proj);
    // fc2: linear
    proj = ggml_mul_mat(ctx, text_proj_fc2_weight, proj);
    // Cast bias to F32 if needed
    struct ggml_tensor * fc2_bias = text_proj_fc2_bias;
    if (text_proj_fc2_bias->type == GGML_TYPE_F16) {
        fc2_bias = ggml_cast(ctx, text_proj_fc2_bias, GGML_TYPE_F32);
    }
    proj = ggml_add(ctx, proj, fc2_bias);

    // Step 3: Pass through N transformer blocks (28 for 0.6B model)
    // Each block applies RoPE for position encoding
    struct ggml_tensor * hidden = proj;
    for (int i = 0; i < n_layers; i++) {
        // Each layer expects 9 weight tensors in order:
        // 0: attn_norm_weight
        // 1: q_weight
        // 2: k_weight
        // 3: v_weight
        // 4: o_weight
        // 5: ffn_norm_weight
        // 6: ffn_gate (w1)
        // 7: ffn_up (w2)
        // 8: ffn_down (w3)
        struct ggml_tensor ** layer_w = &layer_weights[i * 9];

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

    // Step 4: Output projection to semantic codebook vocabulary
    // normalized: [hidden_dim, seq_len]
    // lm_head_weight: [vocab_size, hidden_dim]
    // Output: [vocab_size, seq_len]
    struct ggml_tensor * logits = ggml_mul_mat(ctx, lm_head_weight, normalized);

    return logits;
}

// LLM Forward Pass (generic version, kept for compatibility)
// Architecture: Embedding → Text Projection → N Transformer Blocks → Final Norm → Output Projection
// Input: token_ids with shape [seq_len]
// Weights:
//   - embed_weight: [vocab_size, embedding_dim] embedding matrix
//   - text_proj_*: text projection weights (embedding_dim → hidden_dim)
//   - layer_X_*: weights for each transformer layer (X = 0 to n_layers-1)
//   - norm_weight: [hidden_dim] final RMSNorm weight
//   - lm_head_weight: [vocab_size, hidden_dim] output projection
// Output: [vocab_size, seq_len] logits for next token prediction
struct ggml_tensor * llm_forward(
    struct ggml_context * ctx,
    struct ggml_tensor * token_ids,
    struct ggml_tensor * embed_weight,
    struct ggml_tensor * text_proj_fc1_weight,
    struct ggml_tensor * text_proj_fc1_bias,
    struct ggml_tensor * text_proj_fc2_weight,
    struct ggml_tensor * text_proj_fc2_bias,
    struct ggml_tensor ** layer_weights,  // Array of pointers to layer weights
    int n_layers,
    struct ggml_tensor * norm_weight,
    struct ggml_tensor * lm_head_weight) {

    // Forward to talker_forward (same implementation)
    return talker_forward(ctx, token_ids, embed_weight,
                          text_proj_fc1_weight, text_proj_fc1_bias,
                          text_proj_fc2_weight, text_proj_fc2_bias,
                          layer_weights, n_layers, norm_weight, lm_head_weight);
}

// Token Sampling
// Implements temperature, top-k, and top-p (nucleus) sampling from logits
// Parameters:
//   - logits: [vocab_size] raw logits from model
//   - vocab_size: size of vocabulary
//   - temperature: sampling temperature (>1 = more random, <1 = more deterministic, 0 = greedy)
//   - top_k: keep only top-k highest probability tokens (0 = disabled)
//   - top_p: keep tokens with cumulative probability >= top_p (1.0 = disabled)
//   - rng_state: random number generator state (simple LCG)
// Returns: sampled token ID
int sample_token(
    const float * logits,
    int vocab_size,
    float temperature,
    int top_k,
    float top_p,
    uint64_t * rng_state) {

    // Greedy sampling (argmax)
    if (temperature <= 0.0f) {
        int max_idx = 0;
        float max_val = logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_idx = i;
            }
        }
        return max_idx;
    }

    // Allocate space for token index/probability pairs
    struct TokenProb {
        int id;
        float prob;
    };
    TokenProb * candidates = new TokenProb[vocab_size];

    // Apply temperature and softmax
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        candidates[i].id = i;
        candidates[i].prob = expf((logits[i] - max_logit) / temperature);
        sum_exp += candidates[i].prob;
    }

    // Normalize to probabilities
    for (int i = 0; i < vocab_size; i++) {
        candidates[i].prob /= sum_exp;
    }

    // Sort by probability (descending) for top-k/top-p
    for (int i = 0; i < vocab_size - 1; i++) {
        for (int j = i + 1; j < vocab_size; j++) {
            if (candidates[j].prob > candidates[i].prob) {
                TokenProb tmp = candidates[i];
                candidates[i] = candidates[j];
                candidates[j] = tmp;
            }
        }
    }

    // Apply top-k filtering
    int n_candidates = vocab_size;
    if (top_k > 0 && top_k < vocab_size) {
        n_candidates = top_k;
    }

    // Apply top-p (nucleus) filtering
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

    // Renormalize probabilities
    float prob_sum = 0.0f;
    for (int i = 0; i < n_candidates; i++) {
        prob_sum += candidates[i].prob;
    }
    for (int i = 0; i < n_candidates; i++) {
        candidates[i].prob /= prob_sum;
    }

    // Sample from the distribution using simple LCG RNG
    // LCG parameters: a=1664525, c=1013904223, m=2^32
    *rng_state = (*rng_state * 1664525ULL + 1013904223ULL) & 0xFFFFFFFFULL;
    float rand_val = (float)(*rng_state) / (float)0x100000000ULL;

    // Select token based on cumulative probability
    float cumsum = 0.0f;
    for (int i = 0; i < n_candidates; i++) {
        cumsum += candidates[i].prob;
        if (rand_val < cumsum) {
            int result = candidates[i].id;
            delete[] candidates;
            return result;
        }
    }

    // Fallback (should never reach here)
    int result = candidates[n_candidates - 1].id;
    delete[] candidates;
    return result;
}

// Autoregressive Token Generation
// Generates tokens autoregressively until EOS token is encountered or max_tokens is reached
// Parameters:
//   - ctx: ggml context for tensor operations
//   - prompt_tokens: initial prompt token IDs [prompt_len]
//   - prompt_len: length of prompt
//   - embed_weight: embedding weights [vocab_size, embedding_dim]
//   - text_proj_*: text projection weights (embedding_dim → hidden_dim)
//   - layer_weights: array of pointers to transformer layer weights
//   - n_layers: number of transformer layers
//   - norm_weight: final normalization weights [hidden_dim]
//   - lm_head_weight: output projection weights [vocab_size, hidden_dim]
//   - max_tokens: maximum number of tokens to generate
//   - temperature: sampling temperature
//   - top_k: top-k sampling parameter
//   - top_p: top-p (nucleus) sampling parameter
//   - eos_token_id: token ID that signals end of sequence
//   - rng_state: random number generator state
//   - output_tokens: buffer to store generated tokens (caller must allocate)
// Returns: number of tokens generated (including prompt)
int generate_tokens(
    struct ggml_context * ctx,
    const int * prompt_tokens,
    int prompt_len,
    struct ggml_tensor * embed_weight,
    struct ggml_tensor * text_proj_fc1_weight,
    struct ggml_tensor * text_proj_fc1_bias,
    struct ggml_tensor * text_proj_fc2_weight,
    struct ggml_tensor * text_proj_fc2_bias,
    struct ggml_tensor ** layer_weights,
    int n_layers,
    struct ggml_tensor * norm_weight,
    struct ggml_tensor * lm_head_weight,
    int max_tokens,
    float temperature,
    int top_k,
    float top_p,
    int eos_token_id,
    uint64_t * rng_state,
    int * output_tokens) {

    // Initialize with prompt tokens
    for (int i = 0; i < prompt_len; i++) {
        output_tokens[i] = prompt_tokens[i];
    }

    int current_len = prompt_len;

    // Autoregressive generation loop
    // Create a fresh context for each forward pass to avoid memory accumulation
    // Memory scales with sequence length: base + seq_len * per_token
    // Base ~200MB for model overhead, ~5MB per token for 28 layers
    size_t mem_base = 256ULL * 1024 * 1024;  // 256MB base
    size_t mem_per_token = 5ULL * 1024 * 1024;  // 5MB per token

    while (current_len < max_tokens) {
        // Calculate memory needed for this sequence length
        size_t mem_needed = mem_base + (size_t)current_len * mem_per_token;

        // Create fresh context for this forward pass
        struct ggml_init_params pass_params = {
            .mem_size   = mem_needed,
            .mem_buffer = nullptr,
            .no_alloc   = false,
        };
        struct ggml_context * pass_ctx = ggml_init(pass_params);
        if (!pass_ctx) {
            fprintf(stderr, "generate_tokens: Failed to create context for forward pass\n");
            break;
        }

        // Create tensor for current sequence
        struct ggml_tensor * token_ids = ggml_new_tensor_1d(pass_ctx, GGML_TYPE_I32, current_len);
        int32_t * token_data = (int32_t *)token_ids->data;
        for (int i = 0; i < current_len; i++) {
            token_data[i] = output_tokens[i];
        }

        // Forward pass through LLM
        struct ggml_tensor * logits = llm_forward(
            pass_ctx,
            token_ids,
            embed_weight,
            text_proj_fc1_weight,
            text_proj_fc1_bias,
            text_proj_fc2_weight,
            text_proj_fc2_bias,
            layer_weights,
            n_layers,
            norm_weight,
            lm_head_weight
        );

        // Build and execute compute graph
        struct ggml_cgraph * graph = ggml_new_graph(pass_ctx);
        ggml_build_forward_expand(graph, logits);
        ggml_graph_compute_with_ctx(pass_ctx, graph, 1);  // Single-threaded for now

        // Get logits for last position
        // logits shape: [vocab_size, seq_len]
        int vocab_size = logits->ne[0];
        int seq_len_out = logits->ne[1];
        float * logits_data = (float *)logits->data;

        // Extract logits for last token position
        float * last_logits = logits_data + vocab_size * (seq_len_out - 1);

        // Sample next token
        int next_token = sample_token(
            last_logits,
            vocab_size,
            temperature,
            top_k,
            top_p,
            rng_state
        );

        // Free this pass's context before continuing
        ggml_free(pass_ctx);

        // Append to sequence
        output_tokens[current_len] = next_token;
        current_len++;

        // Progress indicator
        if (current_len % 10 == 0) {
            printf("  Generated %d tokens...\n", current_len);
            fflush(stdout);
        }

        // Check for EOS
        if (next_token == eos_token_id) {
            break;
        }
    }

    return current_len;
}

// Speaker Embedding Lookup
// CustomVoice model has built-in speaker embeddings (aiden, ryan, serena, vivian, etc.)
// Each speaker has a pre-computed embedding vector stored in the model weights
// This function retrieves the embedding tensor for a given speaker name
//
// Parameters:
//   - ctx: ggml context for tensor allocation
//   - speaker_name: name of the speaker (case-insensitive)
//   - speaker_embeddings: tensor containing all speaker embeddings [n_speakers, embedding_dim]
// Returns: tensor containing the speaker embedding [embedding_dim] or default speaker if unknown
struct ggml_tensor * get_speaker_embedding(
    struct ggml_context * ctx,
    const char * speaker_name,
    struct ggml_tensor * speaker_embeddings) {

    // Mapping of speaker names to indices
    // CustomVoice model built-in speakers (0.6B model)
    struct SpeakerMapping {
        const char * name;
        int index;
    };

    static const SpeakerMapping speaker_map[] = {
        {"aiden", 0},
        {"ryan", 1},
        {"serena", 2},
        {"vivian", 3},
        {"aria", 4},
        {"emma", 5},
        {"sophia", 6},
        {nullptr, 0}  // Default fallback
    };

    // Convert speaker_name to lowercase for case-insensitive comparison
    char speaker_lower[64] = {0};
    int i = 0;
    while (speaker_name[i] && i < 63) {
        speaker_lower[i] = (speaker_name[i] >= 'A' && speaker_name[i] <= 'Z')
                          ? (speaker_name[i] + 32)
                          : speaker_name[i];
        i++;
    }
    speaker_lower[i] = '\0';

    // Look up speaker index
    int speaker_idx = 0;  // Default to first speaker (aiden)
    for (int j = 0; speaker_map[j].name != nullptr; j++) {
        if (strcmp(speaker_lower, speaker_map[j].name) == 0) {
            speaker_idx = speaker_map[j].index;
            break;
        }
    }

    // Extract speaker embedding from the embeddings tensor
    // speaker_embeddings shape: [embedding_dim, n_speakers]
    // We need to extract column speaker_idx
    int embedding_dim = speaker_embeddings->ne[0];
    int n_speakers = speaker_embeddings->ne[1];

    // Clamp speaker_idx to valid range
    if (speaker_idx >= n_speakers) {
        speaker_idx = 0;  // Fallback to default
    }

    // Create a view into the speaker embeddings tensor for the selected speaker
    struct ggml_tensor * speaker_emb = ggml_view_1d(
        ctx,
        speaker_embeddings,
        embedding_dim,
        speaker_idx * embedding_dim * ggml_element_size(speaker_embeddings)
    );

    return speaker_emb;
}

// Forward declaration of cached transformer block
struct ggml_tensor * transformer_block_cached(
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
    struct ggml_tensor * ffn_w3,
    int layer_idx,
    KVCache * kv_cache,
    int start_pos);

// Talker forward pass with KV caching
// For prefill: process all tokens, build cache
// For decode: process single token using cache
static struct ggml_tensor * talker_forward_cached(
    struct ggml_context * ctx,
    struct ggml_tensor * token_ids,
    struct ggml_tensor * embed_weight,
    struct ggml_tensor * text_proj_fc1_weight,
    struct ggml_tensor * text_proj_fc1_bias,
    struct ggml_tensor * text_proj_fc2_weight,
    struct ggml_tensor * text_proj_fc2_bias,
    struct ggml_tensor ** layer_weights,
    int n_layers,
    struct ggml_tensor * norm_weight,
    struct ggml_tensor * lm_head_weight,
    KVCache * kv_cache,
    int start_pos) {

    // Token Embedding
    struct ggml_tensor * embedded = ggml_get_rows(ctx, embed_weight, token_ids);

    // Text Projection (embedding_dim=2048 -> hidden_dim=1024)
    struct ggml_tensor * proj = ggml_mul_mat(ctx, text_proj_fc1_weight, embedded);
    struct ggml_tensor * fc1_bias = text_proj_fc1_bias;
    if (text_proj_fc1_bias->type == GGML_TYPE_F16) {
        fc1_bias = ggml_cast(ctx, text_proj_fc1_bias, GGML_TYPE_F32);
    }
    proj = ggml_add(ctx, proj, fc1_bias);
    proj = ggml_silu(ctx, proj);
    proj = ggml_mul_mat(ctx, text_proj_fc2_weight, proj);
    struct ggml_tensor * fc2_bias = text_proj_fc2_bias;
    if (text_proj_fc2_bias->type == GGML_TYPE_F16) {
        fc2_bias = ggml_cast(ctx, text_proj_fc2_bias, GGML_TYPE_F32);
    }
    proj = ggml_add(ctx, proj, fc2_bias);

    // Pass through transformer blocks with caching
    struct ggml_tensor * hidden = proj;
    for (int i = 0; i < n_layers; i++) {
        struct ggml_tensor ** layer_w = &layer_weights[i * 9];

        hidden = transformer_block_cached(
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
            layer_w[8],  // ffn_w3
            i,           // layer_idx
            kv_cache,
            start_pos
        );
    }

    // Final RMSNorm
    struct ggml_tensor * normalized = ops::rms_norm(ctx, hidden, norm_weight, 1e-6f);

    // Output projection
    struct ggml_tensor * logits = ggml_mul_mat(ctx, lm_head_weight, normalized);

    return logits;
}

// Generate tokens with KV caching - much faster!
// Uses KV cache to avoid recomputing attention for previous tokens
int generate_tokens_cached(
    const int * prompt_tokens,
    int prompt_len,
    struct ggml_tensor * embed_weight,
    struct ggml_tensor * text_proj_fc1_weight,
    struct ggml_tensor * text_proj_fc1_bias,
    struct ggml_tensor * text_proj_fc2_weight,
    struct ggml_tensor * text_proj_fc2_bias,
    struct ggml_tensor ** layer_weights,
    int n_layers,
    struct ggml_tensor * norm_weight,
    struct ggml_tensor * lm_head_weight,
    int max_tokens,
    float temperature,
    int top_k,
    float top_p,
    int eos_token_id,
    uint64_t * rng_state,
    int * output_tokens) {

    // Configuration
    const int num_kv_heads = 8;
    const int head_dim = 128;
    const int max_seq_len = max_tokens + 128;  // Allow some extra room

    // Create KV cache
    KVCache * kv_cache = KVCache::create(n_layers, num_kv_heads, head_dim, max_seq_len);
    if (!kv_cache) {
        fprintf(stderr, "generate_tokens_cached: Failed to create KV cache\n");
        return 0;
    }

    // Copy prompt tokens to output
    for (int i = 0; i < prompt_len; i++) {
        output_tokens[i] = prompt_tokens[i];
    }

    int current_len = prompt_len;

    // Memory for compute contexts
    // Prefill needs more memory than decode
    size_t prefill_mem = 512ULL * 1024 * 1024;  // 512MB for prefill
    size_t decode_mem = 256ULL * 1024 * 1024;   // 256MB for decode (single token)

    // Phase 1: Prefill - process all prompt tokens
    printf("  Prefilling %d prompt tokens...\n", prompt_len);
    fflush(stdout);
    {
        struct ggml_init_params params = {
            .mem_size = prefill_mem,
            .mem_buffer = nullptr,
            .no_alloc = false,
        };
        struct ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            fprintf(stderr, "generate_tokens_cached: Failed to create prefill context\n");
            KVCache::destroy(kv_cache);
            return 0;
        }

        // Create token tensor
        struct ggml_tensor * token_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, prompt_len);
        memcpy(token_ids->data, prompt_tokens, prompt_len * sizeof(int32_t));

        // Forward pass
        struct ggml_tensor * logits = talker_forward_cached(
            ctx, token_ids, embed_weight,
            text_proj_fc1_weight, text_proj_fc1_bias,
            text_proj_fc2_weight, text_proj_fc2_bias,
            layer_weights, n_layers, norm_weight, lm_head_weight,
            kv_cache, 0  // start_pos = 0 for prefill
        );

        // Build and execute graph
        struct ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, logits);
        ggml_graph_compute_with_ctx(ctx, graph, 4);  // Use 4 threads

        // Sample first token
        int vocab_size = logits->ne[0];
        float * logits_data = (float *)logits->data;
        float * last_logits = logits_data + vocab_size * (prompt_len - 1);

        int next_token = sample_token(last_logits, vocab_size, temperature, top_k, top_p, rng_state);
        output_tokens[current_len++] = next_token;

        ggml_free(ctx);

        if (next_token == eos_token_id) {
            KVCache::destroy(kv_cache);
            return current_len;
        }
    }

    // Phase 2: Decode - generate tokens one at a time
    printf("  Decoding tokens...\n");
    fflush(stdout);
    while (current_len < max_tokens) {
        struct ggml_init_params params = {
            .mem_size = decode_mem,
            .mem_buffer = nullptr,
            .no_alloc = false,
        };
        struct ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            fprintf(stderr, "generate_tokens_cached: Failed to create decode context\n");
            break;
        }

        // Create tensor for single token
        struct ggml_tensor * token_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ((int32_t *)token_ids->data)[0] = output_tokens[current_len - 1];

        // Forward pass for single token
        struct ggml_tensor * logits = talker_forward_cached(
            ctx, token_ids, embed_weight,
            text_proj_fc1_weight, text_proj_fc1_bias,
            text_proj_fc2_weight, text_proj_fc2_bias,
            layer_weights, n_layers, norm_weight, lm_head_weight,
            kv_cache, current_len - 1  // start_pos = position of new token
        );

        // Build and execute graph
        struct ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, logits);
        ggml_graph_compute_with_ctx(ctx, graph, 4);  // Use 4 threads

        // Sample next token
        int vocab_size = logits->ne[0];
        float * logits_data = (float *)logits->data;

        int next_token = sample_token(logits_data, vocab_size, temperature, top_k, top_p, rng_state);

        ggml_free(ctx);

        output_tokens[current_len++] = next_token;

        // Progress indicator
        if (current_len % 50 == 0) {
            printf("  Generated %d tokens...\n", current_len);
            fflush(stdout);
        }

        // Check for EOS
        if (next_token == eos_token_id) {
            break;
        }
    }

    KVCache::destroy(kv_cache);
    printf("  Generated %d tokens total\n", current_len);
    return current_len;
}

} // namespace model
} // namespace leaxer_qwen
