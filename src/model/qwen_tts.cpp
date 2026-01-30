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
    struct ggml_tensor * q_norm_weight,
    struct ggml_tensor * k_norm_weight,
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
// Codec special tokens (from GGUF metadata - CustomVoice model)
// Note: These values differ from the Python config defaults (4196-4198)
// The actual values come from qwen3.tts.codec_* GGUF metadata fields
constexpr int CODEC_PAD_ID = 2148;
constexpr int CODEC_BOS_ID = 2149;
constexpr int CODEC_EOS_ID = 2150;

// Model dimensions (0.6B model)
constexpr int TEXT_EMBED_DIM = 2048;
constexpr int HIDDEN_DIM = 1024;
constexpr int CODEC_VOCAB_SIZE = 3072;

// ============================================================================
// Text Projection MLP Helper
// Architecture: Linear(2048→2048) → SiLU → Linear(2048→1024)
// Projects text embeddings (2048-dim) to talker hidden dimension (1024-dim)
// ============================================================================
struct ggml_tensor * text_projection(
    struct ggml_context * ctx,
    struct ggml_tensor * input,           // [seq_len, 2048] or [2048, seq_len] text embeddings
    struct ggml_tensor * fc1_weight,      // [2048, 2048]
    struct ggml_tensor * fc1_bias,        // [2048]
    struct ggml_tensor * fc2_weight,      // [1024, 2048]
    struct ggml_tensor * fc2_bias)        // [1024]
{
    // fc1: Linear(2048 → 2048)
    struct ggml_tensor * proj = ggml_mul_mat(ctx, fc1_weight, input);
    
    // Add bias (cast to F32 if needed)
    struct ggml_tensor * bias1 = fc1_bias;
    if (fc1_bias->type == GGML_TYPE_F16) {
        bias1 = ggml_cast(ctx, fc1_bias, GGML_TYPE_F32);
    }
    proj = ggml_add(ctx, proj, bias1);
    
    // SiLU activation
    proj = ggml_silu(ctx, proj);
    
    // fc2: Linear(2048 → 1024)
    proj = ggml_mul_mat(ctx, fc2_weight, proj);
    
    // Add bias
    struct ggml_tensor * bias2 = fc2_bias;
    if (fc2_bias->type == GGML_TYPE_F16) {
        bias2 = ggml_cast(ctx, fc2_bias, GGML_TYPE_F32);
    }
    proj = ggml_add(ctx, proj, bias2);
    
    return proj;  // [seq_len, 1024] or [1024, seq_len]
}

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
        // Each layer expects 11 weight tensors in order:
        // 0: attn_norm_weight
        // 1: q_weight
        // 2: k_weight
        // 3: v_weight
        // 4: o_weight
        // 5: q_norm_weight (Qwen3)
        // 6: k_norm_weight (Qwen3)
        // 7: ffn_norm_weight
        // 8: ffn_gate (w1)
        // 9: ffn_up (w2)
        // 10: ffn_down (w3)
        struct ggml_tensor ** layer_w = &layer_weights[i * 11];

        hidden = transformer_block(
            ctx,
            hidden,
            layer_w[0],  // attn_norm_weight
            layer_w[1],  // q_weight
            layer_w[2],  // k_weight
            layer_w[3],  // v_weight
            layer_w[4],  // o_weight
            layer_w[5],  // q_norm_weight (Qwen3)
            layer_w[6],  // k_norm_weight (Qwen3)
            layer_w[7],  // ffn_norm_weight
            layer_w[8],  // ffn_w1
            layer_w[9],  // ffn_w2
            layer_w[10]  // ffn_w3
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

// Codec special token constants (for suppression during sampling)
// Audio codes are 0-2047, special tokens are 2148-2150
constexpr int CODEC_AUDIO_MAX = 2047;    // Max valid audio code
constexpr int CODEC_SUPPRESS_START = 2048;  // Start of suppression range
// EOS (2150) is allowed, all other special tokens are suppressed

// Token Sampling
// Implements temperature, top-k, and top-p (nucleus) sampling from logits
// Parameters:
//   - logits: [vocab_size] raw logits from model
//   - vocab_size: size of vocabulary
//   - temperature: sampling temperature (>1 = more random, <1 = more deterministic, 0 = greedy)
//   - top_k: keep only top-k highest probability tokens (0 = disabled)
//   - top_p: keep tokens with cumulative probability >= top_p (1.0 = disabled)
//   - rng_state: random number generator state (simple LCG)
//   - eos_token_id: EOS token to allow (default -1 = no special handling)
// Returns: sampled token ID
int sample_token(
    const float * logits,
    int vocab_size,
    float temperature,
    int top_k,
    float top_p,
    uint64_t * rng_state,
    int eos_token_id = -1) {

    // Create modified logits with suppressed tokens
    float * mod_logits = new float[vocab_size];
    memcpy(mod_logits, logits, vocab_size * sizeof(float));

    // Suppress tokens in range [2048, vocab_size) except EOS
    // This follows Python behavior: suppress_tokens = range(vocab_size-1024, vocab_size) except EOS
    for (int i = CODEC_SUPPRESS_START; i < vocab_size; i++) {
        if (i != eos_token_id) {
            mod_logits[i] = -1e10f;  // Very negative = effectively zero probability
        }
    }

    // Greedy sampling (argmax)
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

    // Allocate space for token index/probability pairs
    struct TokenProb {
        int id;
        float prob;
    };
    TokenProb * candidates = new TokenProb[vocab_size];

    // Apply temperature and softmax
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

        // Sample next token (suppress special tokens except EOS)
        int next_token = sample_token(
            last_logits,
            vocab_size,
            temperature,
            top_k,
            top_p,
            rng_state,
            eos_token_id
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
    struct ggml_tensor * q_norm_weight,      // Q normalization (Qwen3)
    struct ggml_tensor * k_norm_weight,      // K normalization (Qwen3)
    struct ggml_tensor * ffn_norm_weight,
    struct ggml_tensor * ffn_w1,
    struct ggml_tensor * ffn_w2,
    struct ggml_tensor * ffn_w3,
    int layer_idx,
    KVCache * kv_cache,
    int start_pos);

// ============================================================================
// Talker Forward with Combined Embeddings (CORRECT IMPLEMENTATION)
// ============================================================================
// CRITICAL: Input to talker is SUMMED embeddings, not concatenated!
//   combined_input = text_projection(text_embed) + codec_embed
//
// This function takes pre-computed combined embeddings and runs through
// the transformer layers. Use this during generation when embeddings
// are already combined (text + codec or summed codebook embeddings).
//
// Parameters:
//   combined_embeds: [hidden_dim, seq_len] pre-combined embeddings (1024-dim)
//   layer_weights: transformer layer weights
//   n_layers: number of transformer layers
//   norm_weight: final RMSNorm weight
//   lm_head_weight: output projection [codec_vocab, hidden_dim]
//   kv_cache: KV cache for attention
//   start_pos: starting position for KV cache
// Returns: logits [codec_vocab, seq_len]
// ============================================================================
static struct ggml_tensor * talker_forward_from_embeds(
    struct ggml_context * ctx,
    struct ggml_tensor * combined_embeds,  // [hidden_dim=1024, seq_len] already combined
    struct ggml_tensor ** layer_weights,
    int n_layers,
    struct ggml_tensor * norm_weight,
    struct ggml_tensor * lm_head_weight,
    KVCache * kv_cache,
    int start_pos) {

    // Pass through transformer blocks with caching
    struct ggml_tensor * hidden = combined_embeds;
    for (int i = 0; i < n_layers; i++) {
        struct ggml_tensor ** layer_w = &layer_weights[i * 11];

        hidden = transformer_block_cached(
            ctx,
            hidden,
            layer_w[0],  // attn_norm_weight
            layer_w[1],  // q_weight
            layer_w[2],  // k_weight
            layer_w[3],  // v_weight
            layer_w[4],  // o_weight
            layer_w[5],  // q_norm_weight (Qwen3)
            layer_w[6],  // k_norm_weight (Qwen3)
            layer_w[7],  // ffn_norm_weight
            layer_w[8],  // ffn_w1
            layer_w[9],  // ffn_w2
            layer_w[10], // ffn_w3
            i,           // layer_idx
            kv_cache,
            start_pos
        );
    }

    // Final RMSNorm
    struct ggml_tensor * normalized = ops::rms_norm(ctx, hidden, norm_weight, 1e-6f);

    // Output projection to codec vocabulary (3072)
    struct ggml_tensor * logits = ggml_mul_mat(ctx, lm_head_weight, normalized);

    return logits;
}

// ============================================================================
// Talker Forward with Hidden State Output
// ============================================================================
// Same as talker_forward_from_embeds but ALSO outputs the hidden state
// (normalized hidden state BEFORE lm_head projection).
// This hidden state is needed by the code predictor.
//
// Parameters:
//   hidden_state_out: If not NULL, receives the normalized hidden state [hidden_dim, seq_len]
//                     This is the hidden state AFTER RMS norm, BEFORE lm_head
// Returns: logits [codec_vocab, seq_len]
// ============================================================================
static struct ggml_tensor * talker_forward_from_embeds_with_hidden(
    struct ggml_context * ctx,
    struct ggml_tensor * combined_embeds,  // [hidden_dim=1024, seq_len] already combined
    struct ggml_tensor ** layer_weights,
    int n_layers,
    struct ggml_tensor * norm_weight,
    struct ggml_tensor * lm_head_weight,
    KVCache * kv_cache,
    int start_pos,
    struct ggml_tensor ** hidden_state_out)  // OUTPUT: normalized hidden state
{
    // Pass through transformer blocks with caching
    struct ggml_tensor * hidden = combined_embeds;
    for (int i = 0; i < n_layers; i++) {
        struct ggml_tensor ** layer_w = &layer_weights[i * 11];

        hidden = transformer_block_cached(
            ctx,
            hidden,
            layer_w[0],  // attn_norm_weight
            layer_w[1],  // q_weight
            layer_w[2],  // k_weight
            layer_w[3],  // v_weight
            layer_w[4],  // o_weight
            layer_w[5],  // q_norm_weight (Qwen3)
            layer_w[6],  // k_norm_weight (Qwen3)
            layer_w[7],  // ffn_norm_weight
            layer_w[8],  // ffn_w1
            layer_w[9],  // ffn_w2
            layer_w[10], // ffn_w3
            i,           // layer_idx
            kv_cache,
            start_pos
        );
    }

    // Final RMSNorm - this is the hidden state needed by code predictor
    struct ggml_tensor * normalized = ops::rms_norm(ctx, hidden, norm_weight, 1e-6f);

    // Output the hidden state if requested
    if (hidden_state_out) {
        *hidden_state_out = normalized;
    }

    // Output projection to codec vocabulary (3072)
    struct ggml_tensor * logits = ggml_mul_mat(ctx, lm_head_weight, normalized);

    return logits;
}

// ============================================================================
// Talker Forward with Proper Embedding Combination
// ============================================================================
// CORRECT FLOW (from Python reference):
// 1. Text tokens → text_embedding(151936 vocab, 2048 dim) → text_embed
// 2. text_embed → text_projection MLP → projected_text (1024 dim)
// 3. Codec tokens → codec_embedding(3072 vocab, 1024 dim) → codec_embed
// 4. combined = projected_text + codec_embed (element-wise SUM!)
// 5. combined → transformer layers → logits
//
// During prefill: codec_ids are CODEC_PAD_ID for all text positions
// During decode: input is sum of all 16 codebook embeddings + trailing text
// ============================================================================
static struct ggml_tensor * talker_forward_with_codec(
    struct ggml_context * ctx,
    struct ggml_tensor * text_token_ids,      // [seq_len] text tokens (151936 vocab)
    struct ggml_tensor * codec_token_ids,     // [seq_len] codec tokens (3072 vocab) - use CODEC_PAD_ID for prefill
    struct ggml_tensor * text_embed_weight,   // [text_vocab, 2048]
    struct ggml_tensor * codec_embed_weight,  // [codec_vocab=3072, 1024]
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

    // Step 1: Text Embedding (2048-dim)
    struct ggml_tensor * text_embedded = ggml_get_rows(ctx, text_embed_weight, text_token_ids);

    // Step 2: Text Projection (2048 → 1024)
    struct ggml_tensor * text_proj = text_projection(
        ctx, text_embedded,
        text_proj_fc1_weight, text_proj_fc1_bias,
        text_proj_fc2_weight, text_proj_fc2_bias
    );

    // Step 3: Codec Embedding (1024-dim)
    struct ggml_tensor * codec_embedded = ggml_get_rows(ctx, codec_embed_weight, codec_token_ids);

    // Step 4: CRITICAL - Element-wise SUM (not concatenation!)
    struct ggml_tensor * combined = ggml_add(ctx, text_proj, codec_embedded);

    // Step 5: Forward through transformer
    return talker_forward_from_embeds(
        ctx, combined, layer_weights, n_layers,
        norm_weight, lm_head_weight, kv_cache, start_pos
    );
}

// Talker forward pass with KV caching (LEGACY - for backward compatibility)
// NOTE: This version doesn't add codec embeddings - use talker_forward_with_codec instead
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

    // Text Projection using helper function
    struct ggml_tensor * proj = text_projection(
        ctx, embedded,
        text_proj_fc1_weight, text_proj_fc1_bias,
        text_proj_fc2_weight, text_proj_fc2_bias
    );

    // Pass through transformer blocks with caching
    struct ggml_tensor * hidden = proj;
    for (int i = 0; i < n_layers; i++) {
        struct ggml_tensor ** layer_w = &layer_weights[i * 11];

        hidden = transformer_block_cached(
            ctx,
            hidden,
            layer_w[0],  // attn_norm_weight
            layer_w[1],  // q_weight
            layer_w[2],  // k_weight
            layer_w[3],  // v_weight
            layer_w[4],  // o_weight
            layer_w[5],  // q_norm_weight (Qwen3)
            layer_w[6],  // k_norm_weight (Qwen3)
            layer_w[7],  // ffn_norm_weight
            layer_w[8],  // ffn_w1
            layer_w[9],  // ffn_w2
            layer_w[10], // ffn_w3
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

        int next_token = sample_token(last_logits, vocab_size, temperature, top_k, top_p, rng_state, eos_token_id);
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

        int next_token = sample_token(logits_data, vocab_size, temperature, top_k, top_p, rng_state, eos_token_id);

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

// ============================================================================
// Generate Tokens with Proper Embedding Combination (CORRECT IMPLEMENTATION)
// ============================================================================
// This is the CORRECT generation function that properly combines embeddings
// AND captures hidden states for the code predictor.
//
// ARCHITECTURE (from Python reference):
// 1. PREFILL: For each text token position:
//    input = text_projection(text_embed) + codec_embedding(CODEC_PAD_ID)
//
// 2. DECODE: For each generation step:
//    - Sample codebook_0 token from logits
//    - Input = codec_embedding(codebook_0_token)
//    - CAPTURE hidden state (normalized, before lm_head) for code_predictor
//
// Parameters:
//   prompt_tokens: text token IDs for prefill
//   codec_embed_weight: talker's codec embedding [3072, 1024]
//   hidden_states_out: OUTPUT buffer for hidden states [max_codec_tokens, 1024]
//                      If not NULL, receives hidden state for each generated codec token
//                      These are needed by code_predictor_forward
//   n_hidden_out: OUTPUT number of hidden states written
//   (other params same as generate_tokens_cached)
// ============================================================================
int generate_tokens_with_codec(
    const int * prompt_tokens,
    int prompt_len,
    struct ggml_tensor * text_embed_weight,    // [text_vocab=151936, 2048]
    struct ggml_tensor * codec_embed_weight,   // [codec_vocab=3072, 1024] - CRITICAL!
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
    int * output_tokens,
    float * hidden_states_out,  // OUTPUT: [max_codec_tokens, 1024] hidden states for code_predictor
    int * n_hidden_out)         // OUTPUT: number of hidden states written
{
    // Configuration
    const int num_kv_heads = 8;
    const int head_dim = 128;
    const int max_seq_len = max_tokens + 128;
    
    // Initialize hidden state count
    int hidden_idx = 0;
    if (n_hidden_out) *n_hidden_out = 0;

    // Create KV cache
    KVCache * kv_cache = KVCache::create(n_layers, num_kv_heads, head_dim, max_seq_len);
    if (!kv_cache) {
        fprintf(stderr, "generate_tokens_with_codec: Failed to create KV cache\n");
        return 0;
    }

    // Copy prompt tokens to output
    for (int i = 0; i < prompt_len; i++) {
        output_tokens[i] = prompt_tokens[i];
    }

    int current_len = prompt_len;

    // Memory for compute contexts
    size_t prefill_mem = 512ULL * 1024 * 1024;
    size_t decode_mem = 256ULL * 1024 * 1024;

    // ========================================================================
    // Phase 1: PREFILL with proper embedding combination
    // ========================================================================
    // For prefill, each text token position combines:
    //   text_projection(text_embed) + codec_embedding(CODEC_PAD_ID)
    // We capture the hidden state at the last position for the first codec token
    // ========================================================================
    printf("  Prefilling %d prompt tokens (with codec embedding combination)...\n", prompt_len);
    fflush(stdout);
    {
        struct ggml_init_params params = {
            .mem_size = prefill_mem,
            .mem_buffer = nullptr,
            .no_alloc = false,
        };
        struct ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            fprintf(stderr, "generate_tokens_with_codec: Failed to create prefill context\n");
            KVCache::destroy(kv_cache);
            return 0;
        }

        // Create text token tensor
        struct ggml_tensor * text_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, prompt_len);
        memcpy(text_ids->data, prompt_tokens, prompt_len * sizeof(int32_t));

        // Create codec token tensor (all CODEC_PAD_ID for prefill)
        struct ggml_tensor * codec_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, prompt_len);
        int32_t * codec_data = (int32_t *)codec_ids->data;
        for (int i = 0; i < prompt_len; i++) {
            codec_data[i] = CODEC_PAD_ID;  // Use pad token for all text positions
        }

        // Forward pass with proper embedding combination
        struct ggml_tensor * logits = talker_forward_with_codec(
            ctx,
            text_ids,
            codec_ids,
            text_embed_weight,
            codec_embed_weight,
            text_proj_fc1_weight, text_proj_fc1_bias,
            text_proj_fc2_weight, text_proj_fc2_bias,
            layer_weights, n_layers, norm_weight, lm_head_weight,
            kv_cache, 0  // start_pos = 0 for prefill
        );

        // Build and execute graph
        struct ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, logits);
        ggml_graph_compute_with_ctx(ctx, graph, 4);

        // Sample first codec token
        int vocab_size = logits->ne[0];
        float * logits_data = (float *)logits->data;
        float * last_logits = logits_data + vocab_size * (prompt_len - 1);

        int next_token = sample_token(last_logits, vocab_size, temperature, top_k, top_p, rng_state, eos_token_id);
        output_tokens[current_len++] = next_token;

        ggml_free(ctx);

        if (next_token == eos_token_id) {
            KVCache::destroy(kv_cache);
            printf("  Generated EOS at prefill\n");
            if (n_hidden_out) *n_hidden_out = hidden_idx;
            return current_len;
        }
    }

    // ========================================================================
    // Phase 2: DECODE with codec embeddings + hidden state capture
    // ========================================================================
    // For decode, the input is the codec embedding of the previously generated
    // codec token. We capture the hidden state for each generated token to
    // pass to the code predictor.
    // ========================================================================
    printf("  Decoding tokens (codec domain) with hidden state capture...\n");
    fflush(stdout);
    while (current_len < max_tokens) {
        struct ggml_init_params params = {
            .mem_size = decode_mem,
            .mem_buffer = nullptr,
            .no_alloc = false,
        };
        struct ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            fprintf(stderr, "generate_tokens_with_codec: Failed to create decode context\n");
            break;
        }

        // Get the last generated codec token
        int last_codec_token = output_tokens[current_len - 1];

        // Clamp to valid codec range (0-3071)
        if (last_codec_token < 0) last_codec_token = 0;
        if (last_codec_token >= CODEC_VOCAB_SIZE) last_codec_token = CODEC_VOCAB_SIZE - 1;

        // Create single codec embedding for the last token
        struct ggml_tensor * codec_id = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ((int32_t *)codec_id->data)[0] = last_codec_token;

        // Get codec embedding
        struct ggml_tensor * codec_embed = ggml_get_rows(ctx, codec_embed_weight, codec_id);

        // Forward pass from embeddings with hidden state output
        struct ggml_tensor * hidden_state = nullptr;
        struct ggml_tensor * logits = talker_forward_from_embeds_with_hidden(
            ctx, codec_embed, layer_weights, n_layers,
            norm_weight, lm_head_weight, kv_cache, current_len - 1,
            &hidden_state  // OUTPUT: capture hidden state
        );

        // Build and execute graph
        struct ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, logits);
        // Also ensure hidden_state is computed
        if (hidden_state) {
            ggml_build_forward_expand(graph, hidden_state);
        }
        ggml_graph_compute_with_ctx(ctx, graph, 4);

        // Capture hidden state for code predictor
        if (hidden_states_out && hidden_state) {
            // hidden_state shape: [hidden_dim, 1] = [1024, 1]
            // Copy to output buffer at position hidden_idx
            const float * hs_data = (const float *)hidden_state->data;
            memcpy(&hidden_states_out[hidden_idx * HIDDEN_DIM], hs_data, HIDDEN_DIM * sizeof(float));
            hidden_idx++;
        }

        // Sample next token
        int vocab_size = logits->ne[0];
        float * logits_data = (float *)logits->data;

        int next_token = sample_token(logits_data, vocab_size, temperature, top_k, top_p, rng_state, eos_token_id);

        ggml_free(ctx);

        output_tokens[current_len++] = next_token;

        // Progress indicator
        if (current_len % 50 == 0) {
            printf("  Generated %d codec tokens (captured %d hidden states)...\n", 
                   current_len - prompt_len, hidden_idx);
            fflush(stdout);
        }

        // Check for EOS
        if (next_token == eos_token_id) {
            printf("  EOS reached at position %d\n", current_len);
            break;
        }
    }

    KVCache::destroy(kv_cache);
    
    // Output number of hidden states captured
    if (n_hidden_out) *n_hidden_out = hidden_idx;
    
    printf("  Generated %d tokens total (%d codec tokens, %d hidden states captured)\n", 
           current_len, current_len - prompt_len, hidden_idx);
    return current_len;
}

// Legacy version without hidden state capture (for backward compatibility)
int generate_tokens_with_codec(
    const int * prompt_tokens,
    int prompt_len,
    struct ggml_tensor * text_embed_weight,
    struct ggml_tensor * codec_embed_weight,
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
    
    // Call the full version without hidden state capture
    return generate_tokens_with_codec(
        prompt_tokens, prompt_len,
        text_embed_weight, codec_embed_weight,
        text_proj_fc1_weight, text_proj_fc1_bias,
        text_proj_fc2_weight, text_proj_fc2_bias,
        layer_weights, n_layers, norm_weight, lm_head_weight,
        max_tokens, temperature, top_k, top_p, eos_token_id, rng_state,
        output_tokens,
        nullptr,  // No hidden state capture
        nullptr
    );
}

} // namespace model
} // namespace leaxer_qwen
