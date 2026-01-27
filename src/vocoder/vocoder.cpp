// Full Vocoder: Qwen3TTSTokenizerV2Decoder
// Converts discrete codec tokens to audio waveform

#include "ggml.h"
#include "ggml-cpu.h"
#include "common.h"
#include <cstring>
#include <cstdint>

namespace leaxer_qwen {
namespace vocoder {

// Forward declarations from other vocoder components
void rvq_decode(
    struct ggml_tensor * dst,
    const struct ggml_tensor * codes,
    const struct ggml_tensor * codebooks);

void causal_conv_project(
    struct ggml_tensor * dst,
    const struct ggml_tensor * latent,
    const struct ggml_tensor * weight,
    const struct ggml_tensor * bias);

void upsample_stage(
    struct ggml_tensor * dst,
    const struct ggml_tensor * input,
    const struct ggml_tensor * weight,
    const struct ggml_tensor * alpha_logscale,
    const struct ggml_tensor * beta_logscale,
    int kernel_size,
    int stride,
    int padding);

} // namespace vocoder
} // namespace leaxer_qwen

// Forward declarations from model components
namespace leaxer_qwen {
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
} // namespace model
} // namespace leaxer_qwen

namespace leaxer_qwen {
namespace vocoder {

// Full pipeline:
// 1. Split RVQ reconstruction
// 2. Causal ConvNet projection
// 3. Transformer decoder with self-attention
// 4. 4-stage progressive upsampling
// 5. Final conv → waveform (output from upsampling)

// Output: 24kHz audio

// Upsample configuration from upsample.cpp
constexpr int UPSAMPLE_RATES[] = {8, 5, 4, 3};
constexpr int NUM_UPSAMPLE_STAGES = 4;

// Vocoder transformer decoder
// Simple self-attention transformer to refine features before upsampling
// Input: x [hidden_dim, seq_len, batch] - projected features from causal conv
// Weights: transformer layer weights (attention + FFN)
// Output: [hidden_dim, seq_len, batch] - refined features
// Note: This is a simplified single-layer decoder. Multi-layer would require weight arrays.
struct ggml_tensor * vocoder_transformer_decode(
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
    struct ggml_tensor * ffn_w3
) {
    // Apply single transformer block for feature refinement
    // In a full implementation, this would loop over multiple layers
    return leaxer_qwen::model::transformer_block(
        ctx, x,
        attn_norm_weight,
        q_weight, k_weight, v_weight, o_weight,
        ffn_norm_weight,
        ffn_w1, ffn_w2, ffn_w3
    );
}

// Vocoder decode with optional transformer refinement
// Input: codes [16, seq_len] - RVQ codes
// Input: codebooks [16, 2048, 512] - codebook embeddings
// Input: upsample_weights [4][kernel_size, in_channels, out_channels] - weights for each upsample stage
// Input: upsample_alphas [4][out_channels] - SnakeBeta alpha parameters
// Input: upsample_betas [4][out_channels] - SnakeBeta beta parameters
// Input: transformer_weights (optional, can be nullptr) - transformer decoder weights
// Output: audio samples [seq_len * 480] (24kHz)
//
// Pipeline: RVQ decode → (optional causal conv) → (optional transformer) → upsample → audio
struct VocoderTransformerWeights {
    struct ggml_tensor * attn_norm_weight;
    struct ggml_tensor * q_weight;
    struct ggml_tensor * k_weight;
    struct ggml_tensor * v_weight;
    struct ggml_tensor * o_weight;
    struct ggml_tensor * ffn_norm_weight;
    struct ggml_tensor * ffn_w1;
    struct ggml_tensor * ffn_w2;
    struct ggml_tensor * ffn_w3;
};

void vocoder_decode(
    struct ggml_tensor * dst,
    const struct ggml_tensor * codes,
    const struct ggml_tensor * codebooks,
    const struct ggml_tensor ** upsample_weights,
    const struct ggml_tensor ** upsample_alphas,
    const struct ggml_tensor ** upsample_betas,
    int * kernel_sizes,
    int * paddings,
    const VocoderTransformerWeights * transformer_weights = nullptr
) {
    GGML_ASSERT(codes->type == GGML_TYPE_I32);
    GGML_ASSERT(codebooks->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t seq_len = codes->ne[0];
    const int64_t num_codebooks = codes->ne[1];
    const int64_t codebook_dim = codebooks->ne[0];

    // Create temporary context for intermediate tensors
    size_t mem_size = 500 * 1024 * 1024;  // 500MB
    struct ggml_context * temp_ctx = create_ggml_context(mem_size);
    GGML_ASSERT(temp_ctx != nullptr);

    // Step 1: RVQ decode
    // Output: [codebook_dim, seq_len]
    struct ggml_tensor * latent = ggml_new_tensor_2d(temp_ctx, GGML_TYPE_F32, codebook_dim, seq_len);
    rvq_decode(latent, codes, codebooks);

    // Step 2 & 3: Optional transformer decoder for feature refinement
    struct ggml_tensor * features = latent;
    if (transformer_weights != nullptr) {
        // Reshape to [hidden_dim, seq_len, 1] for transformer
        // In full implementation, would apply causal conv first to project dimensions
        struct ggml_tensor * x = ggml_new_tensor_3d(temp_ctx, GGML_TYPE_F32, codebook_dim, seq_len, 1);
        memcpy(x->data, latent->data, ggml_nbytes(latent));

        // Apply transformer decoder
        struct ggml_tensor * refined = vocoder_transformer_decode(
            temp_ctx, x,
            transformer_weights->attn_norm_weight,
            transformer_weights->q_weight,
            transformer_weights->k_weight,
            transformer_weights->v_weight,
            transformer_weights->o_weight,
            transformer_weights->ffn_norm_weight,
            transformer_weights->ffn_w1,
            transformer_weights->ffn_w2,
            transformer_weights->ffn_w3
        );

        // Build computation graph and execute
        struct ggml_cgraph * graph = ggml_new_graph(temp_ctx);
        ggml_build_forward_expand(graph, refined);
        ggml_graph_compute_with_ctx(temp_ctx, graph, 1);

        features = refined;
    }

    // Reshape features to [seq_len, codebook_dim, 1] for upsampling
    // ggml upsample expects [seq_len, channels, batch]
    struct ggml_tensor * current = ggml_new_tensor_3d(temp_ctx, GGML_TYPE_F32, seq_len, codebook_dim, 1);

    // Copy and reshape
    const float * features_data = (const float *)features->data;
    float * current_data = (float *)current->data;

    // features: [codebook_dim, seq_len] or [codebook_dim, seq_len, 1]
    // current: [seq_len, codebook_dim, 1]
    for (int64_t t = 0; t < seq_len; t++) {
        for (int64_t c = 0; c < codebook_dim; c++) {
            current_data[t * codebook_dim + c] = features_data[c * seq_len + t];
        }
    }

    // Step 3: Apply 4 upsample stages
    for (int stage = 0; stage < NUM_UPSAMPLE_STAGES; stage++) {
        int stride = UPSAMPLE_RATES[stage];
        int64_t seq_len_out = current->ne[0] * stride;
        int64_t out_channels = upsample_weights[stage]->ne[2];

        // Create output tensor for this stage
        struct ggml_tensor * stage_out = ggml_new_tensor_3d(temp_ctx, GGML_TYPE_F32, seq_len_out, out_channels, 1);

        // Apply upsample stage (transposed conv + SnakeBeta)
        upsample_stage(
            stage_out,
            current,
            upsample_weights[stage],
            upsample_alphas[stage],
            upsample_betas[stage],
            kernel_sizes[stage],
            stride,
            paddings[stage]
        );

        current = stage_out;
    }

    // Step 4: Output audio (flatten to 1D)
    // current should be [seq_len * 480, 1, 1] after 4 upsamples
    // Copy to dst which should be [seq_len * 480]
    const int64_t total_samples = current->ne[0];
    GGML_ASSERT(dst->ne[0] == total_samples);

    float * upsampled_data = (float *)current->data;
    float * dst_data = (float *)dst->data;

    // If output has single channel, copy directly
    if (current->ne[1] == 1) {
        memcpy(dst_data, upsampled_data, ggml_nbytes(dst));
    } else {
        // If multiple channels, take first channel only
        for (int64_t i = 0; i < total_samples; i++) {
            dst_data[i] = upsampled_data[i];
        }
    }

    // Cleanup
    free_ggml_context(temp_ctx);
}

// Vocoder forward pass interface
// Input: codes [16, seq_len] - RVQ codes (int32 tensor)
// Output: audio samples [seq_len * 480] (float32 tensor)
//
// Wraps vocoder_decode with standard configuration:
// - 4 upsample stages with rates [8, 5, 4, 3] = 480x total upsampling
// - Standard kernel sizes and padding
// - Optional transformer decoder (pass nullptr for transformer_weights to skip)
void vocoder_forward(
    struct ggml_tensor * audio_out,
    const struct ggml_tensor * codes,
    const struct ggml_tensor * codebooks,
    const struct ggml_tensor ** upsample_weights,
    const struct ggml_tensor ** upsample_alphas,
    const struct ggml_tensor ** upsample_betas
) {
    GGML_ASSERT(codes->type == GGML_TYPE_I32);
    GGML_ASSERT(codebooks->type == GGML_TYPE_F32);
    GGML_ASSERT(audio_out->type == GGML_TYPE_F32);

    const int64_t seq_len = codes->ne[0];
    const int64_t num_codebooks = codes->ne[1];
    const int64_t expected_audio_len = seq_len * 480;  // 12Hz → 24kHz upsampling

    GGML_ASSERT(num_codebooks == 16);
    GGML_ASSERT(audio_out->ne[0] == expected_audio_len);

    // Standard vocoder configuration from Qwen3-TTS
    // Upsample rates: 8x → 5x → 4x → 3x = 480x total
    int kernel_sizes[NUM_UPSAMPLE_STAGES] = {16, 10, 8, 6};  // Standard kernel sizes
    int paddings[NUM_UPSAMPLE_STAGES] = {4, 2, 2, 1};        // Standard padding

    // Call the full vocoder_decode implementation
    // Pass nullptr for transformer_weights to skip transformer stage (not loaded yet)
    vocoder_decode(
        audio_out,
        codes,
        codebooks,
        upsample_weights,
        upsample_alphas,
        upsample_betas,
        kernel_sizes,
        paddings,
        nullptr  // transformer_weights - skip for now
    );
}

} // namespace vocoder
} // namespace leaxer_qwen
