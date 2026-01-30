// Model weight structs for Qwen3-TTS
// Defines structures to hold tensor pointers for all model components

#ifndef LEAXER_QWEN_MODEL_WEIGHTS_H
#define LEAXER_QWEN_MODEL_WEIGHTS_H

#include "ggml.h"

namespace leaxer_qwen {
namespace model {

// Single transformer layer weights for Talker (main LLM)
// 28 layers in 0.6B model, hidden_size=1024, intermediate=3072
// GQA: 16 query heads, 8 KV heads, head_dim=64
struct TalkerLayer {
    // Pre-attention normalization
    struct ggml_tensor * in_ln_weight;           // [hidden_dim] RMSNorm weight

    // Grouped Query Attention
    struct ggml_tensor * attn_q_proj_weight;     // [hidden_dim, num_heads * head_dim]
    struct ggml_tensor * attn_k_proj_weight;     // [hidden_dim, num_kv_heads * head_dim]
    struct ggml_tensor * attn_v_proj_weight;     // [hidden_dim, num_kv_heads * head_dim]
    struct ggml_tensor * attn_o_proj_weight;     // [hidden_dim, num_heads * head_dim]

    // Q/K normalization (RMSNorm applied to Q and K after projection)
    struct ggml_tensor * attn_q_norm_weight;     // [head_dim] RMSNorm per head
    struct ggml_tensor * attn_k_norm_weight;     // [head_dim] RMSNorm per head

    // Pre-FFN normalization
    struct ggml_tensor * post_ln_weight;         // [hidden_dim] RMSNorm weight

    // SwiGLU Feed-Forward Network
    struct ggml_tensor * ffn_gate_proj_weight;   // [hidden_dim, intermediate_dim]
    struct ggml_tensor * ffn_up_proj_weight;     // [hidden_dim, intermediate_dim]
    struct ggml_tensor * ffn_down_proj_weight;   // [intermediate_dim, hidden_dim]
};

// Talker model weights (LLM component)
struct TalkerWeights {
    // Token embeddings (text domain, 151936 vocab)
    struct ggml_tensor * emb_weight;             // [vocab_size, embedding_dim=2048]

    // Codec embeddings (codec domain, 3072 vocab including special tokens)
    // CRITICAL: Input to talker is: text_projection(text_embed) + codec_embed
    // Both must be 1024-dim and are SUMMED, not concatenated!
    struct ggml_tensor * codec_embedding_weight; // [codec_vocab=3072, hidden_dim=1024]

    // Text projection (embedding_dim → hidden_dim)
    // Flow: input(2048) → fc1 → SiLU → fc2 → output(1024)
    struct ggml_tensor * text_proj_fc1_weight;   // [embedding_dim, embedding_dim]
    struct ggml_tensor * text_proj_fc1_bias;     // [embedding_dim]
    struct ggml_tensor * text_proj_fc2_weight;   // [hidden_dim, embedding_dim]
    struct ggml_tensor * text_proj_fc2_bias;     // [hidden_dim]

    // 28 transformer layers
    TalkerLayer layers[28];

    // Final layer normalization
    struct ggml_tensor * norm_weight;            // [hidden_dim] RMSNorm weight

    // Language model head (semantic codebook prediction)
    // Outputs logits over 3072 vocab (2048 audio codes + special tokens)
    struct ggml_tensor * lm_head_weight;         // [hidden_dim, codec_vocab=3072]
};

// Single transformer layer for Code Predictor
// 5 layers, same architecture as Talker but different dimensions
struct CodePredictorLayer {
    // Pre-attention normalization
    struct ggml_tensor * in_ln_weight;           // [hidden_dim] RMSNorm weight

    // Grouped Query Attention
    struct ggml_tensor * attn_q_proj_weight;     // [hidden_dim, num_heads * head_dim]
    struct ggml_tensor * attn_k_proj_weight;     // [hidden_dim, num_kv_heads * head_dim]
    struct ggml_tensor * attn_v_proj_weight;     // [hidden_dim, num_kv_heads * head_dim]
    struct ggml_tensor * attn_o_proj_weight;     // [hidden_dim, num_heads * head_dim]

    // Q/K normalization (RMSNorm applied to Q and K after projection)
    struct ggml_tensor * attn_q_norm_weight;     // [head_dim] RMSNorm per head
    struct ggml_tensor * attn_k_norm_weight;     // [head_dim] RMSNorm per head

    // Pre-FFN normalization
    struct ggml_tensor * post_ln_weight;         // [hidden_dim] RMSNorm weight

    // SwiGLU Feed-Forward Network
    struct ggml_tensor * ffn_gate_proj_weight;   // [hidden_dim, intermediate_dim]
    struct ggml_tensor * ffn_up_proj_weight;     // [hidden_dim, intermediate_dim]
    struct ggml_tensor * ffn_down_proj_weight;   // [intermediate_dim, hidden_dim]
};

// Code Predictor weights (acoustic codebook prediction)
// Predicts all 16 codebook tokens given semantic codes
struct CodePredictorWeights {
    // Codec token embeddings (one per codebook, 16 total)
    // Each: [hidden_dim, codebook_vocab] = [1024, 2048]
    struct ggml_tensor * codec_embeddings[16];

    // 5 transformer layers
    CodePredictorLayer layers[5];

    // Final layer normalization
    struct ggml_tensor * norm_weight;            // [hidden_dim] RMSNorm weight

    // Output projection heads (one per acoustic codebook, 15 total)
    // First codebook (semantic) is predicted by main Talker LLM
    struct ggml_tensor * output_heads[15];       // Each: [hidden_dim, codebook_vocab]
};

// ConvNeXt block weights
// Standard ConvNeXt architecture: depthwise conv → layernorm → pointwise expansion → pointwise contraction
// Currently not used in vocoder pipeline (transformer blocks used instead)
struct ConvNeXtWeights {
    // Depthwise 7x1 convolution
    struct ggml_tensor * dw_conv_weight;         // [kernel_size, channels, 1]
    struct ggml_tensor * dw_conv_bias;           // [channels]

    // Layer normalization
    struct ggml_tensor * norm_weight;            // [channels]
    struct ggml_tensor * norm_bias;              // [channels]

    // Pointwise expansion (channels → 4*channels)
    struct ggml_tensor * pw_expand_weight;       // [channels, 4*channels]
    struct ggml_tensor * pw_expand_bias;         // [4*channels]

    // Pointwise contraction (4*channels → channels)
    struct ggml_tensor * pw_contract_weight;     // [4*channels, channels]
    struct ggml_tensor * pw_contract_bias;       // [channels]
};

// Pre-transformer layer weights for vocoder
// 8 layers with self-attention, FFN, layer norms, and layer scales
struct PreTransformerLayer {
    // Layer norms
    struct ggml_tensor * input_ln_weight;        // [hidden_dim] RMSNorm weight
    struct ggml_tensor * post_ln_weight;         // [hidden_dim] RMSNorm weight

    // Self-attention (no KV heads, standard attention)
    struct ggml_tensor * attn_q_weight;          // [hidden_dim, head_dim * num_heads]
    struct ggml_tensor * attn_k_weight;          // [hidden_dim, head_dim * num_heads]
    struct ggml_tensor * attn_v_weight;          // [hidden_dim, head_dim * num_heads]
    struct ggml_tensor * attn_o_weight;          // [head_dim * num_heads, hidden_dim]

    // SwiGLU Feed-Forward Network
    struct ggml_tensor * ffn_gate_weight;        // [hidden_dim, intermediate_dim]
    struct ggml_tensor * ffn_up_weight;          // [hidden_dim, intermediate_dim]
    struct ggml_tensor * ffn_down_weight;        // [intermediate_dim, hidden_dim]

    // Layer scales (for residual scaling)
    struct ggml_tensor * attn_scale;             // [hidden_dim]
    struct ggml_tensor * ffn_scale;              // [hidden_dim]
};

// ResBlock weights for vocoder
// Each ResBlock: act1 (SnakeBeta) → conv1 (k=7) → act2 (SnakeBeta) → conv2 (k=1) + residual
struct VocoderResBlock {
    // SnakeBeta activation 1
    struct ggml_tensor * act1_alpha;         // [channels]
    struct ggml_tensor * act1_beta;          // [channels]

    // Convolution 1 (kernel=7, same channels)
    struct ggml_tensor * conv1_weight;       // [7, channels, channels] - 1D conv
    struct ggml_tensor * conv1_bias;         // [channels]

    // SnakeBeta activation 2
    struct ggml_tensor * act2_alpha;         // [channels]
    struct ggml_tensor * act2_beta;          // [channels]

    // Convolution 2 (kernel=1, same channels)
    struct ggml_tensor * conv2_weight;       // [1, channels, channels] - 1D conv (pointwise)
    struct ggml_tensor * conv2_bias;         // [channels]
};

// ConvNeXt block weights (used in upsample stages after pre-transformer)
// Structure: depthwise conv → layernorm → pointwise expansion → pointwise contraction
struct VocoderConvNeXtBlock {
    // Transposed convolution for upsampling
    struct ggml_tensor * transconv_weight;   // [kernel, out_ch, in_ch]
    struct ggml_tensor * transconv_bias;     // [out_ch]

    // Depthwise 7x1 convolution
    struct ggml_tensor * dwconv_weight;      // [7, channels, 1] - depthwise
    struct ggml_tensor * dwconv_bias;        // [channels]

    // Layer normalization
    struct ggml_tensor * norm_weight;        // [channels]
    struct ggml_tensor * norm_bias;          // [channels]

    // Pointwise expansion (channels → 4*channels)
    struct ggml_tensor * pwconv1_weight;     // [4*channels, channels]
    struct ggml_tensor * pwconv1_bias;       // [4*channels]

    // Pointwise contraction (4*channels → channels)
    struct ggml_tensor * pwconv2_weight;     // [channels, 4*channels]
    struct ggml_tensor * pwconv2_bias;       // [channels]

    // Gamma scale parameter
    struct ggml_tensor * gamma;              // [channels]
};

// Upsample stage with ResBlocks
// Structure: SnakeBeta → TransposedConv → 3 ResBlocks
struct VocoderUpsampleStage {
    // SnakeBeta activation (before transposed conv)
    struct ggml_tensor * snake_alpha;        // [in_channels]
    struct ggml_tensor * snake_beta;         // [in_channels]

    // Transposed convolution for upsampling
    struct ggml_tensor * conv_weight;        // [kernel, out_channels, in_channels]
    struct ggml_tensor * conv_bias;          // [out_channels]

    // 3 ResBlocks (after upsampling, operate on out_channels)
    VocoderResBlock resblocks[3];
};

// Vocoder weights (decoder from Tokenizer-12Hz model)
// Converts codec tokens to 24kHz waveform
// Full architecture:
//   1. RVQ decode (codebooks) → 256-dim per codebook
//   2. Output projections (256→512 each) → concat → 1024-dim
//   3. Pre-transformer (8 layers, 512-dim internal) with input/output projections
//   4. Causal conv (1024→1536)
//   5. Upsample stages (1536→768→384→192→96) with ResBlocks
//   6. Final snake + conv (96→1)
struct VocoderWeights {
    // RVQ codebook embeddings
    struct ggml_tensor * codebooks;              // [16, codebook_vocab, codebook_dim=256]

    // RVQ output projections (project each codebook group output before concat)
    struct ggml_tensor * rvq_first_output_proj;  // [512, 256, 1] - 1D conv for first codebook
    struct ggml_tensor * rvq_rest_output_proj;   // [512, 256, 1] - 1D conv for codebooks 1-15

    // Pre-conv layer (512→1024, kernel=3) - comes after RVQ, before pre-transformer
    struct ggml_tensor * pre_conv_weight;        // [1024, 512, 3] - PyTorch Conv1d format
    struct ggml_tensor * pre_conv_bias;          // [1024]

    // Pre-transformer input/output projections
    struct ggml_tensor * pre_transformer_input_proj_weight;   // [512, 1024] - project concat to transformer dim
    struct ggml_tensor * pre_transformer_input_proj_bias;     // [512]
    struct ggml_tensor * pre_transformer_output_proj_weight;  // [1024, 512] - project back to decoder dim
    struct ggml_tensor * pre_transformer_output_proj_bias;    // [1024]

    // Pre-transformer layers (8 layers)
    PreTransformerLayer pre_transformer_layers[8];

    // Upsample ConvNeXt blocks (2 stages, after pre-transformer, before decoder)
    // Each stage: TransConv (stride=2) + ConvNeXt block
    VocoderConvNeXtBlock upsample_convnext[2];

    // Causal convolution (projects pre-transformer output to upsample input)
    struct ggml_tensor * causal_conv_weight;     // [7, 1024, 1536]
    struct ggml_tensor * causal_conv_bias;       // [1536]

    // 4-stage progressive upsampling (12Hz → 24kHz)
    // Each stage: SnakeBeta → TransposedConv → 3 ResBlocks
    VocoderUpsampleStage upsample_stages[4];

    // Legacy arrays for backward compatibility during transition
    struct ggml_tensor * upsample_alphas[4];     // SnakeBeta alpha (log scale)
    struct ggml_tensor * upsample_betas[4];      // SnakeBeta beta (log scale)
    struct ggml_tensor * upsample_weights[4];    // Transposed conv weights
    struct ggml_tensor * upsample_biases[4];     // Transposed conv biases

    // Final SnakeBeta activation
    struct ggml_tensor * final_snake_alpha;      // [96]
    struct ggml_tensor * final_snake_beta;       // [96]

    // Final projection to waveform
    struct ggml_tensor * final_conv_weight;      // [7, 96, 1]
    struct ggml_tensor * final_conv_bias;        // [1]
};

// Top-level model structure
// Combines Talker (LLM), Code Predictor, and Vocoder
struct Qwen3TTSModel {
    // Model configuration
    int vocab_size;              // 151936 + special tokens
    int hidden_dim;              // 1024 for 0.6B model
    int intermediate_dim;        // 3072 for 0.6B model
    int num_layers;              // 28 for 0.6B model
    int num_heads;               // 16
    int num_kv_heads;            // 8 (GQA)
    int head_dim;                // 64

    int code_pred_layers;        // 5
    int code_pred_heads;         // 16
    int code_pred_kv_heads;      // 8

    int num_codebooks;           // 16
    int codebook_vocab;          // 2048
    int codebook_dim;            // 512

    // Model components
    TalkerWeights talker;
    CodePredictorWeights code_predictor;
    VocoderWeights vocoder;

    // Component availability flags
    bool vocoder_loaded;         // true if vocoder weights were successfully loaded

    // ggml context holding the weights
    struct ggml_context * ctx;
};

} // namespace model
} // namespace leaxer_qwen

#endif // LEAXER_QWEN_MODEL_WEIGHTS_H
