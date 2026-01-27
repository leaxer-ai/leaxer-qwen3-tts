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

    // Pre-FFN normalization
    struct ggml_tensor * post_ln_weight;         // [hidden_dim] RMSNorm weight

    // SwiGLU Feed-Forward Network
    struct ggml_tensor * ffn_gate_proj_weight;   // [hidden_dim, intermediate_dim]
    struct ggml_tensor * ffn_up_proj_weight;     // [hidden_dim, intermediate_dim]
    struct ggml_tensor * ffn_down_proj_weight;   // [intermediate_dim, hidden_dim]
};

// Talker model weights (LLM component)
struct TalkerWeights {
    // Token embeddings
    struct ggml_tensor * emb_weight;             // [vocab_size, hidden_dim]

    // 28 transformer layers
    TalkerLayer layers[28];

    // Final layer normalization
    struct ggml_tensor * norm_weight;            // [hidden_dim] RMSNorm weight

    // Language model head (semantic codebook prediction)
    struct ggml_tensor * lm_head_weight;         // [hidden_dim, semantic_vocab]
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
    // Codec token embedding (for conditioning on semantic codes)
    struct ggml_tensor * codec_embedding_weight; // [codebook_vocab, hidden_dim]

    // 5 transformer layers
    CodePredictorLayer layers[5];

    // Final layer normalization
    struct ggml_tensor * norm_weight;            // [hidden_dim] RMSNorm weight

    // Output projection heads (one per codebook)
    struct ggml_tensor * output_heads[16];       // Each: [hidden_dim, codebook_vocab]
};

// Vocoder weights (decoder from Tokenizer-12Hz model)
// Converts codec tokens to 24kHz waveform
struct VocoderWeights {
    // RVQ codebook embeddings
    struct ggml_tensor * codebooks;              // [16, codebook_vocab, codebook_dim]

    // Initial causal convolution (projects RVQ output)
    struct ggml_tensor * causal_conv_weight;     // [kernel_size, in_channels, out_channels]
    struct ggml_tensor * causal_conv_bias;       // [out_channels]

    // ConvNeXt blocks (transformer-like processing)
    // TODO: Add ConvNeXt block weights when implemented
    // Each block has: depthwise conv, norm, pointwise conv layers

    // 4-stage progressive upsampling (12Hz → 24kHz)
    // Each stage: transposed conv + SnakeBeta activation
    struct ggml_tensor * upsample_weights[4];    // Transposed conv weights
    struct ggml_tensor * upsample_biases[4];     // Transposed conv biases
    struct ggml_tensor * upsample_alphas[4];     // SnakeBeta alpha (log scale)
    struct ggml_tensor * upsample_betas[4];      // SnakeBeta beta (log scale)

    // Final projection to waveform
    struct ggml_tensor * final_conv_weight;      // [kernel_size, channels, 1]
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

    // ggml context holding the weights
    struct ggml_context * ctx;
};

} // namespace model
} // namespace leaxer_qwen

#endif // LEAXER_QWEN_MODEL_WEIGHTS_H
