// Full Vocoder: Qwen3TTSTokenizerV2Decoder
// Converts discrete codec tokens to audio waveform
//
// Full architecture:
//   1. RVQ decode (codebooks) → 256-dim per codebook
//   2. Output projections (256→512 each) → concat → 1024-dim
//   3. Pre-transformer (8 layers, 512-dim internal) with input/output projections
//   4. Causal conv (1024→1536)
//   5. Upsample stages (1536→768→384→192→96) with SnakeBeta
//   6. Final SnakeBeta + conv (96→1)

#include "ggml.h"
#include "ggml-cpu.h"
#include "common.h"
#include "model/model_weights.h"
#include <cstring>
#include <cstdint>
#include <cmath>

namespace leaxer_qwen {
namespace vocoder {

// Configuration
constexpr int NUM_CODEBOOKS = 16;
constexpr int CODEBOOK_DIM = 256;
constexpr int PROJ_DIM = 512;
constexpr int CONCAT_DIM = 1024;  // 512 + 512
constexpr int PRE_TRANSFORMER_LAYERS = 8;
constexpr int CAUSAL_CONV_OUT = 1536;
constexpr int UPSAMPLE_RATES[] = {8, 5, 4, 3};
constexpr int NUM_UPSAMPLE_STAGES = 4;

// Forward declarations
namespace ops {
void snake_beta_inplace(
    struct ggml_tensor * dst,
    const struct ggml_tensor * x,
    const struct ggml_tensor * alpha_logscale,
    const struct ggml_tensor * beta_logscale);
}

// Forward declare F16->F32 conversion
static float * convert_f16_to_f32(const struct ggml_tensor * tensor);

// RVQ decode for first codebook (codebook 0 = semantic)
// codes: [seq_len, 16] int32
// codebooks: [16, 2048, 256] float32
// dst: [seq_len, 256] float32 - embeddings for codebook 0 only
static void rvq_decode_first(
    float * dst,
    const int32_t * codes,
    const float * codebooks,
    int64_t seq_len,
    int64_t codebook_size,
    int64_t codebook_dim
) {
    // Zero output
    memset(dst, 0, seq_len * codebook_dim * sizeof(float));

    // First codebook only (index 0)
    const float * cb_embedding = codebooks + 0 * codebook_size * codebook_dim;

    for (int64_t t = 0; t < seq_len; t++) {
        int32_t idx = codes[t * NUM_CODEBOOKS + 0];  // Codebook 0
        if (idx < 0 || idx >= codebook_size) {
            fprintf(stderr, "Warning: codebook 0 index out of range: %d\n", idx);
            continue;
        }

        // Copy embedding
        for (int64_t d = 0; d < codebook_dim; d++) {
            dst[t * codebook_dim + d] = cb_embedding[idx * codebook_dim + d];
        }
    }
}

// RVQ decode for rest codebooks (codebooks 1-15 = acoustic)
// codes: [seq_len, 16] int32
// codebooks: [16, 2048, 256] float32
// dst: [seq_len, 256] float32 - SUM of embeddings for codebooks 1-15
static void rvq_decode_rest(
    float * dst,
    const int32_t * codes,
    const float * codebooks,
    int64_t seq_len,
    int64_t codebook_size,
    int64_t codebook_dim
) {
    // Zero output
    memset(dst, 0, seq_len * codebook_dim * sizeof(float));

    // Sum codebooks 1-15
    for (int cb = 1; cb < NUM_CODEBOOKS; cb++) {
        const float * cb_embedding = codebooks + cb * codebook_size * codebook_dim;

        for (int64_t t = 0; t < seq_len; t++) {
            int32_t idx = codes[t * NUM_CODEBOOKS + cb];
            if (idx < 0 || idx >= codebook_size) {
                fprintf(stderr, "Warning: codebook %d index out of range: %d\n", cb, idx);
                continue;
            }

            // Add embedding to output
            for (int64_t d = 0; d < codebook_dim; d++) {
                dst[t * codebook_dim + d] += cb_embedding[idx * codebook_dim + d];
            }
        }
    }
}

// Apply 1D convolution with kernel_size=1 (effectively a linear projection)
// input: [seq_len, in_dim]
// weight: [out_dim, in_dim, 1] - conv weight
// output: [seq_len, out_dim]
static void conv1d_k1(
    float * output,
    const float * input,
    const float * weight,
    int64_t seq_len,
    int64_t in_dim,
    int64_t out_dim
) {
    for (int64_t t = 0; t < seq_len; t++) {
        for (int64_t o = 0; o < out_dim; o++) {
            float sum = 0.0f;
            for (int64_t i = 0; i < in_dim; i++) {
                sum += input[t * in_dim + i] * weight[o * in_dim + i];
            }
            output[t * out_dim + o] = sum;
        }
    }
}

// Apply linear projection (matrix multiply)
// input: [seq_len, in_dim]
// weight: [out_dim, in_dim]
// bias: [out_dim] or nullptr
// output: [seq_len, out_dim]
static void linear(
    float * output,
    const float * input,
    const float * weight,
    const float * bias,
    int64_t seq_len,
    int64_t in_dim,
    int64_t out_dim
) {
    for (int64_t t = 0; t < seq_len; t++) {
        for (int64_t o = 0; o < out_dim; o++) {
            float sum = bias ? bias[o] : 0.0f;
            for (int64_t i = 0; i < in_dim; i++) {
                sum += input[t * in_dim + i] * weight[o * in_dim + i];
            }
            output[t * out_dim + o] = sum;
        }
    }
}

// RMS normalization
static void rms_norm(
    float * output,
    const float * input,
    const float * weight,
    int64_t seq_len,
    int64_t dim,
    float eps = 1e-6f
) {
    for (int64_t t = 0; t < seq_len; t++) {
        // Compute RMS
        float sum_sq = 0.0f;
        for (int64_t d = 0; d < dim; d++) {
            float v = input[t * dim + d];
            sum_sq += v * v;
        }
        float rms = sqrtf(sum_sq / dim + eps);

        // Normalize and scale
        for (int64_t d = 0; d < dim; d++) {
            output[t * dim + d] = (input[t * dim + d] / rms) * weight[d];
        }
    }
}

// SiLU activation (swish): x * sigmoid(x)
static inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

// SnakeBeta activation: x + (1/b) * sin^2(a * x)
// alpha_logscale and beta_logscale are log-scale parameters
static void snake_beta(
    float * output,
    const float * input,
    const float * alpha_logscale,
    const float * beta_logscale,
    int64_t seq_len,
    int64_t dim
) {
    for (int64_t t = 0; t < seq_len; t++) {
        for (int64_t d = 0; d < dim; d++) {
            float x = input[t * dim + d];
            float a = expf(alpha_logscale[d]);
            float b = expf(beta_logscale[d]);
            float sin_ax = sinf(a * x);
            output[t * dim + d] = x + (1.0f / b) * sin_ax * sin_ax;
        }
    }
}

// 1D convolution with arbitrary kernel size and same-padding
// input: [seq_len, in_channels]
// weight: [kernel_size, out_channels, in_channels] (GGML layout) OR [out_channels, in_channels, kernel_size] (PyTorch layout)
// bias: [out_channels]
// output: [seq_len, out_channels]
static void conv1d_same_padding(
    float * output,
    const float * input,
    const float * weight,
    const float * bias,
    int64_t seq_len,
    int64_t in_channels,
    int64_t out_channels,
    int kernel_size,
    bool pytorch_layout = false  // If true, weight is [out, in, kernel]
) {
    int pad = kernel_size / 2;  // Same padding

    for (int64_t t = 0; t < seq_len; t++) {
        for (int64_t o = 0; o < out_channels; o++) {
            float sum = bias ? bias[o] : 0.0f;

            for (int k = 0; k < kernel_size; k++) {
                int64_t t_in = t - pad + k;
                if (t_in >= 0 && t_in < seq_len) {
                    for (int64_t i = 0; i < in_channels; i++) {
                        float w;
                        if (pytorch_layout) {
                            // PyTorch: [out_channels, in_channels, kernel_size]
                            w = weight[(o * in_channels + i) * kernel_size + k];
                        } else {
                            // GGML: [kernel_size, out_channels, in_channels] -> idx = i * out_ch * kernel + o * kernel + k
                            w = weight[i * out_channels * kernel_size + o * kernel_size + k];
                        }
                        sum += input[t_in * in_channels + i] * w;
                    }
                }
            }
            output[t * out_channels + o] = sum;
        }
    }
}

// ResBlock forward pass
// Structure: x_in = x; x = act1(x) -> conv1(x) -> act2(x) -> conv2(x) -> x + x_in
static void resblock_forward(
    float * output,
    const float * input,
    const model::VocoderResBlock * weights,
    float * act_buf,
    float * conv_buf,
    int64_t seq_len,
    int64_t channels
) {
    // Convert weights to F32 (they may be F16)
    float * act1_alpha = convert_f16_to_f32(weights->act1_alpha);
    float * act1_beta = convert_f16_to_f32(weights->act1_beta);
    float * conv1_w = convert_f16_to_f32(weights->conv1_weight);
    float * conv1_b = convert_f16_to_f32(weights->conv1_bias);
    float * act2_alpha = convert_f16_to_f32(weights->act2_alpha);
    float * act2_beta = convert_f16_to_f32(weights->act2_beta);
    float * conv2_w = convert_f16_to_f32(weights->conv2_weight);
    float * conv2_b = convert_f16_to_f32(weights->conv2_bias);

    // act1: SnakeBeta
    snake_beta(act_buf, input, act1_alpha, act1_beta, seq_len, channels);

    // conv1: kernel=7, same channels
    // PyTorch stores as [out, in, kernel] = [channels, channels, 7]
    conv1d_same_padding(conv_buf, act_buf, conv1_w, conv1_b,
                        seq_len, channels, channels, 7, true);

    // act2: SnakeBeta
    snake_beta(act_buf, conv_buf, act2_alpha, act2_beta, seq_len, channels);

    // conv2: kernel=1, same channels (pointwise)
    conv1d_same_padding(conv_buf, act_buf, conv2_w, conv2_b,
                        seq_len, channels, channels, 1, true);

    // Residual connection: output = conv_buf + input
    for (int64_t i = 0; i < seq_len * channels; i++) {
        output[i] = conv_buf[i] + input[i];
    }

    // Cleanup
    free(act1_alpha);
    free(act1_beta);
    free(conv1_w);
    free(conv1_b);
    free(act2_alpha);
    free(act2_beta);
    free(conv2_w);
    free(conv2_b);
}

// SwiGLU FFN: down(silu(gate(x)) * up(x))
static void swiglu_ffn(
    float * output,
    const float * input,
    const float * gate_weight,
    const float * up_weight,
    const float * down_weight,
    float * gate_buf,
    float * up_buf,
    int64_t seq_len,
    int64_t hidden_dim,
    int64_t intermediate_dim
) {
    // Gate projection
    linear(gate_buf, input, gate_weight, nullptr, seq_len, hidden_dim, intermediate_dim);

    // Up projection
    linear(up_buf, input, up_weight, nullptr, seq_len, hidden_dim, intermediate_dim);

    // SiLU(gate) * up
    for (int64_t t = 0; t < seq_len; t++) {
        for (int64_t i = 0; i < intermediate_dim; i++) {
            gate_buf[t * intermediate_dim + i] = silu(gate_buf[t * intermediate_dim + i]) * up_buf[t * intermediate_dim + i];
        }
    }

    // Down projection
    linear(output, gate_buf, down_weight, nullptr, seq_len, intermediate_dim, hidden_dim);
}

// Simple self-attention (no KV cache, no masking for vocoder)
// input: [seq_len, hidden_dim]
// output: [seq_len, hidden_dim]
static void self_attention(
    float * output,
    const float * input,
    const float * q_weight,
    const float * k_weight,
    const float * v_weight,
    const float * o_weight,
    float * q_buf,
    float * k_buf,
    float * v_buf,
    float * attn_buf,
    int64_t seq_len,
    int64_t hidden_dim,
    int64_t head_dim,
    int num_heads
) {
    // Q, K, V projections
    linear(q_buf, input, q_weight, nullptr, seq_len, hidden_dim, num_heads * head_dim);
    linear(k_buf, input, k_weight, nullptr, seq_len, hidden_dim, num_heads * head_dim);
    linear(v_buf, input, v_weight, nullptr, seq_len, hidden_dim, num_heads * head_dim);

    float scale = 1.0f / sqrtf((float)head_dim);

    // Multi-head attention
    // For simplicity, process all heads together
    // attn_buf: [seq_len, seq_len] attention scores
    for (int h = 0; h < num_heads; h++) {
        // Compute attention scores for this head
        for (int64_t i = 0; i < seq_len; i++) {
            for (int64_t j = 0; j < seq_len; j++) {
                float score = 0.0f;
                for (int64_t d = 0; d < head_dim; d++) {
                    score += q_buf[i * num_heads * head_dim + h * head_dim + d] *
                             k_buf[j * num_heads * head_dim + h * head_dim + d];
                }
                attn_buf[i * seq_len + j] = score * scale;
            }
        }

        // Softmax
        for (int64_t i = 0; i < seq_len; i++) {
            float max_val = attn_buf[i * seq_len];
            for (int64_t j = 1; j < seq_len; j++) {
                max_val = fmaxf(max_val, attn_buf[i * seq_len + j]);
            }

            float sum = 0.0f;
            for (int64_t j = 0; j < seq_len; j++) {
                attn_buf[i * seq_len + j] = expf(attn_buf[i * seq_len + j] - max_val);
                sum += attn_buf[i * seq_len + j];
            }

            for (int64_t j = 0; j < seq_len; j++) {
                attn_buf[i * seq_len + j] /= sum;
            }
        }

        // Apply attention to values
        for (int64_t i = 0; i < seq_len; i++) {
            for (int64_t d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                for (int64_t j = 0; j < seq_len; j++) {
                    sum += attn_buf[i * seq_len + j] *
                           v_buf[j * num_heads * head_dim + h * head_dim + d];
                }
                // Store in q_buf temporarily (we don't need Q anymore)
                q_buf[i * num_heads * head_dim + h * head_dim + d] = sum;
            }
        }
    }

    // Output projection
    linear(output, q_buf, o_weight, nullptr, seq_len, num_heads * head_dim, hidden_dim);
}

// Pre-transformer layer weights converted to F32
struct PreTransformerLayerF32 {
    float * input_ln_weight;
    float * post_ln_weight;
    float * attn_q_weight;
    float * attn_k_weight;
    float * attn_v_weight;
    float * attn_o_weight;
    float * ffn_gate_weight;
    float * ffn_up_weight;
    float * ffn_down_weight;
    float * attn_scale;
    float * ffn_scale;
};

// Convert a pre-transformer layer's weights from F16 to F32
static PreTransformerLayerF32 convert_pre_transformer_layer(const model::PreTransformerLayer * layer) {
    PreTransformerLayerF32 f32_layer = {};
    f32_layer.input_ln_weight = convert_f16_to_f32(layer->input_ln_weight);
    f32_layer.post_ln_weight = convert_f16_to_f32(layer->post_ln_weight);
    f32_layer.attn_q_weight = convert_f16_to_f32(layer->attn_q_weight);
    f32_layer.attn_k_weight = convert_f16_to_f32(layer->attn_k_weight);
    f32_layer.attn_v_weight = convert_f16_to_f32(layer->attn_v_weight);
    f32_layer.attn_o_weight = convert_f16_to_f32(layer->attn_o_weight);
    f32_layer.ffn_gate_weight = convert_f16_to_f32(layer->ffn_gate_weight);
    f32_layer.ffn_up_weight = convert_f16_to_f32(layer->ffn_up_weight);
    f32_layer.ffn_down_weight = convert_f16_to_f32(layer->ffn_down_weight);
    f32_layer.attn_scale = convert_f16_to_f32(layer->attn_scale);
    f32_layer.ffn_scale = convert_f16_to_f32(layer->ffn_scale);
    return f32_layer;
}

// Free converted pre-transformer layer weights
static void free_pre_transformer_layer_f32(PreTransformerLayerF32 * layer) {
    free(layer->input_ln_weight);
    free(layer->post_ln_weight);
    free(layer->attn_q_weight);
    free(layer->attn_k_weight);
    free(layer->attn_v_weight);
    free(layer->attn_o_weight);
    free(layer->ffn_gate_weight);
    free(layer->ffn_up_weight);
    free(layer->ffn_down_weight);
    free(layer->attn_scale);
    free(layer->ffn_scale);
}

// Single pre-transformer layer
static void pre_transformer_layer(
    float * output,
    const float * input,
    const PreTransformerLayerF32 * layer,
    float * norm_buf,
    float * attn_out,
    float * ffn_out,
    float * q_buf,
    float * k_buf,
    float * v_buf,
    float * attn_scores,
    float * gate_buf,
    float * up_buf,
    int64_t seq_len,
    int64_t hidden_dim,
    int64_t intermediate_dim
) {
    const int num_heads = 8;  // 512 / 64 = 8 heads
    const int head_dim = 64;

    // Input layer norm
    rms_norm(norm_buf, input, layer->input_ln_weight, seq_len, hidden_dim);

    // Self-attention
    self_attention(
        attn_out, norm_buf,
        layer->attn_q_weight,
        layer->attn_k_weight,
        layer->attn_v_weight,
        layer->attn_o_weight,
        q_buf, k_buf, v_buf, attn_scores,
        seq_len, hidden_dim, head_dim, num_heads
    );

    // Apply attention layer scale and residual
    for (int64_t t = 0; t < seq_len; t++) {
        for (int64_t d = 0; d < hidden_dim; d++) {
            output[t * hidden_dim + d] = input[t * hidden_dim + d] + attn_out[t * hidden_dim + d] * layer->attn_scale[d];
        }
    }

    // Post-attention layer norm
    rms_norm(norm_buf, output, layer->post_ln_weight, seq_len, hidden_dim);

    // FFN
    swiglu_ffn(
        ffn_out, norm_buf,
        layer->ffn_gate_weight,
        layer->ffn_up_weight,
        layer->ffn_down_weight,
        gate_buf, up_buf,
        seq_len, hidden_dim, intermediate_dim
    );

    // Apply FFN layer scale and residual
    for (int64_t t = 0; t < seq_len; t++) {
        for (int64_t d = 0; d < hidden_dim; d++) {
            output[t * hidden_dim + d] += ffn_out[t * hidden_dim + d] * layer->ffn_scale[d];
        }
    }
}

// 1D causal convolution
// input: [seq_len, in_channels]
// weight: [out_channels, in_channels, kernel_size]
// bias: [out_channels]
// output: [seq_len, out_channels]
static void causal_conv1d(
    float * output,
    const float * input,
    const float * weight,
    const float * bias,
    int64_t seq_len,
    int64_t in_channels,
    int64_t out_channels,
    int kernel_size
) {
    int pad = kernel_size - 1;  // Causal padding

    for (int64_t t = 0; t < seq_len; t++) {
        for (int64_t o = 0; o < out_channels; o++) {
            float sum = bias ? bias[o] : 0.0f;

            for (int k = 0; k < kernel_size; k++) {
                int64_t t_in = t - (kernel_size - 1 - k);  // Causal: look back
                if (t_in >= 0 && t_in < seq_len) {
                    for (int64_t i = 0; i < in_channels; i++) {
                        // weight layout: [out_channels, in_channels, kernel_size]
                        sum += input[t_in * in_channels + i] * weight[(o * in_channels + i) * kernel_size + k];
                    }
                }
            }

            output[t * out_channels + o] = sum;
        }
    }
}

// Helper to convert F16 tensor data to F32 buffer
// Returns newly allocated F32 buffer (caller must free)
static float * convert_f16_to_f32(const struct ggml_tensor * tensor) {
    if (!tensor) return nullptr;

    // Calculate total number of elements (GGML pads ne[] with 1s for unused dims)
    int64_t n_elements = tensor->ne[0] * tensor->ne[1] * tensor->ne[2] * tensor->ne[3];

    float * f32_data = (float *)malloc(n_elements * sizeof(float));
    if (!f32_data) {
        fprintf(stderr, "Failed to allocate F32 buffer for %lld elements\n", (long long)n_elements);
        return nullptr;
    }

    if (tensor->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *)tensor->data, f32_data, n_elements);
    } else if (tensor->type == GGML_TYPE_F32) {
        memcpy(f32_data, tensor->data, n_elements * sizeof(float));
    } else {
        fprintf(stderr, "Unsupported tensor type %d for conversion\n", (int)tensor->type);
        free(f32_data);
        return nullptr;
    }

    return f32_data;
}

// Transposed 1D convolution for upsampling
// input: [seq_len, in_channels]
// weight: [kernel_size, out_channels, in_channels] - GGML layout (ne[0]=kernel, ne[1]=out_ch, ne[2]=in_ch)
// bias: [out_channels]
// output: [seq_len * stride, out_channels]
static void conv1d_transpose(
    float * output,
    const float * input,
    const float * weight,
    const float * bias,
    int64_t seq_len,
    int64_t in_channels,
    int64_t out_channels,
    int kernel_size,
    int stride,
    int64_t output_buffer_size  // For bounds checking
) {
    int64_t out_len = seq_len * stride;
    int64_t required_output_size = out_len * out_channels;

    if (required_output_size > output_buffer_size) {
        fprintf(stderr, "conv_transpose: output buffer too small! need %lld, have %lld\n",
                (long long)required_output_size, (long long)output_buffer_size);
        return;
    }

    // Initialize with bias
    for (int64_t t = 0; t < out_len; t++) {
        for (int64_t o = 0; o < out_channels; o++) {
            output[t * out_channels + o] = bias ? bias[o] : 0.0f;
        }
    }

    // Transposed convolution
    // GGML layout: weight[i * (out_channels * kernel_size) + o * kernel_size + k]
    // i.e., for element (k, o, i): data[i * ne[0]*ne[1] + o * ne[0] + k]
    int64_t weight_stride = out_channels * kernel_size;

    // Full transposed convolution computation
    for (int64_t t_in = 0; t_in < seq_len; t_in++) {
        for (int k = 0; k < kernel_size; k++) {
            int64_t t_out = t_in * stride + k;
            if (t_out >= out_len) continue;

            for (int64_t i = 0; i < in_channels; i++) {
                int64_t in_idx = t_in * in_channels + i;
                float in_val = input[in_idx];

                int64_t weight_base = i * weight_stride + k;
                for (int64_t o = 0; o < out_channels; o++) {
                    int64_t w_idx = weight_base + o * kernel_size;
                    int64_t out_idx = t_out * out_channels + o;
                    output[out_idx] += in_val * weight[w_idx];
                }
            }
        }
    }
}

// Full vocoder forward pass
void vocoder_full_forward(
    float * audio_out,
    const int32_t * codes,
    int64_t seq_len,
    const model::VocoderWeights * weights
) {
    // Verify inputs
    if (!audio_out || !codes || !weights) {
        fprintf(stderr, "vocoder: null input\n");
        return;
    }
    if (!weights->codebooks) {
        fprintf(stderr, "vocoder: codebooks is null!\n");
        return;
    }

    // Allocate buffers
    const int64_t max_seq = seq_len * 480;  // Maximum output length
    const int64_t intermediate_dim = 1024;  // FFN intermediate


    // Main processing buffers
    // Note: rvq_first and rvq_rest are allocated later in the RVQ decode section
    float * proj_first = (float *)malloc(seq_len * PROJ_DIM * sizeof(float));
    float * proj_rest = (float *)malloc(seq_len * PROJ_DIM * sizeof(float));
    // Note: concat is no longer used - replaced by pre_conv_out allocated dynamically
    float * pre_trans_in = (float *)malloc(seq_len * PROJ_DIM * sizeof(float));
    float * pre_trans_out = (float *)malloc(seq_len * PROJ_DIM * sizeof(float));
    float * post_trans = (float *)malloc(seq_len * CONCAT_DIM * sizeof(float));
    float * causal_out = (float *)malloc(seq_len * CAUSAL_CONV_OUT * sizeof(float));

    // Transformer layer buffers
    float * norm_buf = (float *)malloc(seq_len * PROJ_DIM * sizeof(float));
    float * attn_out = (float *)malloc(seq_len * PROJ_DIM * sizeof(float));
    float * ffn_out = (float *)malloc(seq_len * PROJ_DIM * sizeof(float));
    float * q_buf = (float *)malloc(seq_len * intermediate_dim * sizeof(float));
    float * k_buf = (float *)malloc(seq_len * intermediate_dim * sizeof(float));
    float * v_buf = (float *)malloc(seq_len * intermediate_dim * sizeof(float));
    float * attn_scores = (float *)malloc(seq_len * seq_len * sizeof(float));
    float * gate_buf = (float *)malloc(seq_len * intermediate_dim * sizeof(float));
    float * up_buf = (float *)malloc(seq_len * intermediate_dim * sizeof(float));

    // Upsample buffers - compute actual max size needed
    // ConvNeXt stages add 4x (2*2) upsampling before decoder blocks
    // After ConvNeXt: seq_len * 4 = decoder_seq_len
    // After decoder stage 0: decoder_seq_len * 8 = 4800 samples * 768 channels
    // After decoder stage 1: 4800 * 5 = 24000 samples * 384 channels
    // After decoder stage 2: 24000 * 4 = 96000 samples * 192 channels
    // After decoder stage 3: 96000 * 3 = 288000 samples * 96 channels
    // Max buffer needed: max(4800*768, 24000*384, 96000*192, 288000*96) = 27.6M elements
    // For seq_len=150, that's about 110 MB for two buffers
    const int64_t convnext_factor = 4;  // ConvNeXt 2 stages, each 2x
    const int64_t max_decoder_seq = seq_len * convnext_factor * 8 * 5 * 4 * 3;  // Full decoder upsample
    const int64_t max_upsample_size = max_decoder_seq * 192;  // Stage 2 has most elements (96000 * 192)
    printf("  vocoder: allocating upsample buffers: %lld elements (%lld MB)\n",
           (long long)max_upsample_size, (long long)(max_upsample_size * sizeof(float) / 1024 / 1024));
    float * upsample_in = (float *)malloc(max_upsample_size * sizeof(float));
    float * upsample_out = (float *)malloc(max_upsample_size * sizeof(float));

    if (!upsample_in || !upsample_out) {
        fprintf(stderr, "vocoder: failed to allocate upsample buffers!\n");
        return;
    }

    // Step 1: RVQ decode - CORRECTLY separate first vs rest codebooks
    // Convert codebooks from F16 to F32 if needed
    float * codebooks_f32 = convert_f16_to_f32(weights->codebooks);
    if (!codebooks_f32) {
        fprintf(stderr, "vocoder: failed to convert codebooks to F32\n");
        return;
    }
    int64_t codebook_size = weights->codebooks->ne[1];
    printf("  vocoder: codebooks shape: [%lld, %lld, %lld], type=%d (converted to F32)\n",
           (long long)weights->codebooks->ne[0],
           (long long)weights->codebooks->ne[1],
           (long long)weights->codebooks->ne[2],
           (int)weights->codebooks->type);
    printf("  vocoder: first code values: %d %d %d %d\n",
           codes[0], codes[1], codes[2], codes[3]);

    // Decode codebook 0 (semantic) separately
    float * rvq_first = (float *)malloc(seq_len * CODEBOOK_DIM * sizeof(float));
    float * rvq_rest = (float *)malloc(seq_len * CODEBOOK_DIM * sizeof(float));
    rvq_decode_first(rvq_first, codes, codebooks_f32, seq_len, codebook_size, CODEBOOK_DIM);
    rvq_decode_rest(rvq_rest, codes, codebooks_f32, seq_len, codebook_size, CODEBOOK_DIM);

    // Debug: check RVQ outputs
    float rvq_first_min = rvq_first[0], rvq_first_max = rvq_first[0];
    float rvq_rest_min = rvq_rest[0], rvq_rest_max = rvq_rest[0];
    for (int64_t i = 0; i < seq_len * CODEBOOK_DIM; i++) {
        if (rvq_first[i] < rvq_first_min) rvq_first_min = rvq_first[i];
        if (rvq_first[i] > rvq_first_max) rvq_first_max = rvq_first[i];
        if (rvq_rest[i] < rvq_rest_min) rvq_rest_min = rvq_rest[i];
        if (rvq_rest[i] > rvq_rest_max) rvq_rest_max = rvq_rest[i];
    }
    printf("  vocoder: RVQ first (codebook 0) range: [%.4f, %.4f]\n", rvq_first_min, rvq_first_max);
    printf("  vocoder: RVQ rest (codebooks 1-15) range: [%.4f, %.4f]\n", rvq_rest_min, rvq_rest_max);
    fflush(stdout);

    // Step 2: Apply output projections to SEPARATE embeddings, then ADD (not concat!)
    float * first_proj_w = convert_f16_to_f32(weights->rvq_first_output_proj);
    float * rest_proj_w = convert_f16_to_f32(weights->rvq_rest_output_proj);

    // Project codebook 0 embeddings → 512-dim
    conv1d_k1(proj_first, rvq_first, first_proj_w, seq_len, CODEBOOK_DIM, PROJ_DIM);

    // Project codebooks 1-15 embeddings (summed) → 512-dim
    conv1d_k1(proj_rest, rvq_rest, rest_proj_w, seq_len, CODEBOOK_DIM, PROJ_DIM);

    free(rvq_first);
    free(rvq_rest);
    free(first_proj_w);
    free(rest_proj_w);

    // ADD the two projections (not concatenate!) → 512-dim
    // This matches Python: quantized = rvq_first.decode() + rvq_rest.decode()
    float * rvq_output = (float *)malloc(seq_len * PROJ_DIM * sizeof(float));
    for (int64_t i = 0; i < seq_len * PROJ_DIM; i++) {
        rvq_output[i] = proj_first[i] + proj_rest[i];
    }

    // Debug
    float rvq_out_min = rvq_output[0], rvq_out_max = rvq_output[0];
    for (int64_t i = 0; i < seq_len * PROJ_DIM; i++) {
        if (rvq_output[i] < rvq_out_min) rvq_out_min = rvq_output[i];
        if (rvq_output[i] > rvq_out_max) rvq_out_max = rvq_output[i];
    }
    printf("  vocoder: RVQ output (ADD first+rest) range: [%.4f, %.4f]\n", rvq_out_min, rvq_out_max);
    fflush(stdout);

    // Step 3: Apply pre_conv (512 → 1024, kernel=3) if available
    // This layer exists in the Python model but may not be in older GGUF files
    float * pre_conv_out = nullptr;
    if (weights->pre_conv_weight) {
        pre_conv_out = (float *)malloc(seq_len * CONCAT_DIM * sizeof(float));
        float * pre_conv_w = convert_f16_to_f32(weights->pre_conv_weight);
        float * pre_conv_b = convert_f16_to_f32(weights->pre_conv_bias);

        // pre_conv is CausalConvNet with kernel=3
        // PyTorch weight shape: [out_channels=1024, in_channels=512, kernel=3]
        int kernel_size = 3;
        int64_t out_dim = CONCAT_DIM;  // 1024
        int64_t in_dim = PROJ_DIM;     // 512

        // Causal convolution (only look back, not forward)
        for (int64_t t = 0; t < seq_len; t++) {
            for (int64_t o = 0; o < out_dim; o++) {
                float sum = pre_conv_b ? pre_conv_b[o] : 0.0f;
                for (int k = 0; k < kernel_size; k++) {
                    int64_t t_in = t - (kernel_size - 1) + k;  // Causal: look back
                    if (t_in >= 0 && t_in < seq_len) {
                        for (int64_t i = 0; i < in_dim; i++) {
                            // PyTorch layout: [out, in, kernel]
                            sum += rvq_output[t_in * in_dim + i] * pre_conv_w[(o * in_dim + i) * kernel_size + k];
                        }
                    }
                }
                pre_conv_out[t * out_dim + o] = sum;
            }
        }

        free(pre_conv_w);
        free(pre_conv_b);

        // Debug
        float pc_min = pre_conv_out[0], pc_max = pre_conv_out[0];
        for (int64_t i = 0; i < seq_len * CONCAT_DIM; i++) {
            if (pre_conv_out[i] < pc_min) pc_min = pre_conv_out[i];
            if (pre_conv_out[i] > pc_max) pc_max = pre_conv_out[i];
        }
        printf("  vocoder: pre_conv output range: [%.4f, %.4f]\n", pc_min, pc_max);
        fflush(stdout);
    } else {
        // Fallback: concatenate first and rest (old behavior, but incorrect)
        printf("  vocoder: WARNING - pre_conv not available, using concatenation fallback\n");
        pre_conv_out = (float *)malloc(seq_len * CONCAT_DIM * sizeof(float));
        for (int64_t t = 0; t < seq_len; t++) {
            memcpy(pre_conv_out + t * CONCAT_DIM, proj_first + t * PROJ_DIM, PROJ_DIM * sizeof(float));
            memcpy(pre_conv_out + t * CONCAT_DIM + PROJ_DIM, proj_rest + t * PROJ_DIM, PROJ_DIM * sizeof(float));
        }
    }
    free(rvq_output);

    // Step 4: Pre-transformer input projection (1024 -> 512)
    float * in_proj_w = convert_f16_to_f32(weights->pre_transformer_input_proj_weight);
    float * in_proj_b = convert_f16_to_f32(weights->pre_transformer_input_proj_bias);
    linear(pre_trans_in, pre_conv_out, in_proj_w, in_proj_b, seq_len, CONCAT_DIM, PROJ_DIM);
    free(in_proj_w);
    free(in_proj_b);
    free(pre_conv_out);

    // Debug
    float pt_min = pre_trans_in[0], pt_max = pre_trans_in[0];
    for (int64_t i = 0; i < seq_len * PROJ_DIM; i++) {
        if (pre_trans_in[i] < pt_min) pt_min = pre_trans_in[i];
        if (pre_trans_in[i] > pt_max) pt_max = pre_trans_in[i];
    }
    printf("  vocoder: pre_trans_in range: [%.4f, %.4f]\n", pt_min, pt_max);
    fflush(stdout);

    // Step 4: 8 pre-transformer layers (convert F16 weights to F32)
    PreTransformerLayerF32 layers_f32[PRE_TRANSFORMER_LAYERS];
    for (int i = 0; i < PRE_TRANSFORMER_LAYERS; i++) {
        layers_f32[i] = convert_pre_transformer_layer(&weights->pre_transformer_layers[i]);
    }

    memcpy(pre_trans_out, pre_trans_in, seq_len * PROJ_DIM * sizeof(float));
    for (int layer = 0; layer < PRE_TRANSFORMER_LAYERS; layer++) {
        float * layer_in = (layer == 0) ? pre_trans_in : pre_trans_out;
        float * layer_out = pre_trans_out;

        if (layer > 0) {
            memcpy(layer_in, layer_out, seq_len * PROJ_DIM * sizeof(float));
        }

        pre_transformer_layer(
            layer_out, layer_in,
            &layers_f32[layer],
            norm_buf, attn_out, ffn_out,
            q_buf, k_buf, v_buf, attn_scores,
            gate_buf, up_buf,
            seq_len, PROJ_DIM, intermediate_dim
        );
    }

    // Free converted layer weights
    for (int i = 0; i < PRE_TRANSFORMER_LAYERS; i++) {
        free_pre_transformer_layer_f32(&layers_f32[i]);
    }

    // Debug: check pre-transformer output
    float pto_min = pre_trans_out[0], pto_max = pre_trans_out[0];
    for (int64_t i = 0; i < seq_len * PROJ_DIM; i++) {
        if (pre_trans_out[i] < pto_min) pto_min = pre_trans_out[i];
        if (pre_trans_out[i] > pto_max) pto_max = pre_trans_out[i];
    }
    printf("  vocoder: pre_trans_out range: [%.4f, %.4f]\n", pto_min, pto_max);
    fflush(stdout);

    // Step 5: Pre-transformer output projection (512 -> 1024)
    float * out_proj_w = convert_f16_to_f32(weights->pre_transformer_output_proj_weight);
    float * out_proj_b = convert_f16_to_f32(weights->pre_transformer_output_proj_bias);
    linear(post_trans, pre_trans_out, out_proj_w, out_proj_b, seq_len, PROJ_DIM, CONCAT_DIM);
    free(out_proj_w);
    free(out_proj_b);

    // Debug: check post-transformer output
    float pst_min = post_trans[0], pst_max = post_trans[0];
    for (int64_t i = 0; i < seq_len * CONCAT_DIM; i++) {
        if (post_trans[i] < pst_min) pst_min = post_trans[i];
        if (post_trans[i] > pst_max) pst_max = post_trans[i];
    }
    printf("  vocoder: post_trans range: [%.4f, %.4f]\n", pst_min, pst_max);
    fflush(stdout);

    // Step 6: Upsample ConvNeXt blocks (2 stages, each with stride=2)
    // This upsamples the sequence by 4x (2*2) before the decoder
    int64_t convnext_len = seq_len;
    float * convnext_in = post_trans;  // Start with post_trans output (1024-dim)
    float * convnext_out = nullptr;

    if (weights->upsample_convnext[0].transconv_weight) {
        printf("  vocoder: applying upsample ConvNeXt blocks (2 stages)\n");

        // Allocate buffers for ConvNeXt processing
        // After 2 stages: seq_len * 4 * 1024
        const int64_t max_convnext_size = seq_len * 4 * CONCAT_DIM;
        float * cn_buf1 = (float *)malloc(max_convnext_size * sizeof(float));
        float * cn_buf2 = (float *)malloc(max_convnext_size * sizeof(float));

        if (cn_buf1 && cn_buf2) {
            convnext_out = cn_buf1;
            memcpy(cn_buf1, post_trans, seq_len * CONCAT_DIM * sizeof(float));

            for (int stage = 0; stage < 2; stage++) {
                const model::VocoderConvNeXtBlock * block = &weights->upsample_convnext[stage];
                int stride = 2;
                int64_t out_len = convnext_len * stride;

                // Step 6.1: Transposed convolution for upsampling (stride=2, kernel=4)
                float * tc_w = convert_f16_to_f32(block->transconv_weight);
                float * tc_b = convert_f16_to_f32(block->transconv_bias);
                int kernel_size = (int)block->transconv_weight->ne[0];

                // Transposed conv: input [T, 1024] -> output [T*2, 1024]
                // PyTorch ConvTranspose1d: weight [in_ch, out_ch, kernel]
                int64_t in_ch = CONCAT_DIM;
                int64_t out_ch = CONCAT_DIM;

                // Initialize output with bias
                for (int64_t t = 0; t < out_len; t++) {
                    for (int64_t c = 0; c < out_ch; c++) {
                        cn_buf2[t * out_ch + c] = tc_b ? tc_b[c] : 0.0f;
                    }
                }

                // Transposed convolution
                for (int64_t t_in = 0; t_in < convnext_len; t_in++) {
                    for (int k = 0; k < kernel_size; k++) {
                        int64_t t_out = t_in * stride + k;
                        if (t_out >= out_len) continue;

                        for (int64_t i = 0; i < in_ch; i++) {
                            float in_val = cn_buf1[t_in * in_ch + i];
                            for (int64_t o = 0; o < out_ch; o++) {
                                // PyTorch ConvTranspose1d: [in, out, kernel]
                                int64_t w_idx = (i * out_ch + o) * kernel_size + k;
                                cn_buf2[t_out * out_ch + o] += in_val * tc_w[w_idx];
                            }
                        }
                    }
                }

                free(tc_w);
                if (tc_b) free(tc_b);

                // Trim output (similar to Python trimming)
                int pad = kernel_size - stride;
                int left_pad = (pad + 1) / 2;
                int right_pad = pad - left_pad;
                int64_t trimmed_len = out_len - left_pad - right_pad;
                if (trimmed_len > 0 && left_pad > 0) {
                    memmove(cn_buf2, cn_buf2 + left_pad * out_ch, trimmed_len * out_ch * sizeof(float));
                }
                out_len = trimmed_len;

                // Step 6.2: ConvNeXt block (depthwise conv -> layernorm -> pointwise)
                // This is optional - only apply if we have the weights
                if (block->dwconv_weight) {
                    // TODO: Implement full ConvNeXt block
                    // For now, just pass through
                    printf("  vocoder: ConvNeXt stage %d - TODO: implement full ConvNeXt block\n", stage);
                }

                // Swap buffers
                float * temp = cn_buf1;
                cn_buf1 = cn_buf2;
                cn_buf2 = temp;
                convnext_len = out_len;

                printf("  vocoder: ConvNeXt stage %d output len: %lld\n", stage, (long long)convnext_len);
            }

            convnext_out = cn_buf1;
            free(cn_buf2);
        } else {
            if (cn_buf1) free(cn_buf1);
            if (cn_buf2) free(cn_buf2);
            printf("  vocoder: failed to allocate ConvNeXt buffers, skipping\n");
            convnext_out = post_trans;
        }
    } else {
        printf("  vocoder: upsample ConvNeXt blocks not loaded, skipping\n");
        convnext_out = post_trans;
    }

    // Update seq_len for subsequent stages if ConvNeXt was applied
    int64_t decoder_seq_len = convnext_len;
    float * decoder_input = convnext_out;

    // Step 7: Causal convolution (1024 -> 1536)
    // Need to reallocate causal_out if ConvNeXt changed the sequence length
    if (decoder_seq_len != seq_len) {
        free(causal_out);
        causal_out = (float *)malloc(decoder_seq_len * CAUSAL_CONV_OUT * sizeof(float));
    }
    float * causal_w = convert_f16_to_f32(weights->causal_conv_weight);
    float * causal_b = convert_f16_to_f32(weights->causal_conv_bias);
    causal_conv1d(causal_out, decoder_input, causal_w, causal_b, decoder_seq_len, CONCAT_DIM, CAUSAL_CONV_OUT, 7);
    free(causal_w);
    free(causal_b);

    // Debug
    float co_min = causal_out[0], co_max = causal_out[0];
    for (int64_t i = 0; i < decoder_seq_len * CAUSAL_CONV_OUT; i++) {
        if (causal_out[i] < co_min) co_min = causal_out[i];
        if (causal_out[i] > co_max) co_max = causal_out[i];
    }
    printf("  vocoder: causal_out range: [%.4f, %.4f]\n", co_min, co_max);
    fflush(stdout);

    // Step 8: Upsample stages (decoder blocks with ResBlocks)
    int64_t current_len = decoder_seq_len;
    int64_t current_channels = CAUSAL_CONV_OUT;
    memcpy(upsample_in, causal_out, current_len * current_channels * sizeof(float));

    const int kernel_sizes[] = {16, 10, 8, 6};
    const int out_channels[] = {768, 384, 192, 96};

    for (int stage = 0; stage < NUM_UPSAMPLE_STAGES; stage++) {
        int stride = UPSAMPLE_RATES[stage];
        int64_t out_len = current_len * stride;
        int64_t out_ch = out_channels[stage];

        // Check weight tensors
        if (!weights->upsample_alphas[stage] || !weights->upsample_betas[stage] ||
            !weights->upsample_weights[stage] || !weights->upsample_biases[stage]) {
            fprintf(stderr, "  vocoder: upsample weights null at stage %d!\n", stage);
            return;
        }

        // Debug: print weight info for stage 0
        if (stage == 0) {
            printf("  vocoder: upsample stage 0 weights shape: [%lld, %lld, %lld]\n",
                   (long long)weights->upsample_weights[0]->ne[0],
                   (long long)weights->upsample_weights[0]->ne[1],
                   (long long)weights->upsample_weights[0]->ne[2]);
            printf("  vocoder: upsample stage 0 alpha shape: [%lld], beta shape: [%lld]\n",
                   (long long)weights->upsample_alphas[0]->ne[0],
                   (long long)weights->upsample_betas[0]->ne[0]);
        }

        // Apply SnakeBeta activation before upsampling (convert F16 to F32 if needed)
        float * alpha_f32 = convert_f16_to_f32(weights->upsample_alphas[stage]);
        float * beta_f32 = convert_f16_to_f32(weights->upsample_betas[stage]);

        // Debug: print alpha/beta values for stage 0
        if (stage == 0 && alpha_f32 && beta_f32) {
            float a_min = alpha_f32[0], a_max = alpha_f32[0];
            float b_min = beta_f32[0], b_max = beta_f32[0];
            for (int64_t i = 0; i < current_channels; i++) {
                if (alpha_f32[i] < a_min) a_min = alpha_f32[i];
                if (alpha_f32[i] > a_max) a_max = alpha_f32[i];
                if (beta_f32[i] < b_min) b_min = beta_f32[i];
                if (beta_f32[i] > b_max) b_max = beta_f32[i];
            }
            printf("  vocoder: stage 0 alpha (log) range: [%f, %f], beta (log) range: [%f, %f]\n",
                   a_min, a_max, b_min, b_max);
        }

        snake_beta(upsample_out, upsample_in, alpha_f32, beta_f32, current_len, current_channels);

        // Debug: check after snake_beta for stage 0
        if (stage == 0) {
            float sb_min = upsample_out[0], sb_max = upsample_out[0];
            for (int64_t i = 0; i < current_len * current_channels && i < 10000; i++) {
                if (upsample_out[i] < sb_min) sb_min = upsample_out[i];
                if (upsample_out[i] > sb_max) sb_max = upsample_out[i];
            }
            printf("  vocoder: stage 0 after snake_beta: [%f, %f]\n", sb_min, sb_max);
        }

        free(alpha_f32);
        free(beta_f32);

        // Convert F16 weights to F32 if needed
        float * weight_f32 = convert_f16_to_f32(weights->upsample_weights[stage]);
        float * bias_f32 = convert_f16_to_f32(weights->upsample_biases[stage]);

        if (!weight_f32) {
            fprintf(stderr, "  vocoder: failed to convert upsample_weights[%d] to F32\n", stage);
            break;
        }

        // Debug: print weight range for stage 0
        if (stage == 0) {
            int64_t w_size = weights->upsample_weights[0]->ne[0] * weights->upsample_weights[0]->ne[1] * weights->upsample_weights[0]->ne[2];
            float w_min = weight_f32[0], w_max = weight_f32[0];
            for (int64_t i = 0; i < w_size; i++) {
                if (weight_f32[i] < w_min) w_min = weight_f32[i];
                if (weight_f32[i] > w_max) w_max = weight_f32[i];
            }
            printf("  vocoder: stage 0 transposed conv weight range: [%f, %f]\n", w_min, w_max);
        }

        // Transposed convolution for upsampling
        conv1d_transpose(upsample_in, upsample_out,
                         weight_f32,
                         bias_f32,
                         current_len, current_channels, out_ch,
                         kernel_sizes[stage], stride, max_upsample_size);

        // Free converted weights
        free(weight_f32);
        if (bias_f32) free(bias_f32);

        // Debug: check after transposed conv
        float tc_min = upsample_in[0], tc_max = upsample_in[0];
        for (int64_t i = 0; i < out_len * out_ch && i < 10000; i++) {
            if (upsample_in[i] < tc_min) tc_min = upsample_in[i];
            if (upsample_in[i] > tc_max) tc_max = upsample_in[i];
        }
        printf("  vocoder: stage %d after transp_conv: [%.4f, %.4f]\n", stage, tc_min, tc_max);
        fflush(stdout);

        // Apply 3 ResBlocks (if weights are available)
        const model::VocoderUpsampleStage * upsample_stage = &weights->upsample_stages[stage];
        bool has_resblocks = (upsample_stage->resblocks[0].act1_alpha != nullptr);

        if (has_resblocks) {
            // Allocate ResBlock buffers (reuse upsample_out for intermediate)
            float * rb_act_buf = (float *)malloc(out_len * out_ch * sizeof(float));
            float * rb_conv_buf = (float *)malloc(out_len * out_ch * sizeof(float));

            if (rb_act_buf && rb_conv_buf) {
                for (int rb = 0; rb < 3; rb++) {
                    const model::VocoderResBlock * resblock = &upsample_stage->resblocks[rb];
                    if (resblock->act1_alpha) {
                        // Input is in upsample_in, output to upsample_out
                        resblock_forward(upsample_out, upsample_in, resblock,
                                        rb_act_buf, rb_conv_buf, out_len, out_ch);
                        // Copy back to upsample_in for next iteration
                        memcpy(upsample_in, upsample_out, out_len * out_ch * sizeof(float));
                    }
                }

                // Debug: check after ResBlocks
                float rb_min = upsample_in[0], rb_max = upsample_in[0];
                for (int64_t i = 0; i < out_len * out_ch && i < 10000; i++) {
                    if (upsample_in[i] < rb_min) rb_min = upsample_in[i];
                    if (upsample_in[i] > rb_max) rb_max = upsample_in[i];
                }
                printf("  vocoder: stage %d after ResBlocks: [%.4f, %.4f]\n", stage, rb_min, rb_max);
                fflush(stdout);

                free(rb_act_buf);
                free(rb_conv_buf);
            } else {
                fprintf(stderr, "  vocoder: failed to allocate ResBlock buffers for stage %d\n", stage);
            }
        } else {
            printf("  vocoder: stage %d ResBlocks not loaded, skipping\n", stage);
        }

        // Debug: check upsample stage output
        float us_min = upsample_in[0], us_max = upsample_in[0];
        for (int64_t i = 0; i < out_len * out_ch && i < 10000; i++) {
            if (upsample_in[i] < us_min) us_min = upsample_in[i];
            if (upsample_in[i] > us_max) us_max = upsample_in[i];
        }
        printf("  vocoder: upsample stage %d output range: [%.4f, %.4f] (len=%lld, ch=%lld)\n",
               stage, us_min, us_max, (long long)out_len, (long long)out_ch);
        fflush(stdout);

        current_len = out_len;
        current_channels = out_ch;
    }

    // Step 8: Final SnakeBeta activation (convert F16 to F32 if needed)
    float * final_alpha_f32 = convert_f16_to_f32(weights->final_snake_alpha);
    float * final_beta_f32 = convert_f16_to_f32(weights->final_snake_beta);
    snake_beta(upsample_out, upsample_in, final_alpha_f32, final_beta_f32, current_len, current_channels);
    free(final_alpha_f32);
    free(final_beta_f32);

    // Debug: check final snake output
    float fs_min = upsample_out[0], fs_max = upsample_out[0];
    for (int64_t i = 0; i < current_len * current_channels && i < 10000; i++) {
        if (upsample_out[i] < fs_min) fs_min = upsample_out[i];
        if (upsample_out[i] > fs_max) fs_max = upsample_out[i];
    }
    printf("  vocoder: final_snake output range: [%.4f, %.4f]\n", fs_min, fs_max);
    fflush(stdout);

    // Step 9: Final convolution (96 -> 1)
    float * final_weight_f32 = convert_f16_to_f32(weights->final_conv_weight);
    float * final_bias_f32 = convert_f16_to_f32(weights->final_conv_bias);
    const float * final_weight = final_weight_f32;
    const float * final_bias = final_bias_f32;

    // Debug: print final conv weight info
    printf("  vocoder: final_conv_weight shape: [%lld, %lld, %lld], bias=%f\n",
           (long long)weights->final_conv_weight->ne[0],
           (long long)weights->final_conv_weight->ne[1],
           (long long)weights->final_conv_weight->ne[2],
           final_bias[0]);
    // Print a few weight samples
    float fw_min = final_weight[0], fw_max = final_weight[0];
    int64_t final_weight_size = weights->final_conv_weight->ne[0] * weights->final_conv_weight->ne[1] * weights->final_conv_weight->ne[2];
    for (int64_t i = 0; i < final_weight_size; i++) {
        if (final_weight[i] < fw_min) fw_min = final_weight[i];
        if (final_weight[i] > fw_max) fw_max = final_weight[i];
    }
    printf("  vocoder: final_conv_weight range: [%f, %f]\n", fw_min, fw_max);
    fflush(stdout);

    // GGML weight shape: [7, 96, 1] = [kernel_size, in_channels, out_channels]
    // Memory layout for (k, c, o): index = o * ne[0] * ne[1] + c * ne[0] + k
    // For out_channel=0: index = c * 7 + k
    int64_t kernel_size = weights->final_conv_weight->ne[0];  // 7

    for (int64_t t = 0; t < current_len; t++) {
        float sum = final_bias[0];
        // Simple conv with kernel_size=7, centered
        for (int64_t k = 0; k < kernel_size; k++) {
            int64_t t_in = t - 3 + k;  // Centered padding
            if (t_in >= 0 && t_in < current_len) {
                for (int64_t c = 0; c < (int64_t)current_channels; c++) {
                    // weight index for (k, c, out=0): c * kernel_size + k
                    sum += upsample_out[t_in * current_channels + c] * final_weight[c * kernel_size + k];
                }
            }
        }
        audio_out[t] = sum;
    }

    // Debug: check audio output range
    float audio_min = audio_out[0], audio_max = audio_out[0];
    for (int64_t t = 0; t < current_len; t++) {
        if (audio_out[t] < audio_min) audio_min = audio_out[t];
        if (audio_out[t] > audio_max) audio_max = audio_out[t];
    }
    printf("  vocoder: audio output range: [%.4f, %.4f]\n", audio_min, audio_max);
    fflush(stdout);

    // Cleanup
    free(codebooks_f32);
    free(final_weight_f32);
    free(final_bias_f32);
    free(proj_first);
    free(proj_rest);
    // Note: concat is no longer used (replaced by pre_conv_out which is freed earlier)
    free(pre_trans_in);
    free(pre_trans_out);
    // Note: post_trans might be same as convnext_out if ConvNeXt was skipped
    if (convnext_out && convnext_out != post_trans) {
        free(convnext_out);
    }
    free(post_trans);
    free(causal_out);
    free(norm_buf);
    free(attn_out);
    free(ffn_out);
    free(q_buf);
    free(k_buf);
    free(v_buf);
    free(attn_scores);
    free(gate_buf);
    free(up_buf);
    free(upsample_in);
    free(upsample_out);
}

} // namespace vocoder
} // namespace leaxer_qwen
