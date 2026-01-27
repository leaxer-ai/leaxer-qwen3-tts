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

// RVQ decode: look up codebook embeddings and sum
// codes: [seq_len, 16] int32
// codebooks: [16, 2048, 256] float32
// dst: [seq_len, 256] float32 - summed embeddings
static void rvq_decode_sum(
    float * dst,
    const int32_t * codes,
    const float * codebooks,
    int64_t seq_len,
    int64_t codebook_size,
    int64_t codebook_dim
) {
    // Zero output
    memset(dst, 0, seq_len * codebook_dim * sizeof(float));

    // Sum contributions from all 16 codebooks
    for (int cb = 0; cb < NUM_CODEBOOKS; cb++) {
        const float * cb_embedding = codebooks + cb * codebook_size * codebook_dim;

        for (int64_t t = 0; t < seq_len; t++) {
            int32_t idx = codes[t * NUM_CODEBOOKS + cb];
            if (idx < 0 || idx >= codebook_size) {
                fprintf(stderr, "Warning: codebook index out of range: %d\n", idx);
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

// Single pre-transformer layer
static void pre_transformer_layer(
    float * output,
    const float * input,
    const model::PreTransformerLayer * layer,
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
    rms_norm(norm_buf, input, (const float *)layer->input_ln_weight->data, seq_len, hidden_dim);

    // Self-attention
    self_attention(
        attn_out, norm_buf,
        (const float *)layer->attn_q_weight->data,
        (const float *)layer->attn_k_weight->data,
        (const float *)layer->attn_v_weight->data,
        (const float *)layer->attn_o_weight->data,
        q_buf, k_buf, v_buf, attn_scores,
        seq_len, hidden_dim, head_dim, num_heads
    );

    // Apply attention layer scale and residual
    const float * attn_scale = (const float *)layer->attn_scale->data;
    for (int64_t t = 0; t < seq_len; t++) {
        for (int64_t d = 0; d < hidden_dim; d++) {
            output[t * hidden_dim + d] = input[t * hidden_dim + d] + attn_out[t * hidden_dim + d] * attn_scale[d];
        }
    }

    // Post-attention layer norm
    rms_norm(norm_buf, output, (const float *)layer->post_ln_weight->data, seq_len, hidden_dim);

    // FFN
    swiglu_ffn(
        ffn_out, norm_buf,
        (const float *)layer->ffn_gate_weight->data,
        (const float *)layer->ffn_up_weight->data,
        (const float *)layer->ffn_down_weight->data,
        gate_buf, up_buf,
        seq_len, hidden_dim, intermediate_dim
    );

    // Apply FFN layer scale and residual
    const float * ffn_scale = (const float *)layer->ffn_scale->data;
    for (int64_t t = 0; t < seq_len; t++) {
        for (int64_t d = 0; d < hidden_dim; d++) {
            output[t * hidden_dim + d] += ffn_out[t * hidden_dim + d] * ffn_scale[d];
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
    float * rvq_sum = (float *)malloc(seq_len * CODEBOOK_DIM * sizeof(float));
    float * proj_first = (float *)malloc(seq_len * PROJ_DIM * sizeof(float));
    float * proj_rest = (float *)malloc(seq_len * PROJ_DIM * sizeof(float));
    float * concat = (float *)malloc(seq_len * CONCAT_DIM * sizeof(float));
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
    // After each stage: len*rate, channels/2
    // Stage 0: 2040*8=16320 * 768 = 12.5M
    // Stage 1: 16320*5=81600 * 384 = 31.3M
    // Stage 2: 81600*4=326400 * 192 = 62.7M
    // Stage 3: 326400*3=979200 * 96 = 94M
    // Max is stage 3 output: 979200 * 96
    const int64_t max_upsample_size = max_seq * 96;  // Final stage: audio_len * 96 channels
    float * upsample_in = (float *)malloc(max_upsample_size * sizeof(float));
    float * upsample_out = (float *)malloc(max_upsample_size * sizeof(float));

    if (!upsample_in || !upsample_out) {
        fprintf(stderr, "vocoder: failed to allocate upsample buffers!\n");
        return;
    }

    // Step 1: RVQ decode - sum all 16 codebook embeddings
    const float * codebooks = (const float *)weights->codebooks->data;
    int64_t codebook_size = weights->codebooks->ne[1];
    rvq_decode_sum(rvq_sum, codes, codebooks, seq_len, codebook_size, CODEBOOK_DIM);

    // Step 2: Apply output projections
    // First codebook (semantic)
    conv1d_k1(proj_first, rvq_sum,
              (const float *)weights->rvq_first_output_proj->data,
              seq_len, CODEBOOK_DIM, PROJ_DIM);

    // Rest codebooks (acoustic) - they share the same RVQ sum for now
    // In proper implementation, we'd separate first vs rest codebook sums
    conv1d_k1(proj_rest, rvq_sum,
              (const float *)weights->rvq_rest_output_proj->data,
              seq_len, CODEBOOK_DIM, PROJ_DIM);

    // Concatenate: [seq_len, 512] + [seq_len, 512] -> [seq_len, 1024]
    for (int64_t t = 0; t < seq_len; t++) {
        memcpy(concat + t * CONCAT_DIM, proj_first + t * PROJ_DIM, PROJ_DIM * sizeof(float));
        memcpy(concat + t * CONCAT_DIM + PROJ_DIM, proj_rest + t * PROJ_DIM, PROJ_DIM * sizeof(float));
    }

    // Step 3: Pre-transformer input projection (1024 -> 512)
    linear(pre_trans_in, concat,
           (const float *)weights->pre_transformer_input_proj_weight->data,
           (const float *)weights->pre_transformer_input_proj_bias->data,
           seq_len, CONCAT_DIM, PROJ_DIM);

    // Step 4: 8 pre-transformer layers
    memcpy(pre_trans_out, pre_trans_in, seq_len * PROJ_DIM * sizeof(float));
    for (int layer = 0; layer < PRE_TRANSFORMER_LAYERS; layer++) {
        float * layer_in = (layer == 0) ? pre_trans_in : pre_trans_out;
        float * layer_out = pre_trans_out;

        if (layer > 0) {
            memcpy(layer_in, layer_out, seq_len * PROJ_DIM * sizeof(float));
        }

        pre_transformer_layer(
            layer_out, layer_in,
            &weights->pre_transformer_layers[layer],
            norm_buf, attn_out, ffn_out,
            q_buf, k_buf, v_buf, attn_scores,
            gate_buf, up_buf,
            seq_len, PROJ_DIM, intermediate_dim
        );
    }

    // Step 5: Pre-transformer output projection (512 -> 1024)
    linear(post_trans, pre_trans_out,
           (const float *)weights->pre_transformer_output_proj_weight->data,
           (const float *)weights->pre_transformer_output_proj_bias->data,
           seq_len, PROJ_DIM, CONCAT_DIM);

    // Step 6: Causal convolution (1024 -> 1536)
    causal_conv1d(causal_out, post_trans,
                  (const float *)weights->causal_conv_weight->data,
                  (const float *)weights->causal_conv_bias->data,
                  seq_len, CONCAT_DIM, CAUSAL_CONV_OUT, 7);

    // Step 7: Upsample stages
    int64_t current_len = seq_len;
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

        // Apply SnakeBeta activation before upsampling (convert F16 to F32 if needed)
        float * alpha_f32 = convert_f16_to_f32(weights->upsample_alphas[stage]);
        float * beta_f32 = convert_f16_to_f32(weights->upsample_betas[stage]);
        snake_beta(upsample_out, upsample_in, alpha_f32, beta_f32, current_len, current_channels);
        free(alpha_f32);
        free(beta_f32);

        // Convert F16 weights to F32 if needed
        float * weight_f32 = convert_f16_to_f32(weights->upsample_weights[stage]);
        float * bias_f32 = convert_f16_to_f32(weights->upsample_biases[stage]);

        if (!weight_f32) {
            fprintf(stderr, "  vocoder: failed to convert upsample_weights[%d] to F32\n", stage);
            break;
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

        current_len = out_len;
        current_channels = out_ch;
    }

    // Step 8: Final SnakeBeta activation (convert F16 to F32 if needed)
    float * final_alpha_f32 = convert_f16_to_f32(weights->final_snake_alpha);
    float * final_beta_f32 = convert_f16_to_f32(weights->final_snake_beta);
    snake_beta(upsample_out, upsample_in, final_alpha_f32, final_beta_f32, current_len, current_channels);
    free(final_alpha_f32);
    free(final_beta_f32);

    // Step 9: Final convolution (96 -> 1)
    float * final_weight_f32 = convert_f16_to_f32(weights->final_conv_weight);
    float * final_bias_f32 = convert_f16_to_f32(weights->final_conv_bias);
    const float * final_weight = final_weight_f32;
    const float * final_bias = final_bias_f32;

    for (int64_t t = 0; t < current_len; t++) {
        float sum = final_bias[0];
        // Simple conv with kernel_size=7, centered
        for (int k = 0; k < 7; k++) {
            int64_t t_in = t - 3 + k;  // Centered padding
            if (t_in >= 0 && t_in < current_len) {
                for (int64_t c = 0; c < current_channels; c++) {
                    sum += upsample_out[t_in * current_channels + c] * final_weight[c * 7 + k];
                }
            }
        }
        audio_out[t] = sum;
    }

    // Cleanup
    free(final_weight_f32);
    free(final_bias_f32);
    free(rvq_sum);
    free(proj_first);
    free(proj_rest);
    free(concat);
    free(pre_trans_in);
    free(pre_trans_out);
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
