// Split Residual Vector Quantizer Decoder
// Reconstructs continuous latent from 16 discrete codebooks
// 1 semantic codebook + 15 acoustic codebooks
// Each codebook has 2048 entries

#include "ggml.h"
#include "common.h"
#include <cstdint>

namespace leaxer_qwen {
namespace vocoder {

// Codebook configuration
constexpr int NUM_CODEBOOKS = 16;
constexpr int NUM_SEMANTIC = 1;
constexpr int NUM_ACOUSTIC = 15;
constexpr int CODEBOOK_SIZE = 2048;
constexpr int CODEBOOK_DIM = 512;  // codebook_dim // 2

// Codebook lookup: index -> vector
// Input: codes [seq_len] - int32 indices in range [0, CODEBOOK_SIZE)
// Input: codebook [CODEBOOK_SIZE, CODEBOOK_DIM] - embedding matrix
// Output: embedded [seq_len, CODEBOOK_DIM] - looked-up vectors
void codebook_lookup(
    struct ggml_tensor * dst,
    const struct ggml_tensor * codes,
    const struct ggml_tensor * codebook) {

    GGML_ASSERT(codes->type == GGML_TYPE_I32);
    GGML_ASSERT(codebook->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int32_t * code_data = (const int32_t *)codes->data;
    const float * codebook_data = (const float *)codebook->data;
    float * dst_data = (float *)dst->data;

    const int64_t seq_len = codes->ne[0];
    const int64_t codebook_dim = codebook->ne[0];
    const int64_t codebook_size = codebook->ne[1];

    GGML_ASSERT(dst->ne[0] == codebook_dim);
    GGML_ASSERT(dst->ne[1] == seq_len);

    // For each timestep, lookup the vector from codebook
    for (int64_t t = 0; t < seq_len; t++) {
        int32_t idx = code_data[t];
        GGML_ASSERT(idx >= 0 && idx < codebook_size);

        // Copy vector from codebook[idx, :] to dst[:, t]
        for (int64_t d = 0; d < codebook_dim; d++) {
            dst_data[t * codebook_dim + d] = codebook_data[idx * codebook_dim + d];
        }
    }
}

// RVQ Decode: Reconstruct latent from all 16 codebooks
// Input: codes [NUM_CODEBOOKS, seq_len] - int32 indices for each codebook
// Input: codebooks [NUM_CODEBOOKS, CODEBOOK_SIZE, CODEBOOK_DIM] - all codebook embeddings
// Output: latent [seq_len, CODEBOOK_DIM] - summed residual vectors
void rvq_decode(
    struct ggml_tensor * dst,
    const struct ggml_tensor * codes,
    const struct ggml_tensor * codebooks) {

    GGML_ASSERT(codes->type == GGML_TYPE_I32);
    GGML_ASSERT(codebooks->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int32_t * code_data = (const int32_t *)codes->data;
    const float * codebook_data = (const float *)codebooks->data;
    float * dst_data = (float *)dst->data;

    const int64_t seq_len = codes->ne[0];
    const int64_t num_codebooks = codes->ne[1];
    const int64_t codebook_dim = codebooks->ne[0];
    const int64_t codebook_size = codebooks->ne[1];

    GGML_ASSERT(num_codebooks == NUM_CODEBOOKS);
    GGML_ASSERT(dst->ne[0] == codebook_dim);
    GGML_ASSERT(dst->ne[1] == seq_len);

    // Initialize output to zero
    const int64_t total_elements = seq_len * codebook_dim;
    for (int64_t i = 0; i < total_elements; i++) {
        dst_data[i] = 0.0f;
    }

    // Sum contributions from all codebooks
    for (int64_t cb = 0; cb < num_codebooks; cb++) {
        const int32_t * cb_codes = code_data + cb * seq_len;
        const float * cb_embedding = codebook_data + cb * codebook_size * codebook_dim;

        for (int64_t t = 0; t < seq_len; t++) {
            int32_t idx = cb_codes[t];
            GGML_ASSERT(idx >= 0 && idx < codebook_size);

            // Add residual from this codebook to output
            for (int64_t d = 0; d < codebook_dim; d++) {
                dst_data[t * codebook_dim + d] += cb_embedding[idx * codebook_dim + d];
            }
        }
    }
}

} // namespace vocoder
} // namespace leaxer_qwen
