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

} // namespace vocoder
} // namespace leaxer_qwen
