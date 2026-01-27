// Transposed Convolution Upsampling for Vocoder
// Upsample rates: 8, 5, 4, 3 (total: 480x per two stages, 1920x total)

#include "ggml.h"
#include "common.h"

namespace leaxer_qwen {
namespace vocoder {

// Upsample configuration
constexpr int UPSAMPLE_RATES[] = {8, 5, 4, 3};
constexpr int NUM_UPSAMPLE_STAGES = 4;
constexpr int TOTAL_UPSAMPLE = 8 * 5 * 4 * 3;  // 480

// Transposed Conv1d for upsampling
// Input: [seq_len, in_channels, batch] (ggml layout)
// Weight: [kernel_size, in_channels, out_channels] (ggml layout)
// Output: [seq_len * stride, out_channels, batch]
struct ggml_tensor * conv1d_transpose(
    struct ggml_context * ctx,
    struct ggml_tensor * weight,  // [kernel_size, in_channels, out_channels]
    struct ggml_tensor * input,   // [seq_len, in_channels, batch]
    int kernel_size,
    int stride,
    int padding = 0,
    int dilation = 1
) {
    // Use ggml_conv_transpose_1d
    // Parameters: kernel, data, stride, padding, dilation
    struct ggml_tensor * output = ggml_conv_transpose_1d(ctx, weight, input, stride, padding, dilation);

    return output;
}

} // namespace vocoder
} // namespace leaxer_qwen
