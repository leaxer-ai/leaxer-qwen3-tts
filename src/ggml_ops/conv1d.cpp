// Causal 1D Convolution
// Left-pads input to ensure output[t] only depends on input[0:t+1]

#include "ggml.h"
#include "common.h"

namespace leaxer_qwen {
namespace ops {

// Causal Conv1d: left-pads by (kernel_size - 1) to preserve causality
// Input: [seq_len, in_channels, batch] (ggml layout)
// Weight: [kernel_size, in_channels, out_channels] (ggml layout)
// Output: [seq_len, out_channels, batch]
struct ggml_tensor * conv1d_causal(
    struct ggml_context * ctx,
    struct ggml_tensor * weight,  // [kernel_size, in_channels, out_channels]
    struct ggml_tensor * input,   // [seq_len, in_channels, batch]
    int kernel_size,
    int stride = 1,
    int dilation = 1
) {
    // For causal conv: pad_left = kernel_size - 1, pad_right = 0
    int pad_left = kernel_size - 1;

    // Manually pad the input on the left (prepend zeros) using ggml_pad_ext
    // Parameters: lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3
    // Input is [seq_len, in_channels, batch], so dimension 0 is seq_len
    struct ggml_tensor * input_padded = ggml_pad_ext(ctx, input,
                                                     pad_left, 0,  // pad seq_len dimension (dim 0)
                                                     0, 0,         // no padding on in_channels (dim 1)
                                                     0, 0,         // no padding on batch (dim 2)
                                                     0, 0);        // no padding on dim 3

    // Use ggml_conv_1d with no padding since we already padded manually
    // Parameters: stride, padding=0, dilation
    struct ggml_tensor * output = ggml_conv_1d(ctx, weight, input_padded, stride, 0, dilation);

    return output;
}

} // namespace ops
} // namespace leaxer_qwen
