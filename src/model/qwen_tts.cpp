// Full Qwen3-TTS Model
// Integrates: Tokenizer → LLM → Code Predictor → Vocoder

namespace leaxer_qwen {
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
constexpr int CODEC_PAD_ID = 4196;
constexpr int CODEC_BOS_ID = 4197;
constexpr int CODEC_EOS_ID = 4198;

// TODO: Implement full TTS model
// Pipeline:
// 1. Format prompt: "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
// 2. Tokenize text
// 3. Run through LLM to generate codec tokens
// 4. Refine with code predictor
// 5. Decode with vocoder

} // namespace model
} // namespace leaxer_qwen
