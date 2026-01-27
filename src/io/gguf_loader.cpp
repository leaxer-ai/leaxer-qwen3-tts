// GGUF Model Loader
// Loads Qwen3-TTS weights from GGUF format

namespace leaxer_qwen {
namespace io {

// TODO: Implement GGUF loader
// Use ggml's GGUF reading utilities
// Map tensor names from Qwen3-TTS to internal names
//
// Key tensors to load:
// - token_embd.weight
// - blk.*.attn_q.weight, attn_k.weight, attn_v.weight, attn_output.weight
// - blk.*.ffn_gate.weight, ffn_up.weight, ffn_down.weight
// - blk.*.attn_norm.weight, ffn_norm.weight
// - output_norm.weight
// - output.weight (or lm_head)
// - Vocoder weights (RVQ codebooks, conv weights, etc.)

} // namespace io
} // namespace leaxer_qwen
