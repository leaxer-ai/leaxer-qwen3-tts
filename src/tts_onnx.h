// ONNX-based TTS Engine for Qwen3-TTS
// Uses ONNX Runtime for inference with pre-exported models

#ifndef LEAXER_QWEN_TTS_ONNX_H
#define LEAXER_QWEN_TTS_ONNX_H

#include <string>
#include <vector>
#include <memory>
#include <array>

// Forward declare ONNX Runtime types
namespace Ort {
    struct Session;
    struct Env;
    struct MemoryInfo;
    struct Value;
    struct SessionOptions;
}

namespace leaxer_qwen {

// Configuration constants from model config
namespace onnx_config {
    // Model architecture
    constexpr int HIDDEN_SIZE = 1024;
    constexpr int NUM_LAYERS = 28;
    constexpr int NUM_KV_HEADS = 8;
    constexpr int HEAD_DIM = 128;
    constexpr int VOCAB_SIZE = 3072;
    constexpr int NUM_CODE_GROUPS = 16;
    constexpr int SUBCODE_VOCAB_SIZE = 2048;
    
    // Special token IDs (TTS tokens)
    constexpr int64_t TTS_BOS = 151672;       // <tts_text_bos>
    constexpr int64_t TTS_EOS = 151673;       // <tts_text_eod>
    constexpr int64_t TTS_PAD = 151671;       // <tts_pad>
    
    // Chat tokens (for prompt formatting)
    constexpr int64_t IM_START = 151644;      // <|im_start|>
    constexpr int64_t IM_END = 151645;        // <|im_end|>
    
    // Codec tokens (within talker vocab of 3072)
    constexpr int64_t CODEC_BOS = 2149;
    constexpr int64_t CODEC_EOS = 2150;
    constexpr int64_t CODEC_PAD = 2148;
    constexpr int64_t CODEC_THINK = 2154;
    constexpr int64_t CODEC_NOTHINK = 2155;
    constexpr int64_t CODEC_THINK_BOS = 2156;
    constexpr int64_t CODEC_THINK_EOS = 2157;
    
    // Generation defaults
    constexpr int MAX_NEW_TOKENS = 2048;
    constexpr float DEFAULT_TEMPERATURE = 0.8f;
    constexpr float DEFAULT_TOP_P = 0.95f;
    constexpr int DEFAULT_TOP_K = 50;
    
    // Audio
    constexpr int SAMPLE_RATE = 24000;
}

// KV Cache for transformer layers
struct ONNXKVCache {
    // Shape: [batch, num_kv_heads, seq_len, head_dim]
    // For 28 layers, we need 28 key tensors and 28 value tensors
    std::vector<std::vector<float>> keys;    // [num_layers][batch * heads * seq * dim]
    std::vector<std::vector<float>> values;  // [num_layers][batch * heads * seq * dim]
    int64_t seq_len = 0;
    
    ONNXKVCache() : keys(onnx_config::NUM_LAYERS), values(onnx_config::NUM_LAYERS) {}
    
    void clear() {
        for (auto& k : keys) k.clear();
        for (auto& v : values) v.clear();
        seq_len = 0;
    }
};

// Sampling parameters for generation
struct SamplingParams {
    float temperature = onnx_config::DEFAULT_TEMPERATURE;
    float top_p = onnx_config::DEFAULT_TOP_P;
    int top_k = onnx_config::DEFAULT_TOP_K;
    float repetition_penalty = 1.0f;
    int max_new_tokens = onnx_config::MAX_NEW_TOKENS;
};

// Main TTS Engine class using ONNX models
class TTSEngine {
public:
    // Constructor: loads all ONNX models from model_dir
    // Expected files:
    //   - text_project.onnx
    //   - codec_embed.onnx
    //   - code_predictor_embed.onnx
    //   - talker_prefill.onnx
    //   - talker_decode.onnx
    //   - code_predictor.onnx
    //   - tokenizer12hz_decode.onnx
    //   - speaker_encoder.onnx (optional, for voice cloning)
    explicit TTSEngine(const std::string& model_dir);
    
    ~TTSEngine();
    
    // Disable copy
    TTSEngine(const TTSEngine&) = delete;
    TTSEngine& operator=(const TTSEngine&) = delete;
    
    // Main synthesis interface
    // text: input text to synthesize
    // params: sampling parameters
    // Returns: audio samples at 24kHz
    std::vector<float> synthesize(const std::string& text, 
                                  const SamplingParams& params = SamplingParams());
    
    // Synthesize with pre-tokenized input
    std::vector<float> synthesize_tokens(const std::vector<int64_t>& token_ids,
                                         const SamplingParams& params = SamplingParams());
    
    // Voice cloning synthesis (future)
    // std::vector<float> synthesize_clone(const std::string& text,
    //                                     const std::vector<float>& reference_audio,
    //                                     const SamplingParams& params = SamplingParams());
    
    // Check if models are loaded successfully
    bool is_ready() const { return ready_; }
    
    // Get last error message
    const std::string& get_error() const { return error_msg_; }
    
private:
    // ONNX Runtime environment (shared)
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::MemoryInfo> memory_info_;
    
    // ONNX Sessions for each model component
    std::unique_ptr<Ort::Session> text_project_;        // Text tokens → embeddings
    std::unique_ptr<Ort::Session> codec_embed_;         // Codec token (codebook 0) → embedding
    std::unique_ptr<Ort::Session> code_predictor_embed_;// Sub-codec token → embedding
    std::unique_ptr<Ort::Session> talker_prefill_;      // First pass (build KV cache)
    std::unique_ptr<Ort::Session> talker_decode_;       // Incremental decode
    std::unique_ptr<Ort::Session> code_predictor_;      // Predict sub-codes (1-15)
    std::unique_ptr<Ort::Session> vocoder_;             // Audio codes → waveform
    std::unique_ptr<Ort::Session> speaker_encoder_;     // Mel → speaker embedding (optional)
    
    // KV cache for transformer
    ONNXKVCache kv_cache_;
    
    // Last hidden state for code predictor
    std::vector<float> last_hidden_;  // [1 x 1 x 1024]
    
    // State
    bool ready_ = false;
    std::string error_msg_;
    std::string model_dir_;
    
    // ------- Internal Methods -------
    
    // Load a single ONNX model
    std::unique_ptr<Ort::Session> load_model(const std::string& filename);
    
    // Build input prompt embeddings
    // Returns: concatenated embeddings [batch x total_tokens x hidden_size]
    std::vector<float> build_prompt_embeddings(const std::vector<int64_t>& text_tokens);
    
    // Run text projection model
    // input_ids: [batch x tokens]
    // Returns: embeddings [batch x tokens x hidden_size]
    std::vector<float> run_text_project(const std::vector<int64_t>& input_ids);
    
    // Run codec embedding model (codebook 0)
    // input_ids: [batch x 1] single codec token
    // Returns: embedding [batch x 1 x hidden_size]
    std::vector<float> run_codec_embed(int64_t codec_token);
    
    // Run sub-code embedding model
    // input_ids: [batch x 1] sub-codec token
    // generation_step: which sub-codebook (0-14)
    // Returns: embedding [batch x 1 x hidden_size]
    std::vector<float> run_code_predictor_embed(int64_t subcode_token, int64_t generation_step);
    
    // Run talker prefill (first pass)
    // inputs_embeds: [batch x seq x hidden_size]
    // attention_mask: [batch x seq]
    // Updates kv_cache_ and returns logits [batch x seq x vocab_size]
    std::vector<float> run_prefill(const std::vector<float>& inputs_embeds,
                                   const std::vector<int64_t>& attention_mask);
    
    // Run talker decode (incremental)
    // input_embed: [batch x 1 x hidden_size]
    // attention_mask: [batch x total_seq] (including new token)
    // Updates kv_cache_ and returns logits [batch x 1 x vocab_size]
    std::vector<float> run_decode(const std::vector<float>& input_embed,
                                  const std::vector<int64_t>& attention_mask);
    
    // Run code predictor to get sub-codebook tokens
    // inputs_embeds: sequence of [last_hidden, first_embed, sub_embeds...]
    // generation_step: which sub-codebook (0-14)
    // Returns: logits [batch x subcode_vocab_size]
    std::vector<float> run_code_predictor(const std::vector<float>& inputs_embeds,
                                          int64_t generation_step);
    
    // Run vocoder to convert codes to audio
    // audio_codes: [batch x codes_length x 16]
    // Returns: audio samples [batch x audio_length]
    std::vector<float> run_vocoder(const std::vector<int64_t>& audio_codes,
                                   int64_t codes_length);
    
    // ------- Generation Loop -------
    
    // Main autoregressive generation loop
    // Returns: generated audio codes [steps x 16]
    std::vector<std::array<int64_t, 16>> generate_codes(
        const std::vector<float>& prompt_embeds,
        const SamplingParams& params);
    
    // Single decode step: generate one frame of 16 codes
    // Returns: [codebook0, codebook1, ..., codebook15]
    std::array<int64_t, 16> decode_step(const std::vector<float>& input_embed,
                                         std::vector<int64_t>& attention_mask,
                                         const SamplingParams& params);
    
    // Predict sub-codes (codebooks 1-15) given codebook 0
    std::array<int64_t, 15> predict_subcodes(int64_t code0, const SamplingParams& params);
    
    // ------- Sampling -------
    
    // Sample from logits with temperature, top-k, top-p
    int64_t sample_token(const std::vector<float>& logits, const SamplingParams& params);
    
    // Apply softmax to logits
    static void softmax(std::vector<float>& logits);
    
    // Apply top-k filtering
    static void top_k_filter(std::vector<float>& logits, int k);
    
    // Apply top-p (nucleus) filtering  
    static void top_p_filter(std::vector<float>& logits, float p);
};

// Utility: Simple tokenizer for testing (actual implementation needs BPE)
class SimpleTokenizer {
public:
    SimpleTokenizer(const std::string& vocab_path);
    std::vector<int64_t> encode(const std::string& text);
    std::string decode(const std::vector<int64_t>& tokens);
    
private:
    // Placeholder - real implementation needs vocab + merges
};

} // namespace leaxer_qwen

#endif // LEAXER_QWEN_TTS_ONNX_H
