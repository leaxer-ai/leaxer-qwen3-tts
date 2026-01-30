// ONNX-based TTS Engine for Qwen3-TTS
// Minimal, clean implementation with control features
//
// GPU Acceleration:
//   - On Apple Silicon (M1/M2/M3), CoreML Execution Provider can be enabled
//     for GPU/Neural Engine acceleration. Build with -DLEAXER_COREML=ON (default on ARM Macs)
//   - Requires onnxruntime built with CoreML support
//   - Falls back gracefully to CPU if CoreML EP is not available

#ifndef LEAXER_QWEN_TTS_ONNX_H
#define LEAXER_QWEN_TTS_ONNX_H

#include <string>
#include <vector>
#include <memory>
#include <array>
#include <optional>

// Forward declare ONNX Runtime types
namespace Ort {
    struct Session;
    struct Env;
    struct MemoryInfo;
}

namespace leaxer_qwen {

// Model configuration constants
namespace config {
    // Architecture
    constexpr int HIDDEN_SIZE = 1024;
    constexpr int NUM_LAYERS = 28;
    constexpr int NUM_KV_HEADS = 8;
    constexpr int HEAD_DIM = 128;
    constexpr int VOCAB_SIZE = 3072;
    constexpr int NUM_CODE_GROUPS = 16;
    constexpr int SUBCODE_VOCAB_SIZE = 2048;
    
    // TTS special tokens
    constexpr int64_t TTS_BOS = 151672;
    constexpr int64_t TTS_EOS = 151673;
    constexpr int64_t TTS_PAD = 151671;
    
    // Chat tokens
    constexpr int64_t IM_START = 151644;
    constexpr int64_t IM_END = 151645;
    constexpr int64_t ASSISTANT = 77091;
    
    // Codec control tokens
    constexpr int64_t CODEC_BOS = 2149;
    constexpr int64_t CODEC_EOS = 2150;
    constexpr int64_t CODEC_PAD = 2148;
    constexpr int64_t CODEC_THINK = 2154;
    constexpr int64_t CODEC_NOTHINK = 2155;
    constexpr int64_t CODEC_THINK_BOS = 2156;
    constexpr int64_t CODEC_THINK_EOS = 2157;
    
    // Language IDs (codec tokens)
    constexpr int64_t LANG_ENGLISH = 2050;
    constexpr int64_t LANG_CHINESE = 2051;
    constexpr int64_t LANG_JAPANESE = 2052;
    constexpr int64_t LANG_KOREAN = 2053;
    
    // Defaults
    constexpr int MAX_NEW_TOKENS = 2048;
    constexpr float DEFAULT_TEMPERATURE = 0.8f;
    constexpr float DEFAULT_TOP_P = 0.95f;
    constexpr int DEFAULT_TOP_K = 50;
    constexpr int SAMPLE_RATE = 24000;
}

// Language enum for cleaner API
enum class Language {
    Auto,      // No language token (nothink mode)
    English,
    Chinese,
    Japanese,
    Korean
};

// Preset speakers (CustomVoice models)
enum class Speaker {
    None,       // No preset speaker
    Serena,     // Chinese - warm, gentle young female
    Vivian,     // Chinese - bright, edgy young female
    Uncle_Fu,   // Chinese - seasoned male, low mellow
    Dylan,      // Chinese (Beijing) - youthful male
    Eric,       // Chinese (Sichuan) - lively male, husky
    Ryan,       // English - dynamic male
    Aiden,      // English - sunny American male
    Ono_Anna,   // Japanese - playful female
    Sohee       // Korean - warm female
};

// Parse speaker name string to enum
Speaker parse_speaker(const std::string& name);

// Sampling parameters
struct SamplingParams {
    float temperature = config::DEFAULT_TEMPERATURE;
    float top_p = config::DEFAULT_TOP_P;
    int top_k = config::DEFAULT_TOP_K;
    float repetition_penalty = 1.0f;
    int max_new_tokens = config::MAX_NEW_TOKENS;
};

// KV Cache for transformer layers
struct KVCache {
    std::vector<std::vector<float>> keys;
    std::vector<std::vector<float>> values;
    int64_t seq_len = 0;
    
    KVCache() : keys(config::NUM_LAYERS), values(config::NUM_LAYERS) {}
    void clear();
};

// Main TTS Engine
class TTSEngine {
public:
    explicit TTSEngine(const std::string& model_dir);
    ~TTSEngine();
    
    // No copy
    TTSEngine(const TTSEngine&) = delete;
    TTSEngine& operator=(const TTSEngine&) = delete;
    
    // Main API
    std::vector<float> synthesize(
        const std::string& text,
        Language lang = Language::Auto,
        const SamplingParams& params = SamplingParams()
    );
    
    // Voice clone: synthesize using reference audio
    std::vector<float> synthesize_clone(
        const std::string& text,
        const std::string& ref_audio_path,
        Language lang = Language::Auto,
        const SamplingParams& params = SamplingParams()
    );
    
    // Preset speaker: synthesize using preset voice
    std::vector<float> synthesize_speaker(
        const std::string& text,
        Speaker speaker,
        Language lang = Language::Auto,
        const SamplingParams& params = SamplingParams()
    );
    
    // Low-level: synthesize from pre-tokenized input
    std::vector<float> synthesize_tokens(
        const std::vector<int64_t>& token_ids,
        Language lang = Language::Auto,
        const SamplingParams& params = SamplingParams()
    );
    
    // Extract speaker embedding from audio file
    std::vector<float> extract_speaker_embedding(const std::string& audio_path);
    
    // Check if speaker encoder is available
    bool has_speaker_encoder() const { return speaker_encoder_ != nullptr; }
    
    bool is_ready() const { return ready_; }
    const std::string& get_error() const { return error_msg_; }
    
private:
    // ONNX Runtime
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::MemoryInfo> memory_info_;
    
    // Model sessions
    std::unique_ptr<Ort::Session> text_project_;
    std::unique_ptr<Ort::Session> codec_embed_;
    std::unique_ptr<Ort::Session> code_predictor_embed_;
    std::unique_ptr<Ort::Session> talker_prefill_;
    std::unique_ptr<Ort::Session> talker_decode_;
    std::unique_ptr<Ort::Session> code_predictor_;
    std::unique_ptr<Ort::Session> vocoder_;
    std::unique_ptr<Ort::Session> speaker_encoder_;
    
    // State
    KVCache kv_cache_;
    std::vector<float> last_hidden_;
    std::vector<float> trailing_text_hidden_;
    std::vector<float> tts_pad_embed_;
    int trailing_len_ = 0;
    
    bool ready_ = false;
    std::string error_msg_;
    std::string model_dir_;
    
    // Internal methods
    std::unique_ptr<Ort::Session> load_model(const std::string& filename);
    
    // Embedding operations
    std::vector<float> run_text_project(const std::vector<int64_t>& input_ids);
    std::vector<float> run_codec_embed(int64_t codec_token);
    std::vector<float> run_codec_embed_batch(const std::vector<int64_t>& codec_tokens);
    std::vector<float> run_code_predictor_embed(int64_t subcode_token, int64_t generation_step);
    
    // Transformer operations
    std::vector<float> run_prefill(const std::vector<float>& inputs_embeds,
                                   const std::vector<int64_t>& attention_mask);
    std::vector<float> run_decode(const std::vector<float>& input_embed,
                                  const std::vector<int64_t>& attention_mask);
    std::vector<float> run_code_predictor(const std::vector<float>& inputs_embeds,
                                          int64_t generation_step);
    std::vector<float> run_vocoder(const std::vector<int64_t>& audio_codes,
                                   int64_t codes_length);
    
    // Speaker encoder
    std::vector<float> run_speaker_encoder(const std::vector<float>& mel);
    
    // Generation
    std::vector<float> build_prompt_embeddings(const std::vector<int64_t>& input_ids,
                                               Language lang,
                                               const std::vector<float>& speaker_embed = {});
    std::vector<std::array<int64_t, 16>> generate_codes(const std::vector<float>& prompt_embeds,
                                                        const SamplingParams& params);
    std::array<int64_t, 15> predict_subcodes(int64_t code0, const SamplingParams& params);
    
    // Sampling
    int64_t sample_token(const std::vector<float>& logits, const SamplingParams& params);
    static void softmax(std::vector<float>& logits);
    static void top_k_filter(std::vector<float>& logits, int k);
    static void top_p_filter(std::vector<float>& probs, float p);
};

// Helper: Convert language enum to codec token ID
inline int64_t language_to_codec_id(Language lang) {
    switch (lang) {
        case Language::English:  return config::LANG_ENGLISH;
        case Language::Chinese:  return config::LANG_CHINESE;
        case Language::Japanese: return config::LANG_JAPANESE;
        case Language::Korean:   return config::LANG_KOREAN;
        default:                 return 0;  // Auto = no language token
    }
}

// Check if CoreML support was compiled in
inline bool is_coreml_enabled() {
#if defined(LEAXER_USE_COREML) && defined(__APPLE__)
    return true;
#else
    return false;
#endif
}

} // namespace leaxer_qwen

#endif // LEAXER_QWEN_TTS_ONNX_H
