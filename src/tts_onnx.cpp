// ONNX-based TTS Engine Implementation
// Clean implementation with language control

#include "tts_onnx.h"
#include "io/tokenizer.h"
#include "io/wav_reader.h"
#include "io/mel.h"
#include <onnxruntime/onnxruntime_cxx_api.h>
#if defined(LEAXER_USE_COREML) && defined(__APPLE__)
#include <onnxruntime/coreml_provider_factory.h>
#endif

#include <algorithm>
#include <cmath>
#include <random>
#include <numeric>
#include <iostream>
#include <filesystem>
#include <unordered_map>

// GPU Execution Provider support
#if defined(LEAXER_USE_COREML) && defined(__APPLE__)
    #define LEAXER_TRY_COREML 1
#else
    #define LEAXER_TRY_COREML 0
#endif

#ifdef LEAXER_USE_CUDA
    #define LEAXER_TRY_CUDA 1
#else
    #define LEAXER_TRY_CUDA 0
#endif

#ifdef LEAXER_USE_ROCM
    #define LEAXER_TRY_ROCM 1
#else
    #define LEAXER_TRY_ROCM 0
#endif

#ifdef LEAXER_USE_DIRECTML
    #define LEAXER_TRY_DIRECTML 1
#else
    #define LEAXER_TRY_DIRECTML 0
#endif

namespace leaxer_qwen {

namespace fs = std::filesystem;

// ===========================================================================
// Speaker parsing
// ===========================================================================

Speaker parse_speaker(const std::string& name) {
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    if (lower == "serena") return Speaker::Serena;
    if (lower == "vivian") return Speaker::Vivian;
    if (lower == "uncle_fu" || lower == "unclefu") return Speaker::Uncle_Fu;
    if (lower == "dylan") return Speaker::Dylan;
    if (lower == "eric") return Speaker::Eric;
    if (lower == "ryan") return Speaker::Ryan;
    if (lower == "aiden") return Speaker::Aiden;
    if (lower == "ono_anna" || lower == "onoanna") return Speaker::Ono_Anna;
    if (lower == "sohee") return Speaker::Sohee;
    return Speaker::None;
}

// ===========================================================================
// KVCache
// ===========================================================================

void KVCache::clear() {
    for (auto& k : keys) k.clear();
    for (auto& v : values) v.clear();
    seq_len = 0;
}

// ===========================================================================
// TTSEngine Constructor / Destructor
// ===========================================================================

TTSEngine::TTSEngine(const std::string& model_dir) : model_dir_(model_dir) {
    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "TTSEngine");
        memory_info_ = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));
        
        // Load required models
        text_project_ = load_model("text_project.onnx");
        codec_embed_ = load_model("codec_embed.onnx");
        code_predictor_embed_ = load_model("code_predictor_embed.onnx");
        talker_prefill_ = load_model("talker_prefill.onnx");
        talker_decode_ = load_model("talker_decode.onnx");
        code_predictor_ = load_model("code_predictor.onnx");
        vocoder_ = load_model("tokenizer12hz_decode.onnx");
        
        // Check required models
        if (!text_project_ || !codec_embed_ || !code_predictor_embed_ ||
            !talker_prefill_ || !talker_decode_ || !code_predictor_ || !vocoder_) {
            error_msg_ = "Failed to load required ONNX models";
            return;
        }
        
        // Optional: speaker encoder for voice cloning
        speaker_encoder_ = load_model("speaker_encoder.onnx");
        
        // Load tokenizer
        fs::path model_base = fs::path(model_dir_).parent_path() / "models" / "Qwen3-TTS-12Hz-0.6B-Base";
        fs::path vocab_path = model_base / "vocab.json";
        fs::path merges_path = model_base / "merges.txt";
        
        if (fs::exists(vocab_path) && fs::exists(merges_path)) {
            if (!io::load_vocab(vocab_path.string()) || !io::load_merges(merges_path.string())) {
                error_msg_ = "Failed to load tokenizer";
                return;
            }
        } else {
            std::cerr << "[TTSEngine] Warning: Tokenizer not found at " << model_base << std::endl;
        }
        
        ready_ = true;
        
    } catch (const Ort::Exception& e) {
        error_msg_ = std::string("ONNX Runtime error: ") + e.what();
    } catch (const std::exception& e) {
        error_msg_ = std::string("Error: ") + e.what();
    }
}

TTSEngine::~TTSEngine() = default;

std::unique_ptr<Ort::Session> TTSEngine::load_model(const std::string& filename) {
    fs::path model_path = fs::path(model_dir_) / filename;
    if (!fs::exists(model_path)) return nullptr;
    
    try {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(4);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // Track which provider we successfully enabled
        static bool logged_provider = false;
        bool provider_added = false;
        
#if LEAXER_TRY_CUDA
        // Try CUDA (NVIDIA GPU) first - typically fastest on supported hardware
        if (!provider_added) {
            try {
                OrtCUDAProviderOptions cuda_options;
                cuda_options.device_id = 0;
                cuda_options.arena_extend_strategy = 0;
                cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
                cuda_options.do_copy_in_default_stream = 1;
                opts.AppendExecutionProvider_CUDA(cuda_options);
                provider_added = true;
                if (!logged_provider) {
                    std::cerr << "[TTSEngine] CUDA Execution Provider enabled (NVIDIA GPU)" << std::endl;
                }
            } catch (const Ort::Exception& e) {
                // CUDA EP not available
            }
        }
#endif

#if LEAXER_TRY_ROCM
        // Try ROCm (AMD GPU)
        if (!provider_added) {
            try {
                OrtROCMProviderOptions rocm_options;
                rocm_options.device_id = 0;
                opts.AppendExecutionProvider_ROCM(rocm_options);
                provider_added = true;
                if (!logged_provider) {
                    std::cerr << "[TTSEngine] ROCm Execution Provider enabled (AMD GPU)" << std::endl;
                }
            } catch (const Ort::Exception& e) {
                // ROCm EP not available
            }
        }
#endif

#if LEAXER_TRY_COREML
        // Try CoreML (Apple Silicon GPU/Neural Engine)
        if (!provider_added) {
            try {
                // Use MLProgram (CoreML 5+) with all compute units (CPU+GPU+ANE)
                uint32_t coreml_flags = COREML_FLAG_CREATE_MLPROGRAM;
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(opts, coreml_flags));
                provider_added = true;
                if (!logged_provider) {
                    std::cerr << "[TTSEngine] CoreML Execution Provider enabled (Apple GPU/ANE)" << std::endl;
                }
            } catch (const Ort::Exception& e) {
                // CoreML EP failed - log the error
                static bool logged_coreml_error = false;
                if (!logged_coreml_error) {
                    std::cerr << "[TTSEngine] CoreML EP failed: " << e.what() << std::endl;
                    logged_coreml_error = true;
                }
            }
        }
#endif

#if LEAXER_TRY_DIRECTML
        // Try DirectML (Windows GPU - works with any GPU vendor)
        if (!provider_added) {
            try {
                opts.AppendExecutionProvider("DML");
                provider_added = true;
                if (!logged_provider) {
                    std::cerr << "[TTSEngine] DirectML Execution Provider enabled (Windows GPU)" << std::endl;
                }
            } catch (const Ort::Exception& e) {
                // DirectML EP not available
            }
        }
#endif

        // Log CPU fallback only once
        if (!provider_added && !logged_provider) {
            std::cerr << "[TTSEngine] Using CPU (no GPU acceleration available)" << std::endl;
        }
        
        logged_provider = true;
        
        return std::make_unique<Ort::Session>(*env_, model_path.c_str(), opts);
    } catch (...) {
        return nullptr;
    }
}

// ===========================================================================
// Main Synthesis Interface
// ===========================================================================

std::vector<float> TTSEngine::synthesize(const std::string& text,
                                          Language lang,
                                          const SamplingParams& params) {
    if (!ready_) return {};
    
    // Build token sequence: [IM_START, ASSISTANT, TTS_BOS, ...text..., TTS_EOS, IM_END]
    std::vector<int64_t> token_ids;
    token_ids.push_back(config::IM_START);
    token_ids.push_back(config::ASSISTANT);
    token_ids.push_back(config::TTS_BOS);
    
    if (io::is_tokenizer_ready()) {
        for (int32_t t : io::tokenize(text)) {
            token_ids.push_back(static_cast<int64_t>(t));
        }
    } else {
        std::cerr << "[TTSEngine] Tokenizer not ready" << std::endl;
        return {};
    }
    
    token_ids.push_back(config::TTS_EOS);
    token_ids.push_back(config::IM_END);
    
    return synthesize_tokens(token_ids, lang, params);
}

std::vector<float> TTSEngine::synthesize_clone(const std::string& text,
                                                const std::string& ref_audio_path,
                                                Language lang,
                                                const SamplingParams& params) {
    if (!ready_) return {};
    if (!speaker_encoder_) {
        std::cerr << "[TTSEngine] Speaker encoder not available" << std::endl;
        return {};
    }
    
    // Extract speaker embedding from reference audio
    auto speaker_embed = extract_speaker_embedding(ref_audio_path);
    if (speaker_embed.empty()) {
        std::cerr << "[TTSEngine] Failed to extract speaker embedding" << std::endl;
        return {};
    }
    
    // Build token sequence
    std::vector<int64_t> token_ids;
    token_ids.push_back(config::IM_START);
    token_ids.push_back(config::ASSISTANT);
    token_ids.push_back(config::TTS_BOS);
    
    if (io::is_tokenizer_ready()) {
        for (int32_t t : io::tokenize(text)) {
            token_ids.push_back(static_cast<int64_t>(t));
        }
    } else {
        std::cerr << "[TTSEngine] Tokenizer not ready" << std::endl;
        return {};
    }
    
    token_ids.push_back(config::TTS_EOS);
    token_ids.push_back(config::IM_END);
    
    // Synthesize with speaker embedding
    try {
        kv_cache_.clear();
        auto prompt_embeds = build_prompt_embeddings(token_ids, lang, speaker_embed);
        auto audio_codes = generate_codes(prompt_embeds, params);
        if (audio_codes.empty()) return {};
        
        std::vector<int64_t> codes_flat;
        codes_flat.reserve(audio_codes.size() * 16);
        for (const auto& frame : audio_codes) {
            for (int i = 0; i < 16; ++i) {
                codes_flat.push_back(frame[i]);
            }
        }
        return run_vocoder(codes_flat, static_cast<int64_t>(audio_codes.size()));
    } catch (const std::exception& e) {
        std::cerr << "[TTSEngine] Synthesis error: " << e.what() << std::endl;
        return {};
    }
}

std::vector<float> TTSEngine::synthesize_speaker(const std::string& text,
                                                  Speaker speaker,
                                                  Language lang,
                                                  const SamplingParams& params) {
    // For preset speakers, we need CustomVoice model with spk_id config
    // Currently just fall back to regular synthesis
    // TODO: Implement when CustomVoice ONNX exports are available
    std::cerr << "[TTSEngine] Preset speakers require CustomVoice model (not yet supported)" << std::endl;
    return synthesize(text, lang, params);
}

std::vector<float> TTSEngine::extract_speaker_embedding(const std::string& audio_path) {
    if (!speaker_encoder_) return {};
    
    // Load and resample audio to 24kHz
    int sample_rate;
    auto audio = io::read_wav(audio_path, sample_rate);
    if (audio.empty()) {
        std::cerr << "[TTSEngine] Failed to read audio: " << audio_path << std::endl;
        return {};
    }
    
    if (sample_rate != config::SAMPLE_RATE) {
        audio = io::resample(audio, sample_rate, config::SAMPLE_RATE);
    }
    
    // Extract mel spectrogram
    io::MelConfig mel_config;
    mel_config.sample_rate = config::SAMPLE_RATE;
    mel_config.n_fft = 1024;
    mel_config.hop_size = 256;
    mel_config.win_size = 1024;
    mel_config.num_mels = 128;
    mel_config.fmin = 0.0f;
    mel_config.fmax = 12000.0f;
    
    io::MelExtractor mel_extractor(mel_config);
    auto mel = mel_extractor.extract(audio);
    if (mel.empty()) {
        std::cerr << "[TTSEngine] Failed to extract mel spectrogram" << std::endl;
        return {};
    }
    
    // Run speaker encoder
    return run_speaker_encoder(mel);
}

std::vector<float> TTSEngine::run_speaker_encoder(const std::vector<float>& mel) {
    // mel is [num_mels, num_frames] row-major from MelExtractor
    // speaker_encoder expects [1, num_frames, num_mels]
    // So we need to transpose
    const int64_t num_mels = 128;
    const int64_t num_frames = static_cast<int64_t>(mel.size() / num_mels);
    
    // Transpose from [num_mels, num_frames] to [num_frames, num_mels]
    std::vector<float> mel_transposed(mel.size());
    for (int64_t m = 0; m < num_mels; ++m) {
        for (int64_t t = 0; t < num_frames; ++t) {
            mel_transposed[t * num_mels + m] = mel[m * num_frames + t];
        }
    }
    
    std::array<int64_t, 3> shape = {1, num_frames, num_mels};
    
    Ort::Value input = Ort::Value::CreateTensor<float>(
        *memory_info_, mel_transposed.data(), mel_transposed.size(),
        shape.data(), shape.size());
    
    // Get input/output names from model
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = speaker_encoder_->GetInputNameAllocated(0, allocator);
    auto output_name = speaker_encoder_->GetOutputNameAllocated(0, allocator);
    const char* in[] = {input_name.get()};
    const char* out[] = {output_name.get()};
    
    auto outputs = speaker_encoder_->Run(Ort::RunOptions{nullptr}, in, &input, 1, out, 1);
    
    auto& output_tensor = outputs[0];
    float* data = output_tensor.GetTensorMutableData<float>();
    auto tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
    size_t total = tensor_info.GetElementCount();
    
    return std::vector<float>(data, data + total);
}

std::vector<float> TTSEngine::synthesize_tokens(const std::vector<int64_t>& token_ids,
                                                 Language lang,
                                                 const SamplingParams& params) {
    if (!ready_) return {};
    
    try {
        kv_cache_.clear();
        
        // Build prompt embeddings with language control
        auto prompt_embeds = build_prompt_embeddings(token_ids, lang);
        
        // Generate audio codes
        auto audio_codes = generate_codes(prompt_embeds, params);
        if (audio_codes.empty()) return {};
        
        // Flatten codes for vocoder: [1 x frames x 16]
        std::vector<int64_t> codes_flat;
        codes_flat.reserve(audio_codes.size() * 16);
        for (const auto& frame : audio_codes) {
            for (int i = 0; i < 16; ++i) {
                codes_flat.push_back(frame[i]);
            }
        }
        
        // Run vocoder
        return run_vocoder(codes_flat, static_cast<int64_t>(audio_codes.size()));
        
    } catch (const std::exception& e) {
        std::cerr << "[TTSEngine] Synthesis error: " << e.what() << std::endl;
        return {};
    }
}

// ===========================================================================
// Prompt Building (with Language Control)
// ===========================================================================

std::vector<float> TTSEngine::build_prompt_embeddings(const std::vector<int64_t>& input_ids,
                                                       Language lang,
                                                       const std::vector<float>& speaker_embed) {
    constexpr int H = config::HIDDEN_SIZE;
    const bool has_speaker = !speaker_embed.empty();
    
    auto slice = [](const std::vector<float>& v, size_t start, size_t end) {
        return std::vector<float>(v.begin() + start, v.begin() + end);
    };
    
    auto add_vectors = [](const std::vector<float>& a, const std::vector<float>& b) {
        std::vector<float> result(a.size());
        for (size_t i = 0; i < a.size(); ++i) result[i] = a[i] + b[i];
        return result;
    };
    
    // 1. Get TTS special embeddings
    std::vector<int64_t> tts_ids = {config::TTS_BOS, config::TTS_EOS, config::TTS_PAD};
    auto tts_embeds = run_text_project(tts_ids);
    auto tts_bos = slice(tts_embeds, 0, H);
    auto tts_eos = slice(tts_embeds, H, 2*H);
    tts_pad_embed_ = slice(tts_embeds, 2*H, 3*H);
    
    // 2. Build codec prefill based on language
    std::vector<int64_t> codec_prefill;
    if (lang == Language::Auto) {
        // No-think mode (language auto-detect)
        codec_prefill = {config::CODEC_NOTHINK, config::CODEC_THINK_BOS, config::CODEC_THINK_EOS};
    } else {
        // Think mode with explicit language
        codec_prefill = {config::CODEC_THINK, config::CODEC_THINK_BOS,
                         language_to_codec_id(lang), config::CODEC_THINK_EOS};
    }
    codec_prefill.push_back(config::CODEC_PAD);
    codec_prefill.push_back(config::CODEC_BOS);
    
    auto codec_embeds = run_codec_embed_batch(codec_prefill);
    
    // Insert speaker embedding if provided (between prefill and BOS)
    if (has_speaker) {
        // Insert speaker embedding before the last element (BOS)
        size_t insert_pos = (codec_prefill.size() - 1) * H;
        std::vector<float> new_embeds;
        new_embeds.reserve(codec_embeds.size() + H);
        new_embeds.insert(new_embeds.end(), codec_embeds.begin(), codec_embeds.begin() + insert_pos);
        new_embeds.insert(new_embeds.end(), speaker_embed.begin(), speaker_embed.end());
        new_embeds.insert(new_embeds.end(), codec_embeds.begin() + insert_pos, codec_embeds.end());
        codec_embeds = std::move(new_embeds);
    }
    
    // 3. Role embedding (first 3 tokens: IM_START, ASSISTANT, TTS_BOS)
    std::vector<int64_t> role_ids(input_ids.begin(), input_ids.begin() + 3);
    auto role_embed = run_text_project(role_ids);
    
    // 4. Build pad block to align with codec embeddings
    int pad_count = static_cast<int>(codec_prefill.size()) - 2;
    if (has_speaker) pad_count += 1;  // Account for speaker embedding position
    std::vector<float> pad_block;
    pad_block.reserve(pad_count * H);
    for (int i = 0; i < pad_count; ++i) {
        pad_block.insert(pad_block.end(), tts_pad_embed_.begin(), tts_pad_embed_.end());
    }
    
    // 5. Combine pad_block + tts_bos, then ADD to codec embeddings
    std::vector<float> text_part;
    text_part.reserve((pad_count + 1) * H);
    text_part.insert(text_part.end(), pad_block.begin(), pad_block.end());
    text_part.insert(text_part.end(), tts_bos.begin(), tts_bos.end());
    
    auto codec_partial = slice(codec_embeds, 0, (pad_count + 1) * H);
    auto talker_embed = add_vectors(text_part, codec_partial);
    
    // 6. First text token + last codec embedding (BOS position)
    size_t text_start = 3;
    size_t text_end = input_ids.size() - 2;
    
    auto first_text_embed = run_text_project({input_ids[text_start]});
    auto last_codec = slice(codec_embeds, (pad_count + 1) * H, (pad_count + 2) * H);
    auto text_first_combined = add_vectors(first_text_embed, last_codec);
    
    // 7. Build full prompt: role + talker + first_text
    std::vector<float> prompt;
    prompt.reserve((3 + pad_count + 1 + 1) * H);
    prompt.insert(prompt.end(), role_embed.begin(), role_embed.end());
    prompt.insert(prompt.end(), talker_embed.begin(), talker_embed.end());
    prompt.insert(prompt.end(), text_first_combined.begin(), text_first_combined.end());
    
    // 8. Build trailing_text_hidden (remaining text + tts_eos)
    trailing_text_hidden_.clear();
    for (size_t i = text_start + 1; i < text_end; ++i) {
        auto embed = run_text_project({input_ids[i]});
        trailing_text_hidden_.insert(trailing_text_hidden_.end(), embed.begin(), embed.end());
    }
    trailing_text_hidden_.insert(trailing_text_hidden_.end(), tts_eos.begin(), tts_eos.end());
    trailing_len_ = static_cast<int>(trailing_text_hidden_.size() / H);
    
    return prompt;
}

// ===========================================================================
// ONNX Model Runners
// ===========================================================================

std::vector<float> TTSEngine::run_text_project(const std::vector<int64_t>& input_ids) {
    const int64_t seq_len = static_cast<int64_t>(input_ids.size());
    std::array<int64_t, 2> shape = {1, seq_len};
    
    Ort::Value input = Ort::Value::CreateTensor<int64_t>(
        *memory_info_, const_cast<int64_t*>(input_ids.data()), input_ids.size(),
        shape.data(), shape.size());
    
    const char* in[] = {"input_ids"};
    const char* out[] = {"embeds"};
    
    auto outputs = text_project_->Run(Ort::RunOptions{nullptr}, in, &input, 1, out, 1);
    float* data = outputs[0].GetTensorMutableData<float>();
    return std::vector<float>(data, data + seq_len * config::HIDDEN_SIZE);
}

std::vector<float> TTSEngine::run_codec_embed(int64_t codec_token) {
    std::vector<int64_t> ids = {codec_token};
    std::array<int64_t, 2> shape = {1, 1};
    
    Ort::Value input = Ort::Value::CreateTensor<int64_t>(
        *memory_info_, ids.data(), 1, shape.data(), shape.size());
    
    const char* in[] = {"input_ids"};
    const char* out[] = {"embeds"};
    
    auto outputs = codec_embed_->Run(Ort::RunOptions{nullptr}, in, &input, 1, out, 1);
    float* data = outputs[0].GetTensorMutableData<float>();
    return std::vector<float>(data, data + config::HIDDEN_SIZE);
}

std::vector<float> TTSEngine::run_codec_embed_batch(const std::vector<int64_t>& codec_tokens) {
    const int64_t n = static_cast<int64_t>(codec_tokens.size());
    std::array<int64_t, 2> shape = {1, n};
    
    Ort::Value input = Ort::Value::CreateTensor<int64_t>(
        *memory_info_, const_cast<int64_t*>(codec_tokens.data()), codec_tokens.size(),
        shape.data(), shape.size());
    
    const char* in[] = {"input_ids"};
    const char* out[] = {"embeds"};
    
    auto outputs = codec_embed_->Run(Ort::RunOptions{nullptr}, in, &input, 1, out, 1);
    float* data = outputs[0].GetTensorMutableData<float>();
    return std::vector<float>(data, data + n * config::HIDDEN_SIZE);
}

std::vector<float> TTSEngine::run_code_predictor_embed(int64_t subcode_token, int64_t generation_step) {
    std::vector<int64_t> ids = {subcode_token};
    std::vector<int64_t> step = {generation_step};
    std::array<int64_t, 2> id_shape = {1, 1};
    std::array<int64_t, 1> step_shape = {1};
    
    Ort::Value id_tensor = Ort::Value::CreateTensor<int64_t>(
        *memory_info_, ids.data(), 1, id_shape.data(), id_shape.size());
    Ort::Value step_tensor = Ort::Value::CreateTensor<int64_t>(
        *memory_info_, step.data(), 1, step_shape.data(), step_shape.size());
    
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(id_tensor));
    inputs.push_back(std::move(step_tensor));
    
    const char* in[] = {"input_ids", "generation_step"};
    const char* out[] = {"embeds"};
    
    auto outputs = code_predictor_embed_->Run(Ort::RunOptions{nullptr}, in, inputs.data(), 2, out, 1);
    float* data = outputs[0].GetTensorMutableData<float>();
    return std::vector<float>(data, data + config::HIDDEN_SIZE);
}

std::vector<float> TTSEngine::run_prefill(const std::vector<float>& inputs_embeds,
                                           const std::vector<int64_t>& attention_mask) {
    const int64_t seq_len = static_cast<int64_t>(attention_mask.size());
    std::array<int64_t, 3> embeds_shape = {1, seq_len, config::HIDDEN_SIZE};
    std::array<int64_t, 2> mask_shape = {1, seq_len};
    
    Ort::Value embeds = Ort::Value::CreateTensor<float>(
        *memory_info_, const_cast<float*>(inputs_embeds.data()), inputs_embeds.size(),
        embeds_shape.data(), embeds_shape.size());
    Ort::Value mask = Ort::Value::CreateTensor<int64_t>(
        *memory_info_, const_cast<int64_t*>(attention_mask.data()), attention_mask.size(),
        mask_shape.data(), mask_shape.size());
    
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(embeds));
    inputs.push_back(std::move(mask));
    
    std::vector<const char*> in_names = {"inputs_embeds", "attention_mask"};
    std::vector<const char*> out_names = {"logits", "last_hidden"};
    
    std::vector<std::string> kv_names;
    for (int i = 0; i < config::NUM_LAYERS; ++i) {
        kv_names.push_back("present_key_" + std::to_string(i));
        kv_names.push_back("present_value_" + std::to_string(i));
    }
    for (const auto& name : kv_names) out_names.push_back(name.c_str());
    
    auto outputs = talker_prefill_->Run(Ort::RunOptions{nullptr},
        in_names.data(), inputs.data(), inputs.size(),
        out_names.data(), out_names.size());
    
    // Extract logits
    float* logits_data = outputs[0].GetTensorMutableData<float>();
    std::vector<float> logits(logits_data, logits_data + seq_len * config::VOCAB_SIZE);
    
    // Extract last_hidden
    float* hidden_data = outputs[1].GetTensorMutableData<float>();
    last_hidden_.assign(hidden_data, hidden_data + config::HIDDEN_SIZE);
    
    // Store KV cache
    kv_cache_.seq_len = seq_len;
    for (int i = 0; i < config::NUM_LAYERS; ++i) {
        size_t kv_size = config::NUM_KV_HEADS * seq_len * config::HEAD_DIM;
        float* k = outputs[2 + i*2].GetTensorMutableData<float>();
        float* v = outputs[3 + i*2].GetTensorMutableData<float>();
        kv_cache_.keys[i].assign(k, k + kv_size);
        kv_cache_.values[i].assign(v, v + kv_size);
    }
    
    return logits;
}

std::vector<float> TTSEngine::run_decode(const std::vector<float>& input_embed,
                                          const std::vector<int64_t>& attention_mask) {
    const int64_t total_seq = static_cast<int64_t>(attention_mask.size());
    const int64_t past_seq = kv_cache_.seq_len;
    
    std::array<int64_t, 3> embeds_shape = {1, 1, config::HIDDEN_SIZE};
    std::array<int64_t, 2> mask_shape = {1, total_seq};
    std::array<int64_t, 4> kv_shape = {1, config::NUM_KV_HEADS, past_seq, config::HEAD_DIM};
    
    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<float>(
        *memory_info_, const_cast<float*>(input_embed.data()), input_embed.size(),
        embeds_shape.data(), embeds_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(
        *memory_info_, const_cast<int64_t*>(attention_mask.data()), attention_mask.size(),
        mask_shape.data(), mask_shape.size()));
    
    for (int i = 0; i < config::NUM_LAYERS; ++i) {
        inputs.push_back(Ort::Value::CreateTensor<float>(
            *memory_info_, kv_cache_.keys[i].data(), kv_cache_.keys[i].size(),
            kv_shape.data(), kv_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(
            *memory_info_, kv_cache_.values[i].data(), kv_cache_.values[i].size(),
            kv_shape.data(), kv_shape.size()));
    }
    
    std::vector<const char*> in_names = {"inputs_embeds", "attention_mask"};
    std::vector<std::string> past_names;
    for (int i = 0; i < config::NUM_LAYERS; ++i) {
        past_names.push_back("past_key_" + std::to_string(i));
        past_names.push_back("past_value_" + std::to_string(i));
    }
    for (const auto& name : past_names) in_names.push_back(name.c_str());
    
    std::vector<const char*> out_names = {"logits", "last_hidden"};
    std::vector<std::string> present_names;
    for (int i = 0; i < config::NUM_LAYERS; ++i) {
        present_names.push_back("present_key_" + std::to_string(i));
        present_names.push_back("present_value_" + std::to_string(i));
    }
    for (const auto& name : present_names) out_names.push_back(name.c_str());
    
    auto outputs = talker_decode_->Run(Ort::RunOptions{nullptr},
        in_names.data(), inputs.data(), inputs.size(),
        out_names.data(), out_names.size());
    
    // Extract logits
    float* logits_data = outputs[0].GetTensorMutableData<float>();
    std::vector<float> logits(logits_data, logits_data + config::VOCAB_SIZE);
    
    // Extract last_hidden
    float* hidden_data = outputs[1].GetTensorMutableData<float>();
    last_hidden_.assign(hidden_data, hidden_data + config::HIDDEN_SIZE);
    
    // Update KV cache
    kv_cache_.seq_len = total_seq;
    for (int i = 0; i < config::NUM_LAYERS; ++i) {
        size_t kv_size = config::NUM_KV_HEADS * total_seq * config::HEAD_DIM;
        float* k = outputs[2 + i*2].GetTensorMutableData<float>();
        float* v = outputs[3 + i*2].GetTensorMutableData<float>();
        kv_cache_.keys[i].assign(k, k + kv_size);
        kv_cache_.values[i].assign(v, v + kv_size);
    }
    
    return logits;
}

std::vector<float> TTSEngine::run_code_predictor(const std::vector<float>& inputs_embeds,
                                                  int64_t generation_step) {
    const int64_t steps = static_cast<int64_t>(inputs_embeds.size() / config::HIDDEN_SIZE);
    std::array<int64_t, 3> embeds_shape = {1, steps, config::HIDDEN_SIZE};
    std::array<int64_t, 1> step_shape = {1};
    std::vector<int64_t> gen_step = {generation_step};
    
    Ort::Value embeds = Ort::Value::CreateTensor<float>(
        *memory_info_, const_cast<float*>(inputs_embeds.data()), inputs_embeds.size(),
        embeds_shape.data(), embeds_shape.size());
    Ort::Value step = Ort::Value::CreateTensor<int64_t>(
        *memory_info_, gen_step.data(), 1, step_shape.data(), step_shape.size());
    
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(embeds));
    inputs.push_back(std::move(step));
    
    const char* in[] = {"inputs_embeds", "generation_step"};
    const char* out[] = {"logits"};
    
    auto outputs = code_predictor_->Run(Ort::RunOptions{nullptr}, in, inputs.data(), 2, out, 1);
    float* data = outputs[0].GetTensorMutableData<float>();
    return std::vector<float>(data, data + config::SUBCODE_VOCAB_SIZE);
}

std::vector<float> TTSEngine::run_vocoder(const std::vector<int64_t>& audio_codes, int64_t codes_length) {
    std::array<int64_t, 3> shape = {1, codes_length, 16};
    
    Ort::Value codes = Ort::Value::CreateTensor<int64_t>(
        *memory_info_, const_cast<int64_t*>(audio_codes.data()), audio_codes.size(),
        shape.data(), shape.size());
    
    const char* in[] = {"audio_codes"};
    const char* out[] = {"audio_values", "lengths"};
    
    auto outputs = vocoder_->Run(Ort::RunOptions{nullptr}, in, &codes, 1, out, 2);
    
    int64_t* lengths = outputs[1].GetTensorMutableData<int64_t>();
    int64_t audio_length = lengths[0];
    
    float* audio = outputs[0].GetTensorMutableData<float>();
    return std::vector<float>(audio, audio + audio_length);
}

// ===========================================================================
// Generation Loop
// ===========================================================================

std::vector<std::array<int64_t, 16>> TTSEngine::generate_codes(
    const std::vector<float>& prompt_embeds,
    const SamplingParams& params) {
    
    constexpr int H = config::HIDDEN_SIZE;
    std::vector<std::array<int64_t, 16>> all_codes;
    
    // Build attention mask
    size_t prompt_len = prompt_embeds.size() / H;
    std::vector<int64_t> attention_mask(prompt_len, 1);
    
    // Prefill
    auto logits = run_prefill(prompt_embeds, attention_mask);
    
    // Get logits for last position
    size_t last_pos = (prompt_len - 1) * config::VOCAB_SIZE;
    std::vector<float> last_logits(logits.begin() + last_pos, logits.begin() + last_pos + config::VOCAB_SIZE);
    
    // Autoregressive generation
    for (int step = 0; step < params.max_new_tokens; ++step) {
        // Suppress special tokens (2048-3071 except EOS)
        for (int i = 2048; i < 3072; ++i) {
            if (i != config::CODEC_EOS) {
                last_logits[i] = -std::numeric_limits<float>::infinity();
            }
        }
        
        // Sample codebook 0
        int64_t code0 = sample_token(last_logits, params);
        
        if (code0 == config::CODEC_EOS) break;
        
        // Predict sub-codes (1-15)
        auto subcodes = predict_subcodes(code0, params);
        
        // Store frame
        std::array<int64_t, 16> frame;
        frame[0] = code0;
        for (int i = 0; i < 15; ++i) frame[i + 1] = subcodes[i];
        all_codes.push_back(frame);
        
        // Build next input: sum of all codec embeddings + trailing text
        auto codec_embed = run_codec_embed(code0);
        for (int i = 0; i < 15; ++i) {
            auto sub_embed = run_code_predictor_embed(subcodes[i], i);
            for (size_t j = 0; j < codec_embed.size(); ++j) {
                codec_embed[j] += sub_embed[j];
            }
        }
        
        // Add trailing text or pad
        if (step < trailing_len_) {
            size_t offset = step * H;
            for (size_t j = 0; j < H; ++j) {
                codec_embed[j] += trailing_text_hidden_[offset + j];
            }
        } else {
            for (size_t j = 0; j < H; ++j) {
                codec_embed[j] += tts_pad_embed_[j];
            }
        }
        
        attention_mask.push_back(1);
        last_logits = run_decode(codec_embed, attention_mask);
    }
    
    return all_codes;
}

std::array<int64_t, 15> TTSEngine::predict_subcodes(int64_t code0, const SamplingParams& params) {
    std::array<int64_t, 15> subcodes;
    
    auto first_embed = run_codec_embed(code0);
    
    // Build sequence: [last_hidden, first_embed, sub_embeds...]
    std::vector<float> predictor_input;
    predictor_input.reserve(17 * config::HIDDEN_SIZE);
    predictor_input.insert(predictor_input.end(), last_hidden_.begin(), last_hidden_.end());
    predictor_input.insert(predictor_input.end(), first_embed.begin(), first_embed.end());
    
    for (int j = 0; j < 15; ++j) {
        auto logits = run_code_predictor(predictor_input, j);
        int64_t subcode = sample_token(logits, params);
        subcodes[j] = subcode;
        
        auto sub_embed = run_code_predictor_embed(subcode, j);
        predictor_input.insert(predictor_input.end(), sub_embed.begin(), sub_embed.end());
    }
    
    return subcodes;
}

// ===========================================================================
// Sampling
// ===========================================================================

int64_t TTSEngine::sample_token(const std::vector<float>& logits, const SamplingParams& params) {
    std::vector<float> probs = logits;
    
    // Temperature
    if (params.temperature > 0.0f && params.temperature != 1.0f) {
        for (float& p : probs) p /= params.temperature;
    }
    
    // Top-k
    if (params.top_k > 0) top_k_filter(probs, params.top_k);
    
    // Softmax
    softmax(probs);
    
    // Top-p
    if (params.top_p < 1.0f) {
        top_p_filter(probs, params.top_p);
        float sum = 0.0f;
        for (float p : probs) sum += p;
        if (sum > 0.0f) for (float& p : probs) p /= sum;
    }
    
    // Sample
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::discrete_distribution<int64_t> dist(probs.begin(), probs.end());
    return dist(gen);
}

void TTSEngine::softmax(std::vector<float>& logits) {
    float max_val = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;
    for (float& x : logits) {
        x = std::exp(x - max_val);
        sum += x;
    }
    for (float& x : logits) x /= sum;
}

void TTSEngine::top_k_filter(std::vector<float>& logits, int k) {
    if (k <= 0 || k >= static_cast<int>(logits.size())) return;
    
    std::vector<float> sorted = logits;
    std::partial_sort(sorted.begin(), sorted.begin() + k, sorted.end(), std::greater<float>());
    float threshold = sorted[k - 1];
    
    for (float& x : logits) {
        if (x < threshold) x = -std::numeric_limits<float>::infinity();
    }
}

void TTSEngine::top_p_filter(std::vector<float>& probs, float p) {
    if (p >= 1.0f) return;
    
    std::vector<size_t> indices(probs.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&probs](size_t a, size_t b) { return probs[a] > probs[b]; });
    
    float cumsum = 0.0f;
    size_t cutoff = probs.size();
    for (size_t i = 0; i < indices.size(); ++i) {
        cumsum += probs[indices[i]];
        if (cumsum > p) {
            cutoff = i + 1;
            break;
        }
    }
    
    for (size_t i = cutoff; i < indices.size(); ++i) {
        probs[indices[i]] = 0.0f;
    }
}

} // namespace leaxer_qwen
