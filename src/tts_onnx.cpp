// ONNX-based TTS Engine Implementation
// Implements the generation loop using ONNX Runtime

#include "tts_onnx.h"
#include "io/tokenizer.h"

// ONNX Runtime headers
#include <onnxruntime/onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <numeric>
#include <fstream>
#include <iostream>
#include <filesystem>

namespace leaxer_qwen {

namespace fs = std::filesystem;

// ===========================================================================
// TTSEngine Constructor / Destructor
// ===========================================================================

TTSEngine::TTSEngine(const std::string& model_dir) : model_dir_(model_dir) {
    try {
        // Initialize ONNX Runtime environment
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "TTSEngine");
        memory_info_ = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));
        
        // Load all required models
        std::cout << "[TTSEngine] Loading ONNX models from: " << model_dir << std::endl;
        
        text_project_ = load_model("text_project.onnx");
        if (!text_project_) {
            error_msg_ = "Failed to load text_project.onnx";
            return;
        }
        std::cout << "[TTSEngine] Loaded text_project.onnx" << std::endl;
        
        codec_embed_ = load_model("codec_embed.onnx");
        if (!codec_embed_) {
            error_msg_ = "Failed to load codec_embed.onnx";
            return;
        }
        std::cout << "[TTSEngine] Loaded codec_embed.onnx" << std::endl;
        
        code_predictor_embed_ = load_model("code_predictor_embed.onnx");
        if (!code_predictor_embed_) {
            error_msg_ = "Failed to load code_predictor_embed.onnx";
            return;
        }
        std::cout << "[TTSEngine] Loaded code_predictor_embed.onnx" << std::endl;
        
        talker_prefill_ = load_model("talker_prefill.onnx");
        if (!talker_prefill_) {
            error_msg_ = "Failed to load talker_prefill.onnx";
            return;
        }
        std::cout << "[TTSEngine] Loaded talker_prefill.onnx" << std::endl;
        
        talker_decode_ = load_model("talker_decode.onnx");
        if (!talker_decode_) {
            error_msg_ = "Failed to load talker_decode.onnx";
            return;
        }
        std::cout << "[TTSEngine] Loaded talker_decode.onnx" << std::endl;
        
        code_predictor_ = load_model("code_predictor.onnx");
        if (!code_predictor_) {
            error_msg_ = "Failed to load code_predictor.onnx";
            return;
        }
        std::cout << "[TTSEngine] Loaded code_predictor.onnx" << std::endl;
        
        vocoder_ = load_model("tokenizer12hz_decode.onnx");
        if (!vocoder_) {
            error_msg_ = "Failed to load tokenizer12hz_decode.onnx";
            return;
        }
        std::cout << "[TTSEngine] Loaded tokenizer12hz_decode.onnx (vocoder)" << std::endl;
        
        // Optional: speaker encoder for voice cloning
        speaker_encoder_ = load_model("speaker_encoder.onnx");
        if (speaker_encoder_) {
            std::cout << "[TTSEngine] Loaded speaker_encoder.onnx (optional)" << std::endl;
        }
        
        // Load tokenizer (vocab.json and merges.txt)
        // They're in ../models/Qwen3-TTS-12Hz-0.6B-Base/ relative to model_dir
        fs::path model_base = fs::path(model_dir_).parent_path() / "models" / "Qwen3-TTS-12Hz-0.6B-Base";
        fs::path vocab_path = model_base / "vocab.json";
        fs::path merges_path = model_base / "merges.txt";
        
        if (fs::exists(vocab_path) && fs::exists(merges_path)) {
            std::cout << "[TTSEngine] Loading tokenizer from: " << model_base << std::endl;
            if (!io::load_vocab(vocab_path.string())) {
                error_msg_ = "Failed to load vocab.json";
                return;
            }
            if (!io::load_merges(merges_path.string())) {
                error_msg_ = "Failed to load merges.txt";
                return;
            }
            std::cout << "[TTSEngine] Tokenizer loaded successfully!" << std::endl;
        } else {
            std::cerr << "[TTSEngine] Warning: Tokenizer files not found at " << model_base << std::endl;
            std::cerr << "[TTSEngine] Will use placeholder tokenization (results may be poor)" << std::endl;
        }
        
        ready_ = true;
        std::cout << "[TTSEngine] All models loaded successfully!" << std::endl;
        
    } catch (const Ort::Exception& e) {
        error_msg_ = std::string("ONNX Runtime error: ") + e.what();
        std::cerr << "[TTSEngine] " << error_msg_ << std::endl;
    } catch (const std::exception& e) {
        error_msg_ = std::string("Error: ") + e.what();
        std::cerr << "[TTSEngine] " << error_msg_ << std::endl;
    }
}

TTSEngine::~TTSEngine() = default;

std::unique_ptr<Ort::Session> TTSEngine::load_model(const std::string& filename) {
    fs::path model_path = fs::path(model_dir_) / filename;
    
    if (!fs::exists(model_path)) {
        std::cerr << "[TTSEngine] Model not found: " << model_path << std::endl;
        return nullptr;
    }
    
    try {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        return std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);
    } catch (const Ort::Exception& e) {
        std::cerr << "[TTSEngine] Failed to load " << filename << ": " << e.what() << std::endl;
        return nullptr;
    }
}

// ===========================================================================
// Main Synthesis Interface
// ===========================================================================

std::vector<float> TTSEngine::synthesize(const std::string& text, 
                                          const SamplingParams& params) {
    if (!ready_) {
        std::cerr << "[TTSEngine] Engine not ready: " << error_msg_ << std::endl;
        return {};
    }
    
    std::cout << "[TTSEngine] Text: " << text << std::endl;
    
    // Build token sequence with proper special token IDs:
    // [IM_START, ...text_tokens..., IM_END, TTS_BOS]
    // The BPE tokenizer doesn't recognize special tokens, so we add them manually
    
    std::vector<int64_t> token_ids;
    
    // Add IM_START (151644)
    token_ids.push_back(onnx_config::IM_START);
    
    // Tokenize just the text content
    if (io::is_tokenizer_ready()) {
        std::vector<int32_t> text_tokens = io::tokenize(text);
        std::cout << "[TTSEngine] Tokenized to " << text_tokens.size() << " tokens: ";
        for (size_t i = 0; i < std::min(text_tokens.size(), size_t(10)); i++) {
            std::cout << text_tokens[i] << " ";
        }
        if (text_tokens.size() > 10) std::cout << "...";
        std::cout << std::endl;
        
        for (int32_t t : text_tokens) {
            token_ids.push_back(static_cast<int64_t>(t));
        }
    } else {
        std::cerr << "[TTSEngine] Warning: Tokenizer not ready!" << std::endl;
        return {};
    }
    
    // Add IM_END (151645) and TTS_BOS (151672)
    token_ids.push_back(onnx_config::IM_END);
    token_ids.push_back(onnx_config::TTS_BOS);
    
    std::cout << "[TTSEngine] Full token sequence: [";
    for (size_t i = 0; i < token_ids.size(); i++) {
        std::cout << token_ids[i];
        if (i < token_ids.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    return synthesize_tokens(token_ids, params);
}

std::vector<float> TTSEngine::synthesize_tokens(const std::vector<int64_t>& token_ids,
                                                 const SamplingParams& params) {
    if (!ready_) {
        std::cerr << "[TTSEngine] Engine not ready: " << error_msg_ << std::endl;
        return {};
    }
    
    try {
        // Clear KV cache for new generation
        kv_cache_.clear();
        
        std::cout << "[TTSEngine] Building prompt embeddings..." << std::endl;
        
        // Step 1: Build prompt embeddings
        std::vector<float> prompt_embeds = build_prompt_embeddings(token_ids);
        
        std::cout << "[TTSEngine] Starting generation loop..." << std::endl;
        
        // Step 2: Run generation loop to get audio codes
        std::vector<std::array<int64_t, 16>> audio_codes = generate_codes(prompt_embeds, params);
        
        if (audio_codes.empty()) {
            std::cerr << "[TTSEngine] Generation produced no codes" << std::endl;
            return {};
        }
        
        std::cout << "[TTSEngine] Generated " << audio_codes.size() << " code frames" << std::endl;
        
        // Step 3: Convert codes to flat tensor for vocoder
        // Shape: [1 x codes_length x 16]
        std::vector<int64_t> codes_flat;
        codes_flat.reserve(audio_codes.size() * 16);
        for (const auto& frame : audio_codes) {
            for (int i = 0; i < 16; ++i) {
                codes_flat.push_back(frame[i]);
            }
        }
        
        std::cout << "[TTSEngine] Running vocoder..." << std::endl;
        
        // Step 4: Run vocoder
        std::vector<float> audio = run_vocoder(codes_flat, static_cast<int64_t>(audio_codes.size()));
        
        std::cout << "[TTSEngine] Generated " << audio.size() << " audio samples" << std::endl;
        
        return audio;
        
    } catch (const std::exception& e) {
        std::cerr << "[TTSEngine] Synthesis error: " << e.what() << std::endl;
        return {};
    }
}

// ===========================================================================
// Prompt Building
// ===========================================================================

std::vector<float> TTSEngine::build_prompt_embeddings(const std::vector<int64_t>& input_ids) {
    // Build the proper prompt following zukky's example:
    // 1. Get TTS special embeddings (bos, eos, pad)
    // 2. Build codec prefill sequence
    // 3. Role embedding (first 3 tokens: <|im_start|>, assistant, <tts>)
    // 4. CRITICAL: ADD text embeddings to codec embeddings (not concatenate!)
    // 5. Build trailing_text_hidden for generation phase
    
    constexpr int HIDDEN = onnx_config::HIDDEN_SIZE;
    
    // Helper lambda to slice a vector
    auto slice = [](const std::vector<float>& v, size_t start, size_t end) {
        return std::vector<float>(v.begin() + start, v.begin() + end);
    };
    
    // Helper lambda to add two vectors element-wise
    auto add_vectors = [](const std::vector<float>& a, const std::vector<float>& b) {
        std::vector<float> result(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] = a[i] + b[i];
        }
        return result;
    };
    
    // 1. Get TTS special token embeddings
    std::vector<int64_t> tts_ids = {
        onnx_config::TTS_BOS, 
        onnx_config::TTS_EOS, 
        onnx_config::TTS_PAD
    };
    std::vector<float> tts_embeds = run_text_project(tts_ids);  // [3 x 1024]
    
    std::vector<float> tts_bos = slice(tts_embeds, 0, HIDDEN);
    std::vector<float> tts_eos = slice(tts_embeds, HIDDEN, 2 * HIDDEN);
    tts_pad_embed_ = slice(tts_embeds, 2 * HIDDEN, 3 * HIDDEN);
    
    // 2. Build codec prefill sequence (no-think mode)
    // [NOTHINK, THINK_BOS, THINK_EOS, PAD, BOS]
    std::vector<int64_t> codec_prefill = {
        onnx_config::CODEC_NOTHINK,
        onnx_config::CODEC_THINK_BOS,
        onnx_config::CODEC_THINK_EOS,
        onnx_config::CODEC_PAD,
        onnx_config::CODEC_BOS
    };
    std::vector<float> codec_embeds = run_codec_embed_batch(codec_prefill);  // [5 x 1024]
    
    // 3. Role embedding (first token: IM_START)
    // Format: [IM_START, ...text..., IM_END, TTS_BOS]
    std::vector<int64_t> role_ids = {input_ids[0]};  // Just IM_START
    std::vector<float> role_embed = run_text_project(role_ids);  // [1 x 1024]
    
    // 4. Build pad block to align with codec embeddings
    // We have 5 codec positions, need pad for first 3, then tts_bos for 4th
    // Since role is now 1 token (not 3), adjust accordingly
    int pad_count = 3;
    std::vector<float> pad_block;
    pad_block.reserve(pad_count * HIDDEN);
    for (int i = 0; i < pad_count; ++i) {
        pad_block.insert(pad_block.end(), tts_pad_embed_.begin(), tts_pad_embed_.end());
    }
    
    // 5. Combine pad_block + tts_bos
    // text_part = [pad, pad, pad, tts_bos] = [4 x 1024]
    std::vector<float> text_part;
    text_part.reserve((pad_count + 1) * HIDDEN);
    text_part.insert(text_part.end(), pad_block.begin(), pad_block.end());
    text_part.insert(text_part.end(), tts_bos.begin(), tts_bos.end());
    
    // 6. CRITICAL: ADD text_part to codec embeddings (first 4 positions)
    // codec_partial = codec_embeds[0:4] = [NOTHINK, THINK_BOS, THINK_EOS, PAD]
    std::vector<float> codec_partial = slice(codec_embeds, 0, 4 * HIDDEN);
    std::vector<float> talker_embed = add_vectors(text_part, codec_partial);
    
    // 7. First text token + last codec embedding (BOS)
    // Format: [IM_START, ...text..., IM_END, TTS_BOS]
    // text_start = 1 (after IM_START)
    // text_end = len - 2 (before IM_END and TTS_BOS)
    
    size_t text_start = 1;  // After IM_START
    size_t text_end = input_ids.size() - 2;  // Before IM_END and TTS_BOS
    
    // First text token embedding
    std::vector<int64_t> first_text_id = {input_ids[text_start]};
    std::vector<float> first_text_embed = run_text_project(first_text_id);
    
    // Last codec embedding (BOS position)
    std::vector<float> last_codec = slice(codec_embeds, 4 * HIDDEN, 5 * HIDDEN);
    
    // ADD first text with last codec
    std::vector<float> text_first_combined = add_vectors(first_text_embed, last_codec);
    
    // 8. Build full prompt
    std::vector<float> prompt;
    prompt.reserve((1 + 4 + 1) * HIDDEN);  // role(1) + talker(4) + first_text(1)
    prompt.insert(prompt.end(), role_embed.begin(), role_embed.end());
    prompt.insert(prompt.end(), talker_embed.begin(), talker_embed.end());
    prompt.insert(prompt.end(), text_first_combined.begin(), text_first_combined.end());
    
    // 9. Build trailing_text_hidden (remaining text tokens + tts_eos)
    // This is what gets ADDED during generation, one embedding per step
    trailing_text_hidden_.clear();
    
    // Remaining text tokens (from position text_start+1 to text_end)
    for (size_t i = text_start + 1; i < text_end; ++i) {
        std::vector<int64_t> token_id = {input_ids[i]};
        std::vector<float> embed = run_text_project(token_id);
        trailing_text_hidden_.insert(trailing_text_hidden_.end(), embed.begin(), embed.end());
    }
    
    // Add tts_eos at the end
    trailing_text_hidden_.insert(trailing_text_hidden_.end(), tts_eos.begin(), tts_eos.end());
    
    trailing_len_ = static_cast<int>(trailing_text_hidden_.size() / HIDDEN);
    
    std::cout << "[TTSEngine] Prompt structure:" << std::endl;
    std::cout << "  - Role: 1 (IM_START)" << std::endl;
    std::cout << "  - Talker embed: 4 (pad+bos added to codec prefill)" << std::endl;
    std::cout << "  - First text: 1" << std::endl;
    std::cout << "  - Trailing text: " << trailing_len_ << " steps" << std::endl;
    std::cout << "  - Total prompt: " << (prompt.size() / HIDDEN) << " positions" << std::endl;
    
    return prompt;
}

// ===========================================================================
// ONNX Model Runners
// ===========================================================================

std::vector<float> TTSEngine::run_text_project(const std::vector<int64_t>& input_ids) {
    // Input: input_ids [batch x tokens]
    // Output: embeds [batch x tokens x 1024]
    
    const int64_t batch = 1;
    const int64_t seq_len = static_cast<int64_t>(input_ids.size());
    
    std::array<int64_t, 2> input_shape = {batch, seq_len};
    
    Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
        *memory_info_, const_cast<int64_t*>(input_ids.data()), input_ids.size(),
        input_shape.data(), input_shape.size());
    
    const char* input_names[] = {"input_ids"};
    const char* output_names[] = {"embeds"};
    
    auto output_tensors = text_project_->Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 1);
    
    // Extract output
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    size_t output_size = seq_len * onnx_config::HIDDEN_SIZE;
    
    return std::vector<float>(output_data, output_data + output_size);
}

std::vector<float> TTSEngine::run_codec_embed(int64_t codec_token) {
    // Input: input_ids [batch x 1]
    // Output: embeds [batch x 1 x 1024]
    
    std::vector<int64_t> input_ids = {codec_token};
    std::array<int64_t, 2> input_shape = {1, 1};
    
    Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
        *memory_info_, input_ids.data(), 1,
        input_shape.data(), input_shape.size());
    
    const char* input_names[] = {"input_ids"};
    const char* output_names[] = {"embeds"};
    
    auto output_tensors = codec_embed_->Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 1);
    
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    return std::vector<float>(output_data, output_data + onnx_config::HIDDEN_SIZE);
}

std::vector<float> TTSEngine::run_codec_embed_batch(const std::vector<int64_t>& codec_tokens) {
    // Input: input_ids [batch x num_tokens]
    // Output: embeds [batch x num_tokens x 1024]
    
    const int64_t num_tokens = static_cast<int64_t>(codec_tokens.size());
    std::array<int64_t, 2> input_shape = {1, num_tokens};
    
    Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
        *memory_info_, const_cast<int64_t*>(codec_tokens.data()), codec_tokens.size(),
        input_shape.data(), input_shape.size());
    
    const char* input_names[] = {"input_ids"};
    const char* output_names[] = {"embeds"};
    
    auto output_tensors = codec_embed_->Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 1);
    
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    size_t output_size = num_tokens * onnx_config::HIDDEN_SIZE;
    return std::vector<float>(output_data, output_data + output_size);
}

std::vector<float> TTSEngine::run_code_predictor_embed(int64_t subcode_token, int64_t generation_step) {
    // Inputs: input_ids [batch x 1], generation_step [1]
    // Output: embeds [batch x 1 x 1024]
    
    std::vector<int64_t> input_ids = {subcode_token};
    std::vector<int64_t> gen_step = {generation_step};
    
    std::array<int64_t, 2> input_shape = {1, 1};
    std::array<int64_t, 1> step_shape = {1};
    
    Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
        *memory_info_, input_ids.data(), 1,
        input_shape.data(), input_shape.size());
    
    Ort::Value step_tensor = Ort::Value::CreateTensor<int64_t>(
        *memory_info_, gen_step.data(), 1,
        step_shape.data(), step_shape.size());
    
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(input_tensor));
    inputs.push_back(std::move(step_tensor));
    
    const char* input_names[] = {"input_ids", "generation_step"};
    const char* output_names[] = {"embeds"};
    
    auto output_tensors = code_predictor_embed_->Run(
        Ort::RunOptions{nullptr},
        input_names, inputs.data(), 2,
        output_names, 1);
    
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    return std::vector<float>(output_data, output_data + onnx_config::HIDDEN_SIZE);
}

std::vector<float> TTSEngine::run_prefill(const std::vector<float>& inputs_embeds,
                                           const std::vector<int64_t>& attention_mask) {
    // Inputs:
    //   inputs_embeds: [batch x tokens x 1024]
    //   attention_mask: [batch x tokens]
    // Outputs:
    //   logits: [batch x tokens x 3072]
    //   last_hidden: [batch x 1 x 1024]
    //   present_key_0..27, present_value_0..27
    
    const int64_t batch = 1;
    const int64_t seq_len = static_cast<int64_t>(attention_mask.size());
    
    std::array<int64_t, 3> embeds_shape = {batch, seq_len, onnx_config::HIDDEN_SIZE};
    std::array<int64_t, 2> mask_shape = {batch, seq_len};
    
    // Create input tensors
    Ort::Value embeds_tensor = Ort::Value::CreateTensor<float>(
        *memory_info_, const_cast<float*>(inputs_embeds.data()), inputs_embeds.size(),
        embeds_shape.data(), embeds_shape.size());
    
    Ort::Value mask_tensor = Ort::Value::CreateTensor<int64_t>(
        *memory_info_, const_cast<int64_t*>(attention_mask.data()), attention_mask.size(),
        mask_shape.data(), mask_shape.size());
    
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(embeds_tensor));
    inputs.push_back(std::move(mask_tensor));
    
    // Build input/output names
    std::vector<const char*> input_names = {"inputs_embeds", "attention_mask"};
    
    std::vector<const char*> output_names;
    output_names.push_back("logits");
    output_names.push_back("last_hidden");
    
    // Add present key/value names for each layer
    std::vector<std::string> kv_names;
    for (int i = 0; i < onnx_config::NUM_LAYERS; ++i) {
        kv_names.push_back("present_key_" + std::to_string(i));
        kv_names.push_back("present_value_" + std::to_string(i));
    }
    for (const auto& name : kv_names) {
        output_names.push_back(name.c_str());
    }
    
    // Run prefill
    auto output_tensors = talker_prefill_->Run(
        Ort::RunOptions{nullptr},
        input_names.data(), inputs.data(), inputs.size(),
        output_names.data(), output_names.size());
    
    // Extract logits
    float* logits_data = output_tensors[0].GetTensorMutableData<float>();
    size_t logits_size = seq_len * onnx_config::VOCAB_SIZE;
    std::vector<float> logits(logits_data, logits_data + logits_size);
    
    // Extract last_hidden for code predictor
    float* hidden_data = output_tensors[1].GetTensorMutableData<float>();
    last_hidden_.assign(hidden_data, hidden_data + onnx_config::HIDDEN_SIZE);
    
    // Extract and store KV cache
    // Output order: logits, last_hidden, present_key_0, present_value_0, ...
    kv_cache_.seq_len = seq_len;
    for (int i = 0; i < onnx_config::NUM_LAYERS; ++i) {
        // present_key_i shape: [batch x num_kv_heads x seq_len x head_dim]
        size_t kv_size = onnx_config::NUM_KV_HEADS * seq_len * onnx_config::HEAD_DIM;
        
        float* key_data = output_tensors[2 + i * 2].GetTensorMutableData<float>();
        kv_cache_.keys[i].assign(key_data, key_data + kv_size);
        
        float* value_data = output_tensors[3 + i * 2].GetTensorMutableData<float>();
        kv_cache_.values[i].assign(value_data, value_data + kv_size);
    }
    
    return logits;
}

std::vector<float> TTSEngine::run_decode(const std::vector<float>& input_embed,
                                          const std::vector<int64_t>& attention_mask) {
    // Inputs:
    //   inputs_embeds: [batch x 1 x 1024]
    //   attention_mask: [batch x total_seq]
    //   past_key_0..27, past_value_0..27
    // Outputs:
    //   logits: [batch x 1 x 3072]
    //   last_hidden: [batch x 1 x 1024]
    //   present_key_0..27, present_value_0..27
    
    const int64_t batch = 1;
    const int64_t total_seq = static_cast<int64_t>(attention_mask.size());
    const int64_t past_seq = kv_cache_.seq_len;
    
    std::array<int64_t, 3> embeds_shape = {batch, 1, onnx_config::HIDDEN_SIZE};
    std::array<int64_t, 2> mask_shape = {batch, total_seq};
    std::array<int64_t, 4> kv_shape = {batch, onnx_config::NUM_KV_HEADS, past_seq, onnx_config::HEAD_DIM};
    
    // Build input tensors
    std::vector<Ort::Value> inputs;
    
    inputs.push_back(Ort::Value::CreateTensor<float>(
        *memory_info_, const_cast<float*>(input_embed.data()), input_embed.size(),
        embeds_shape.data(), embeds_shape.size()));
    
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(
        *memory_info_, const_cast<int64_t*>(attention_mask.data()), attention_mask.size(),
        mask_shape.data(), mask_shape.size()));
    
    // Add past KV cache tensors
    for (int i = 0; i < onnx_config::NUM_LAYERS; ++i) {
        inputs.push_back(Ort::Value::CreateTensor<float>(
            *memory_info_, kv_cache_.keys[i].data(), kv_cache_.keys[i].size(),
            kv_shape.data(), kv_shape.size()));
        
        inputs.push_back(Ort::Value::CreateTensor<float>(
            *memory_info_, kv_cache_.values[i].data(), kv_cache_.values[i].size(),
            kv_shape.data(), kv_shape.size()));
    }
    
    // Build input/output names
    std::vector<const char*> input_names = {"inputs_embeds", "attention_mask"};
    std::vector<std::string> past_kv_names;
    for (int i = 0; i < onnx_config::NUM_LAYERS; ++i) {
        past_kv_names.push_back("past_key_" + std::to_string(i));
        past_kv_names.push_back("past_value_" + std::to_string(i));
    }
    for (const auto& name : past_kv_names) {
        input_names.push_back(name.c_str());
    }
    
    std::vector<const char*> output_names = {"logits", "last_hidden"};
    std::vector<std::string> present_kv_names;
    for (int i = 0; i < onnx_config::NUM_LAYERS; ++i) {
        present_kv_names.push_back("present_key_" + std::to_string(i));
        present_kv_names.push_back("present_value_" + std::to_string(i));
    }
    for (const auto& name : present_kv_names) {
        output_names.push_back(name.c_str());
    }
    
    // Run decode
    auto output_tensors = talker_decode_->Run(
        Ort::RunOptions{nullptr},
        input_names.data(), inputs.data(), inputs.size(),
        output_names.data(), output_names.size());
    
    // Extract logits
    float* logits_data = output_tensors[0].GetTensorMutableData<float>();
    std::vector<float> logits(logits_data, logits_data + onnx_config::VOCAB_SIZE);
    
    // Extract last_hidden
    float* hidden_data = output_tensors[1].GetTensorMutableData<float>();
    last_hidden_.assign(hidden_data, hidden_data + onnx_config::HIDDEN_SIZE);
    
    // Update KV cache (present includes past + new)
    kv_cache_.seq_len = total_seq;
    for (int i = 0; i < onnx_config::NUM_LAYERS; ++i) {
        size_t kv_size = onnx_config::NUM_KV_HEADS * total_seq * onnx_config::HEAD_DIM;
        
        float* key_data = output_tensors[2 + i * 2].GetTensorMutableData<float>();
        kv_cache_.keys[i].assign(key_data, key_data + kv_size);
        
        float* value_data = output_tensors[3 + i * 2].GetTensorMutableData<float>();
        kv_cache_.values[i].assign(value_data, value_data + kv_size);
    }
    
    return logits;
}

std::vector<float> TTSEngine::run_code_predictor(const std::vector<float>& inputs_embeds,
                                                  int64_t generation_step) {
    // Inputs:
    //   inputs_embeds: [batch x steps x 1024]
    //   generation_step: [batch] (0-14)
    // Output:
    //   logits: [batch x 2048]
    
    const int64_t batch = 1;
    const int64_t steps = static_cast<int64_t>(inputs_embeds.size() / onnx_config::HIDDEN_SIZE);
    
    std::array<int64_t, 3> embeds_shape = {batch, steps, onnx_config::HIDDEN_SIZE};
    std::array<int64_t, 1> step_shape = {batch};
    
    std::vector<int64_t> gen_step = {generation_step};
    
    Ort::Value embeds_tensor = Ort::Value::CreateTensor<float>(
        *memory_info_, const_cast<float*>(inputs_embeds.data()), inputs_embeds.size(),
        embeds_shape.data(), embeds_shape.size());
    
    Ort::Value step_tensor = Ort::Value::CreateTensor<int64_t>(
        *memory_info_, gen_step.data(), 1,
        step_shape.data(), step_shape.size());
    
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(embeds_tensor));
    inputs.push_back(std::move(step_tensor));
    
    const char* input_names[] = {"inputs_embeds", "generation_step"};
    const char* output_names[] = {"logits"};
    
    auto output_tensors = code_predictor_->Run(
        Ort::RunOptions{nullptr},
        input_names, inputs.data(), 2,
        output_names, 1);
    
    float* logits_data = output_tensors[0].GetTensorMutableData<float>();
    return std::vector<float>(logits_data, logits_data + onnx_config::SUBCODE_VOCAB_SIZE);
}

std::vector<float> TTSEngine::run_vocoder(const std::vector<int64_t>& audio_codes,
                                           int64_t codes_length) {
    // Input: audio_codes [batch x codes_length x 16]
    // Output: audio_values [batch x audio_length]
    
    const int64_t batch = 1;
    std::array<int64_t, 3> codes_shape = {batch, codes_length, 16};
    
    Ort::Value codes_tensor = Ort::Value::CreateTensor<int64_t>(
        *memory_info_, const_cast<int64_t*>(audio_codes.data()), audio_codes.size(),
        codes_shape.data(), codes_shape.size());
    
    const char* input_names[] = {"audio_codes"};
    const char* output_names[] = {"audio_values", "lengths"};
    
    auto output_tensors = vocoder_->Run(
        Ort::RunOptions{nullptr},
        input_names, &codes_tensor, 1,
        output_names, 2);
    
    // Get audio length from the lengths output
    int64_t* lengths_data = output_tensors[1].GetTensorMutableData<int64_t>();
    int64_t audio_length = lengths_data[0];
    
    // Extract audio samples
    float* audio_data = output_tensors[0].GetTensorMutableData<float>();
    return std::vector<float>(audio_data, audio_data + audio_length);
}

// ===========================================================================
// Generation Loop
// ===========================================================================

std::vector<std::array<int64_t, 16>> TTSEngine::generate_codes(
    const std::vector<float>& prompt_embeds,
    const SamplingParams& params) {
    
    constexpr int HIDDEN = onnx_config::HIDDEN_SIZE;
    std::vector<std::array<int64_t, 16>> all_codes;
    
    // Build attention mask (all ones for prompt)
    size_t prompt_len = prompt_embeds.size() / HIDDEN;
    std::vector<int64_t> attention_mask(prompt_len, 1);
    
    // Phase 1: Prefill - process entire prompt
    std::cout << "[Generate] Running prefill with " << prompt_len << " tokens..." << std::endl;
    std::vector<float> logits = run_prefill(prompt_embeds, attention_mask);
    
    // Get logits for last position
    size_t last_pos = (prompt_len - 1) * onnx_config::VOCAB_SIZE;
    std::vector<float> last_logits(
        logits.begin() + last_pos,
        logits.begin() + last_pos + onnx_config::VOCAB_SIZE);
    
    std::cout << "[Generate] Starting autoregressive loop (trailing_len=" << trailing_len_ << ")..." << std::endl;
    
    // Phase 2: Autoregressive generation loop
    for (int step = 0; step < params.max_new_tokens; ++step) {
        // Suppress special tokens (2048-3071 except EOS=2150)
        // These are control tokens that shouldn't be sampled during audio generation
        for (int i = 2048; i < 3072; ++i) {
            if (i != onnx_config::CODEC_EOS) {
                last_logits[i] = -std::numeric_limits<float>::infinity();
            }
        }
        
        // Sample codebook 0 token from logits
        int64_t code0 = sample_token(last_logits, params);
        
        // Check for EOS
        if (code0 == onnx_config::CODEC_EOS) {
            std::cout << "[Generate] EOS reached at step " << step << std::endl;
            break;
        }
        
        // Predict sub-codes (codebooks 1-15)
        std::array<int64_t, 15> subcodes = predict_subcodes(code0, params);
        
        // Store all 16 codes for this frame
        std::array<int64_t, 16> frame;
        frame[0] = code0;
        for (int i = 0; i < 15; ++i) {
            frame[i + 1] = subcodes[i];
        }
        all_codes.push_back(frame);
        
        if ((step + 1) % 100 == 0) {
            std::cout << "[Generate] Step " << (step + 1) << " / " << params.max_new_tokens << std::endl;
        }
        
        // Prepare input for next decode step
        // Sum all codec embeddings: codec0 + sum(subcode embeds)
        std::vector<float> codec_embed = run_codec_embed(code0);
        
        for (int i = 0; i < 15; ++i) {
            std::vector<float> sub_embed = run_code_predictor_embed(subcodes[i], i);
            // Add to codec_embed
            for (size_t j = 0; j < codec_embed.size(); ++j) {
                codec_embed[j] += sub_embed[j];
            }
        }
        
        // CRITICAL: Add trailing_text_hidden at each step!
        // This is what makes the model "see" the remaining text during generation
        if (step < trailing_len_) {
            size_t offset = step * HIDDEN;
            for (size_t j = 0; j < HIDDEN; ++j) {
                codec_embed[j] += trailing_text_hidden_[offset + j];
            }
        } else {
            // After trailing text is exhausted, add tts_pad_embed
            for (size_t j = 0; j < HIDDEN; ++j) {
                codec_embed[j] += tts_pad_embed_[j];
            }
        }
        
        // Extend attention mask
        attention_mask.push_back(1);
        
        // Run decode step
        last_logits = run_decode(codec_embed, attention_mask);
    }
    
    return all_codes;
}

std::array<int64_t, 15> TTSEngine::predict_subcodes(int64_t code0, const SamplingParams& params) {
    std::array<int64_t, 15> subcodes;
    
    // Get embedding for codebook 0 token
    std::vector<float> first_embed = run_codec_embed(code0);
    
    // Build sequence for code predictor: [last_hidden, first_embed, sub_embeds...]
    std::vector<float> predictor_input;
    predictor_input.reserve((2 + 15) * onnx_config::HIDDEN_SIZE);
    
    // Start with last_hidden and first_embed
    predictor_input.insert(predictor_input.end(), last_hidden_.begin(), last_hidden_.end());
    predictor_input.insert(predictor_input.end(), first_embed.begin(), first_embed.end());
    
    // Predict each sub-codebook token
    for (int j = 0; j < 15; ++j) {
        // Run code predictor
        std::vector<float> logits = run_code_predictor(predictor_input, j);
        
        // Sample sub-code
        int64_t subcode = sample_token(logits, params);
        subcodes[j] = subcode;
        
        // Get embedding for this sub-code and append to input
        std::vector<float> sub_embed = run_code_predictor_embed(subcode, j);
        predictor_input.insert(predictor_input.end(), sub_embed.begin(), sub_embed.end());
    }
    
    return subcodes;
}

// ===========================================================================
// Sampling
// ===========================================================================

int64_t TTSEngine::sample_token(const std::vector<float>& logits, const SamplingParams& params) {
    std::vector<float> probs = logits;
    
    // Apply temperature
    if (params.temperature > 0.0f && params.temperature != 1.0f) {
        for (float& p : probs) {
            p /= params.temperature;
        }
    }
    
    // Apply top-k filtering
    if (params.top_k > 0) {
        top_k_filter(probs, params.top_k);
    }
    
    // Apply softmax
    softmax(probs);
    
    // Apply top-p filtering
    if (params.top_p < 1.0f) {
        top_p_filter(probs, params.top_p);
        // Re-normalize after top-p
        float sum = 0.0f;
        for (float p : probs) sum += p;
        if (sum > 0.0f) {
            for (float& p : probs) p /= sum;
        }
    }
    
    // Sample from distribution
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
    
    for (float& x : logits) {
        x /= sum;
    }
}

void TTSEngine::top_k_filter(std::vector<float>& logits, int k) {
    if (k <= 0 || k >= static_cast<int>(logits.size())) return;
    
    // Find k-th largest value
    std::vector<float> sorted = logits;
    std::partial_sort(sorted.begin(), sorted.begin() + k, sorted.end(), std::greater<float>());
    float threshold = sorted[k - 1];
    
    // Zero out values below threshold
    for (float& x : logits) {
        if (x < threshold) {
            x = -std::numeric_limits<float>::infinity();
        }
    }
}

void TTSEngine::top_p_filter(std::vector<float>& probs, float p) {
    if (p >= 1.0f) return;
    
    // Sort indices by probability (descending)
    std::vector<size_t> indices(probs.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), 
              [&probs](size_t a, size_t b) { return probs[a] > probs[b]; });
    
    // Find cutoff where cumulative probability exceeds p
    float cumsum = 0.0f;
    size_t cutoff = probs.size();
    
    for (size_t i = 0; i < indices.size(); ++i) {
        cumsum += probs[indices[i]];
        if (cumsum > p) {
            cutoff = i + 1;  // Include this token
            break;
        }
    }
    
    // Zero out tokens beyond cutoff
    for (size_t i = cutoff; i < indices.size(); ++i) {
        probs[indices[i]] = 0.0f;
    }
}

} // namespace leaxer_qwen
