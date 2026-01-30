// Qwen3-TTS ONNX Inference CLI
// Usage: leaxer-tts -m <model_dir> -p "text" [-o output.wav] [--lang en|zh|ja|ko|auto]

#include "tts_onnx.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

static int write_wav(const char* path, const float* audio, size_t n_samples, int sample_rate) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    
    uint32_t byte_rate = sample_rate * 2;
    uint32_t data_size = static_cast<uint32_t>(n_samples * 2);
    uint32_t file_size = 36 + data_size;
    
    // RIFF header
    fwrite("RIFF", 1, 4, f);
    fwrite(&file_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    
    // fmt chunk
    fwrite("fmt ", 1, 4, f);
    uint32_t fmt_size = 16;
    uint16_t audio_format = 1;
    uint16_t num_channels = 1;
    uint16_t bits_per_sample = 16;
    uint16_t block_align = 2;
    fwrite(&fmt_size, 4, 1, f);
    fwrite(&audio_format, 2, 1, f);
    fwrite(&num_channels, 2, 1, f);
    fwrite(&sample_rate, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    fwrite(&block_align, 2, 1, f);
    fwrite(&bits_per_sample, 2, 1, f);
    
    // data chunk
    fwrite("data", 1, 4, f);
    fwrite(&data_size, 4, 1, f);
    
    // Convert float to 16-bit PCM
    for (size_t i = 0; i < n_samples; i++) {
        float sample = audio[i];
        if (sample > 1.0f) sample = 1.0f;
        if (sample < -1.0f) sample = -1.0f;
        int16_t pcm = static_cast<int16_t>(sample * 32767.0f);
        fwrite(&pcm, 2, 1, f);
    }
    
    fclose(f);
    return 0;
}

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n\n", prog);
    printf("Qwen3-TTS ONNX inference\n\n");
    printf("Options:\n");
    printf("  -m, --model DIR       ONNX model directory (required)\n");
    printf("  -p, --prompt TEXT     Text to synthesize (required)\n");
    printf("  -o, --output PATH     Output WAV file (default: output.wav)\n");
    printf("  --lang LANG           Language: auto, en, zh, ja, ko (default: auto)\n");
    printf("  --temp FLOAT          Temperature (default: 0.8)\n");
    printf("  --top-k N             Top-k sampling (default: 50)\n");
    printf("  --top-p FLOAT         Top-p sampling (default: 0.95)\n");
    printf("  --max-tokens N        Max tokens (default: 2048)\n");
    printf("  -h, --help            Show this help\n");
    printf("\nExample:\n");
    printf("  %s -m onnx/onnx_kv_06b -p \"Hello world\" -o hello.wav\n", prog);
}

static leaxer_qwen::Language parse_language(const char* lang) {
    std::string s = lang;
    if (s == "en" || s == "english") return leaxer_qwen::Language::English;
    if (s == "zh" || s == "chinese") return leaxer_qwen::Language::Chinese;
    if (s == "ja" || s == "japanese") return leaxer_qwen::Language::Japanese;
    if (s == "ko" || s == "korean") return leaxer_qwen::Language::Korean;
    return leaxer_qwen::Language::Auto;
}

int main(int argc, char** argv) {
    const char* model_dir = nullptr;
    const char* prompt = nullptr;
    const char* output_path = "output.wav";
    const char* lang_str = "auto";
    float temperature = 0.8f;
    int top_k = 50;
    float top_p = 0.95f;
    int max_tokens = 2048;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            model_dir = argv[++i];
        } else if ((arg == "-p" || arg == "--prompt") && i + 1 < argc) {
            prompt = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--lang" && i + 1 < argc) {
            lang_str = argv[++i];
        } else if (arg == "--temp" && i + 1 < argc) {
            temperature = std::atof(argv[++i]);
        } else if (arg == "--top-k" && i + 1 < argc) {
            top_k = std::atoi(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            top_p = std::atof(argv[++i]);
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            max_tokens = std::atoi(argv[++i]);
        }
    }
    
    if (!model_dir || !prompt) {
        fprintf(stderr, "Error: --model and --prompt are required\n");
        print_usage(argv[0]);
        return 1;
    }
    
    if (!fs::exists(model_dir)) {
        fprintf(stderr, "Error: model directory not found: %s\n", model_dir);
        return 1;
    }
    
    auto lang = parse_language(lang_str);
    
    printf("Model: %s\n", model_dir);
    printf("Text: %s\n", prompt);
    printf("Language: %s\n", lang_str);
    printf("Output: %s\n\n", output_path);
    
    // Create output directory if needed
    fs::path out(output_path);
    if (out.has_parent_path()) fs::create_directories(out.parent_path());
    
    // Initialize engine
    leaxer_qwen::TTSEngine engine(model_dir);
    if (!engine.is_ready()) {
        fprintf(stderr, "Error: %s\n", engine.get_error().c_str());
        return 1;
    }
    
    // Set sampling params
    leaxer_qwen::SamplingParams params;
    params.temperature = temperature;
    params.top_k = top_k;
    params.top_p = top_p;
    params.max_new_tokens = max_tokens;
    
    // Synthesize
    printf("Synthesizing...\n");
    auto audio = engine.synthesize(prompt, lang, params);
    
    if (audio.empty()) {
        fprintf(stderr, "Error: synthesis failed\n");
        return 1;
    }
    
    printf("Generated %.2f seconds of audio\n", 
           static_cast<float>(audio.size()) / leaxer_qwen::config::SAMPLE_RATE);
    
    // Write WAV
    if (write_wav(output_path, audio.data(), audio.size(), leaxer_qwen::config::SAMPLE_RATE) != 0) {
        fprintf(stderr, "Error: failed to write WAV\n");
        return 1;
    }
    
    printf("Saved to: %s\n", output_path);
    return 0;
}
