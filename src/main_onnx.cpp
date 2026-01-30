// ONNX-based TTS Main Entry Point
// Uses ONNX Runtime for inference with pre-exported models

#include "tts_onnx.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

// Simple WAV writer for output
static int write_wav(const char* path, const float* audio, size_t n_samples, int sample_rate) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "Error: cannot open file for writing: %s\n", path);
        return -1;
    }
    
    // WAV header
    uint32_t byte_rate = sample_rate * 2;  // 16-bit mono
    uint32_t data_size = static_cast<uint32_t>(n_samples * 2);
    uint32_t file_size = 36 + data_size;
    
    // RIFF header
    fwrite("RIFF", 1, 4, f);
    fwrite(&file_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    
    // fmt chunk
    fwrite("fmt ", 1, 4, f);
    uint32_t fmt_size = 16;
    uint16_t audio_format = 1;  // PCM
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
    
    // Convert float samples to 16-bit PCM
    for (size_t i = 0; i < n_samples; i++) {
        float sample = audio[i];
        // Clamp to [-1, 1]
        if (sample > 1.0f) sample = 1.0f;
        if (sample < -1.0f) sample = -1.0f;
        // Convert to 16-bit
        int16_t pcm = static_cast<int16_t>(sample * 32767.0f);
        fwrite(&pcm, 2, 1, f);
    }
    
    fclose(f);
    return 0;
}

static void print_usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("\n");
    printf("ONNX-based TTS inference for Qwen3-TTS\n");
    printf("\n");
    printf("Options:\n");
    printf("  -m, --model DIR       Path to ONNX model directory (required)\n");
    printf("  -p, --prompt TEXT     Text to synthesize (required)\n");
    printf("  -o, --output PATH     Output WAV file (default: output.wav)\n");
    printf("  --temp FLOAT          Temperature (default: 0.8)\n");
    printf("  --top-k N             Top-k sampling (default: 50)\n");
    printf("  --top-p FLOAT         Top-p sampling (default: 0.95)\n");
    printf("  --max-tokens N        Maximum new tokens (default: 2048)\n");
    printf("  -h, --help            Print this help and exit\n");
    printf("\n");
    printf("Example:\n");
    printf("  %s -m hf_onnx_bundle/onnx_kv_06b -p \"Hello world\" -o hello.wav\n", progname);
}

int main(int argc, char** argv) {
    // Parse arguments
    const char* model_dir = nullptr;
    const char* prompt = nullptr;
    const char* output_path = "output.wav";
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
        } else if (arg == "--temp" && i + 1 < argc) {
            temperature = std::atof(argv[++i]);
        } else if (arg == "--top-k" && i + 1 < argc) {
            top_k = std::atoi(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            top_p = std::atof(argv[++i]);
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            max_tokens = std::atoi(argv[++i]);
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // Validate required arguments
    if (!model_dir) {
        fprintf(stderr, "Error: --model is required\n");
        print_usage(argv[0]);
        return 1;
    }
    if (!prompt) {
        fprintf(stderr, "Error: --prompt is required\n");
        print_usage(argv[0]);
        return 1;
    }
    
    // Check model directory exists
    if (!fs::exists(model_dir)) {
        fprintf(stderr, "Error: model directory not found: %s\n", model_dir);
        return 1;
    }
    
    printf("=== ONNX TTS Engine ===\n");
    printf("Model directory: %s\n", model_dir);
    printf("Prompt: %s\n", prompt);
    printf("Output: %s\n", output_path);
    printf("Temperature: %.2f\n", temperature);
    printf("Top-k: %d\n", top_k);
    printf("Top-p: %.2f\n", top_p);
    printf("Max tokens: %d\n", max_tokens);
    printf("\n");
    
    // Create output directory if needed
    fs::path output_file(output_path);
    if (output_file.has_parent_path()) {
        fs::create_directories(output_file.parent_path());
    }
    
    // Initialize TTS Engine
    printf("Loading ONNX models...\n");
    leaxer_qwen::TTSEngine engine(model_dir);
    
    if (!engine.is_ready()) {
        fprintf(stderr, "Error: Failed to initialize TTS engine: %s\n", engine.get_error().c_str());
        return 1;
    }
    
    printf("TTS engine ready!\n\n");
    
    // Set up sampling parameters
    leaxer_qwen::SamplingParams params;
    params.temperature = temperature;
    params.top_k = top_k;
    params.top_p = top_p;
    params.max_new_tokens = max_tokens;
    
    // Synthesize
    printf("Synthesizing: \"%s\"\n", prompt);
    std::vector<float> audio = engine.synthesize(prompt, params);
    
    if (audio.empty()) {
        fprintf(stderr, "Error: Synthesis failed - no audio generated\n");
        return 1;
    }
    
    printf("Generated %zu audio samples (%.2f seconds at 24kHz)\n", 
           audio.size(), 
           static_cast<float>(audio.size()) / leaxer_qwen::onnx_config::SAMPLE_RATE);
    
    // Write WAV file
    printf("Writing WAV file: %s\n", output_path);
    int result = write_wav(output_path, audio.data(), audio.size(), leaxer_qwen::onnx_config::SAMPLE_RATE);
    
    if (result != 0) {
        fprintf(stderr, "Error: Failed to write WAV file\n");
        return 1;
    }
    
    printf("\nSuccess! Audio saved to: %s\n", output_path);
    return 0;
}
