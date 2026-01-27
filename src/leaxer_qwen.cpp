// leaxer-qwen: Pure C++ Qwen3-TTS implementation
// Main CLI entry point

#include "leaxer_qwen.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static void print_usage(const char * progname) {
    printf("Usage: %s [options]\n", progname);
    printf("\n");
    printf("Options:\n");
    printf("  -m, --model PATH      Path to GGUF model file (required)\n");
    printf("  -p, --prompt TEXT     Text to synthesize (required)\n");
    printf("  -o, --output PATH     Output WAV file (default: output.wav)\n");
    printf("  -t, --threads N       Number of threads (default: 4)\n");
    printf("  --temp FLOAT          Temperature (default: 0.9)\n");
    printf("  --top-k N             Top-k sampling (default: 50)\n");
    printf("  --top-p FLOAT         Top-p sampling (default: 0.95)\n");
    printf("  --seed N              Random seed (default: -1 for random)\n");
    printf("  -v, --version         Print version and exit\n");
    printf("  -h, --help            Print this help and exit\n");
    printf("\n");
    printf("Example:\n");
    printf("  %s -m qwen3-tts-1.7b.gguf -p \"Hello world\" -o hello.wav\n", progname);
}

int main(int argc, char ** argv) {
    // Parse arguments
    const char * model_path = nullptr;
    const char * prompt = nullptr;
    const char * output_path = "output.wav";
    int n_threads = 4;
    float temperature = 0.9f;
    int top_k = 50;
    float top_p = 0.95f;
    int seed = -1;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        if (arg == "-v" || arg == "--version") {
            printf("leaxer-qwen version %s\n", leaxer_qwen_version());
            return 0;
        }
        if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            model_path = argv[++i];
        } else if ((arg == "-p" || arg == "--prompt") && i + 1 < argc) {
            prompt = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            output_path = argv[++i];
        } else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
            n_threads = std::atoi(argv[++i]);
        } else if (arg == "--temp" && i + 1 < argc) {
            temperature = std::atof(argv[++i]);
        } else if (arg == "--top-k" && i + 1 < argc) {
            top_k = std::atoi(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            top_p = std::atof(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = std::atoi(argv[++i]);
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    // Validate required arguments
    if (!model_path) {
        fprintf(stderr, "Error: --model is required\n");
        print_usage(argv[0]);
        return 1;
    }
    if (!prompt) {
        fprintf(stderr, "Error: --prompt is required\n");
        print_usage(argv[0]);
        return 1;
    }

    printf("leaxer-qwen v%s\n", leaxer_qwen_version());
    printf("Model: %s\n", model_path);
    printf("Prompt: %s\n", prompt);
    printf("Output: %s\n", output_path);
    printf("Threads: %d\n", n_threads);
    printf("\n");

    // TODO: Implement actual TTS pipeline
    // For now, just print a placeholder message
    printf("[TODO] Loading model...\n");
    printf("[TODO] Generating speech...\n");
    printf("[TODO] Writing WAV file...\n");
    printf("\n");
    printf("Note: This is a placeholder. Implementation in progress.\n");

    return 0;
}

// API implementations

const char * leaxer_qwen_version(void) {
    return LEAXER_QWEN_VERSION;
}

void leaxer_qwen_print_system_info(void) {
    printf("leaxer-qwen system info:\n");
    printf("  Version: %s\n", LEAXER_QWEN_VERSION);
    // TODO: Add ggml backend info, SIMD support, etc.
}

struct leaxer_qwen_model_params leaxer_qwen_model_default_params(void) {
    return {
        .n_threads    = 4,
        .n_gpu_layers = 0,
        .use_mmap     = true,
        .use_mlock    = false,
    };
}

struct leaxer_qwen_gen_params leaxer_qwen_gen_default_params(void) {
    return {
        .temperature = 0.9f,
        .top_k       = 50,
        .top_p       = 0.95f,
        .max_tokens  = 2048,
        .seed        = -1,
    };
}

// Placeholder implementations - to be filled in by agents

struct leaxer_qwen_model * leaxer_qwen_load_model(
    const char * path,
    struct leaxer_qwen_model_params params
) {
    (void)path;
    (void)params;
    fprintf(stderr, "[TODO] leaxer_qwen_load_model not implemented\n");
    return nullptr;
}

void leaxer_qwen_free_model(struct leaxer_qwen_model * model) {
    (void)model;
}

struct leaxer_qwen_context * leaxer_qwen_new_context(
    struct leaxer_qwen_model * model
) {
    (void)model;
    fprintf(stderr, "[TODO] leaxer_qwen_new_context not implemented\n");
    return nullptr;
}

void leaxer_qwen_free_context(struct leaxer_qwen_context * ctx) {
    (void)ctx;
}

float * leaxer_qwen_generate(
    struct leaxer_qwen_context * ctx,
    const char * text,
    struct leaxer_qwen_gen_params params,
    size_t * n_samples
) {
    (void)ctx;
    (void)text;
    (void)params;
    *n_samples = 0;
    fprintf(stderr, "[TODO] leaxer_qwen_generate not implemented\n");
    return nullptr;
}

void leaxer_qwen_free_audio(float * audio) {
    free(audio);
}

// Forward declaration of internal write_wav function
namespace leaxer_qwen {
namespace io {
int write_wav(const char* path, const float* audio, size_t n_samples, int sample_rate);
}
}

int leaxer_qwen_write_wav(
    const char * path,
    const float * audio,
    size_t n_samples,
    int sample_rate
) {
    return leaxer_qwen::io::write_wav(path, audio, n_samples, sample_rate);
}
