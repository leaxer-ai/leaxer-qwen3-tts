// leaxer-qwen: Pure C++ Qwen3-TTS implementation
// Main CLI entry point

#include "leaxer_qwen.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

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

// Forward declarations of internal functions
namespace leaxer_qwen {
namespace io {
int write_wav(const char* path, const float* audio, size_t n_samples, int sample_rate);
std::vector<int32_t> tokenize(const std::string& text);
}

namespace model {
int generate_tokens(
    struct ggml_context * ctx,
    const int * prompt_tokens,
    int prompt_len,
    struct ggml_tensor * embed_weight,
    struct ggml_tensor ** layer_weights,
    int n_layers,
    struct ggml_tensor * norm_weight,
    struct ggml_tensor * lm_head_weight,
    int max_tokens,
    float temperature,
    int top_k,
    float top_p,
    int eos_token_id,
    uint64_t * rng_state,
    int * output_tokens);
}

namespace vocoder {
void vocoder_decode(
    struct ggml_tensor * dst,
    const struct ggml_tensor * codes,
    const struct ggml_tensor * codebooks,
    const struct ggml_tensor ** upsample_weights,
    const struct ggml_tensor ** upsample_alphas,
    const struct ggml_tensor ** upsample_betas,
    int * kernel_sizes,
    int * paddings);
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

// End-to-end TTS generation function
// Connects: text → tokenize → LLM → vocoder → audio
// This is a simplified version that skips the code predictor (not yet implemented)
//
// Parameters:
//   ctx: ggml context with sufficient memory
//   text: input text to synthesize
//   embed_weight: token embedding matrix
//   layer_weights: array of pointers to transformer layer weights
//   n_layers: number of transformer layers
//   norm_weight: final layer norm weight
//   lm_head_weight: output projection weight
//   codebooks: vocoder codebook embeddings [16, 2048, 512]
//   upsample_weights: array of 4 upsample layer weights
//   upsample_alphas: array of 4 alpha parameters
//   upsample_betas: array of 4 beta parameters
//   temperature: sampling temperature
//   top_k: top-k sampling parameter
//   top_p: top-p sampling parameter
//   seed: random seed (-1 for time-based)
//   n_samples_out: output parameter for number of audio samples generated
//
// Returns: pointer to audio samples (24kHz float32 mono), or nullptr on failure
//         Caller must free with leaxer_qwen_free_audio()
float * tts_generate(
    struct ggml_context * ctx,
    const char * text,
    struct ggml_tensor * embed_weight,
    struct ggml_tensor ** layer_weights,
    int n_layers,
    struct ggml_tensor * norm_weight,
    struct ggml_tensor * lm_head_weight,
    struct ggml_tensor * codebooks,
    struct ggml_tensor ** upsample_weights,
    struct ggml_tensor ** upsample_alphas,
    struct ggml_tensor ** upsample_betas,
    float temperature,
    int top_k,
    float top_p,
    int seed,
    size_t * n_samples_out
) {
    using namespace leaxer_qwen;

    // Special token IDs (from qwen_tts.cpp)
    constexpr int IM_START_TOKEN_ID = 151644;
    constexpr int IM_END_TOKEN_ID = 151645;
    constexpr int TTS_BOS_TOKEN_ID = 151672;
    constexpr int CODEC_EOS_ID = 4198;

    // Step 1: Tokenize input text
    std::string text_str(text);
    std::vector<int32_t> text_tokens = io::tokenize(text_str);

    if (text_tokens.empty()) {
        fprintf(stderr, "Error: tokenization produced empty token sequence\n");
        *n_samples_out = 0;
        return nullptr;
    }

    // Step 2: Build prompt with special tokens
    // Format: <|im_start|> text_tokens <|im_end|> <TTS_BOS>
    std::vector<int> prompt;
    prompt.push_back(IM_START_TOKEN_ID);
    for (int32_t tok : text_tokens) {
        prompt.push_back((int)tok);
    }
    prompt.push_back(IM_END_TOKEN_ID);
    prompt.push_back(TTS_BOS_TOKEN_ID);

    // Step 3: Generate codec tokens using LLM
    constexpr int MAX_TOKENS = 2048;
    int * generated_tokens = (int *)malloc(MAX_TOKENS * sizeof(int));
    if (!generated_tokens) {
        fprintf(stderr, "Error: failed to allocate token buffer\n");
        *n_samples_out = 0;
        return nullptr;
    }

    // Initialize RNG
    uint64_t rng_state = (seed < 0) ? (uint64_t)time(nullptr) : (uint64_t)seed;

    int n_generated = model::generate_tokens(
        ctx,
        prompt.data(),
        (int)prompt.size(),
        embed_weight,
        layer_weights,
        n_layers,
        norm_weight,
        lm_head_weight,
        MAX_TOKENS,
        temperature,
        top_k,
        top_p,
        CODEC_EOS_ID,
        &rng_state,
        generated_tokens
    );

    if (n_generated <= (int)prompt.size()) {
        fprintf(stderr, "Error: no tokens generated beyond prompt\n");
        free(generated_tokens);
        *n_samples_out = 0;
        return nullptr;
    }

    // Extract codec tokens (skip prompt, keep until EOS)
    int codec_start = (int)prompt.size();
    int codec_len = n_generated - codec_start;

    // Remove EOS token if present
    if (codec_len > 0 && generated_tokens[n_generated - 1] == CODEC_EOS_ID) {
        codec_len--;
    }

    if (codec_len <= 0) {
        fprintf(stderr, "Error: no codec tokens generated\n");
        free(generated_tokens);
        *n_samples_out = 0;
        return nullptr;
    }

    // Step 4: Convert to vocoder input format
    // Generated tokens are flat sequence, need to reshape to [16, seq_len]
    // For now, assume single codebook (semantic only) and replicate across 16 codebooks
    // TODO: Proper code predictor to generate all 16 codebooks
    int seq_len = codec_len;
    struct ggml_tensor * codes = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, seq_len, 16);
    int32_t * codes_data = (int32_t *)codes->data;

    // Copy semantic codebook to first row, zero out others
    for (int cb = 0; cb < 16; cb++) {
        for (int t = 0; t < seq_len; t++) {
            if (cb == 0) {
                // First codebook gets the generated tokens (offset by codec base)
                // Codec tokens start at vocab_offset (e.g., 151936)
                // We need to convert to 0-2047 range for codebook lookup
                constexpr int CODEC_VOCAB_START = 151936;
                int token = generated_tokens[codec_start + t];
                int code = token - CODEC_VOCAB_START;
                // Clamp to valid range
                if (code < 0) code = 0;
                if (code >= 2048) code = 2047;
                codes_data[cb * seq_len + t] = code;
            } else {
                // Other codebooks are zero (acoustic refinement not implemented)
                codes_data[cb * seq_len + t] = 0;
            }
        }
    }

    free(generated_tokens);

    // Step 5: Run vocoder to generate audio
    // Output audio length: seq_len * 480 (24kHz / 12Hz * 240 samples per token)
    // Actually, 480 = 8 * 5 * 4 * 3 = product of upsample rates
    constexpr int UPSAMPLE_FACTOR = 480;
    size_t audio_len = seq_len * UPSAMPLE_FACTOR;

    float * audio = (float *)malloc(audio_len * sizeof(float));
    if (!audio) {
        fprintf(stderr, "Error: failed to allocate audio buffer\n");
        *n_samples_out = 0;
        return nullptr;
    }

    struct ggml_tensor * audio_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, audio_len);
    audio_tensor->data = audio;

    // Upsample configuration
    int kernel_sizes[] = {16, 10, 8, 6};
    int paddings[] = {7, 4, 3, 2};

    vocoder::vocoder_decode(
        audio_tensor,
        codes,
        codebooks,
        (const struct ggml_tensor **)upsample_weights,
        (const struct ggml_tensor **)upsample_alphas,
        (const struct ggml_tensor **)upsample_betas,
        kernel_sizes,
        paddings
    );

    // Step 6: Return audio samples
    *n_samples_out = audio_len;
    return audio;
}
