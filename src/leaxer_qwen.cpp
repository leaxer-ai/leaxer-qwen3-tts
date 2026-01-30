// leaxer-qwen3-tts: Pure C++ Qwen3-TTS implementation
// Main CLI entry point

#include "leaxer_qwen.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

static void print_speakers(void) {
    printf("Available speakers for CustomVoice model:\n");
    printf("  aiden   (default)\n");
    printf("  ryan\n");
    printf("  serena\n");
    printf("  vivian\n");
    printf("  aria\n");
    printf("  emma\n");
    printf("  sophia\n");
}

static void print_usage(const char * progname) {
    printf("Usage: %s [options]\n", progname);
    printf("\n");
    printf("Options:\n");
    printf("  -m, --model PATH      Path to GGUF model file (required)\n");
    printf("  -p, --prompt TEXT     Text to synthesize (required)\n");
    printf("  -o, --output PATH     Output WAV file (default: output.wav)\n");
    printf("  -t, --threads N       Number of threads (default: 4)\n");
    printf("  --speaker NAME        Speaker voice (default: aiden)\n");
    printf("  --list-speakers       List available speakers and exit\n");
    printf("  --temp FLOAT          Temperature (default: 0.9)\n");
    printf("  --top-k N             Top-k sampling (default: 50)\n");
    printf("  --top-p FLOAT         Top-p sampling (default: 0.95)\n");
    printf("  --seed N              Random seed (default: -1 for random)\n");
    printf("  -v, --version         Print version and exit\n");
    printf("  -h, --help            Print this help and exit\n");
    printf("\n");
    printf("Example:\n");
    printf("  %s -m qwen3-tts-1.7b.gguf -p \"Hello world\" -o hello.wav\n", progname);
    printf("  %s -m qwen3-tts-1.7b.gguf --speaker ryan -p \"Hello world\"\n", progname);
}

int main(int argc, char ** argv) {
    // Parse arguments
    const char * model_path = nullptr;
    const char * prompt = nullptr;
    const char * output_path = "output.wav";
    const char * speaker = "aiden";
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
            printf("leaxer-qwen3-tts version %s\n", leaxer_qwen_version());
            leaxer_qwen_print_system_info();
            return 0;
        }
        if (arg == "--list-speakers") {
            print_speakers();
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
        } else if (arg == "--speaker" && i + 1 < argc) {
            speaker = argv[++i];
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

    printf("leaxer-qwen3-tts v%s\n", leaxer_qwen_version());
    printf("Model: %s\n", model_path);
    printf("Prompt: %s\n", prompt);
    printf("Speaker: %s\n", speaker);
    printf("Output: %s\n", output_path);
    printf("Threads: %d\n", n_threads);
    printf("\n");

    // Load model
    struct leaxer_qwen_model_params model_params = leaxer_qwen_model_default_params();
    model_params.n_threads = n_threads;

    struct leaxer_qwen_model * model = leaxer_qwen_load_model(model_path, model_params);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    // Create context
    struct leaxer_qwen_context * ctx = leaxer_qwen_new_context(model);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        leaxer_qwen_free_model(model);
        return 1;
    }

    // Generate audio
    printf("\nGenerating speech...\n");
    struct leaxer_qwen_gen_params gen_params = leaxer_qwen_gen_default_params();
    gen_params.temperature = temperature;
    gen_params.top_k = top_k;
    gen_params.top_p = top_p;
    gen_params.seed = seed;
    gen_params.speaker = speaker;

    size_t n_samples = 0;
    float * audio = leaxer_qwen_generate(ctx, prompt, gen_params, &n_samples);
    if (!audio) {
        fprintf(stderr, "Failed to generate audio\n");
        leaxer_qwen_free_context(ctx);
        leaxer_qwen_free_model(model);
        return 1;
    }

    printf("Generated %zu audio samples\n", n_samples);

    // Write WAV file
    printf("Writing WAV file: %s\n", output_path);
    int write_result = leaxer_qwen_write_wav(output_path, audio, n_samples, 24000);
    if (write_result != 0) {
        fprintf(stderr, "Failed to write WAV file\n");
        leaxer_qwen_free_audio(audio);
        leaxer_qwen_free_context(ctx);
        leaxer_qwen_free_model(model);
        return 1;
    }

    // Cleanup
    leaxer_qwen_free_audio(audio);
    leaxer_qwen_free_context(ctx);
    leaxer_qwen_free_model(model);

    printf("\nSuccess! Audio saved to: %s\n", output_path);
    return 0;
}

// API implementations

const char * leaxer_qwen_version(void) {
    return LEAXER_QWEN_VERSION;
}

void leaxer_qwen_print_system_info(void) {
    printf("\nSystem Information:\n");
    printf("  Version: %s\n", LEAXER_QWEN_VERSION);

    // Backend detection
    printf("  Backend: ");
#if defined(GGML_USE_CUDA)
    printf("CUDA");
#elif defined(GGML_USE_METAL)
    printf("Metal");
#elif defined(GGML_USE_VULKAN)
    printf("Vulkan");
#elif defined(GGML_USE_SYCL)
    printf("SYCL");
#elif defined(GGML_USE_OPENCL)
    printf("OpenCL");
#else
    printf("CPU");
#endif
    printf("\n");

    // SIMD support detection
    printf("  SIMD Support:");
#if defined(__AVX512F__)
    printf(" AVX512");
#endif
#if defined(__AVX2__)
    printf(" AVX2");
#endif
#if defined(__AVX__)
    printf(" AVX");
#endif
#if defined(__SSE4_2__)
    printf(" SSE4.2");
#endif
#if defined(__SSE4_1__)
    printf(" SSE4.1");
#endif
#if defined(__SSSE3__)
    printf(" SSSE3");
#endif
#if defined(__SSE3__)
    printf(" SSE3");
#endif
#if defined(__SSE2__)
    printf(" SSE2");
#endif
#if defined(__ARM_NEON)
    printf(" NEON");
#endif
#if defined(__ARM_FEATURE_FMA)
    printf(" FMA");
#endif
#if defined(__wasm_simd128__)
    printf(" WASM-SIMD");
#endif
#if !defined(__AVX512F__) && !defined(__AVX2__) && !defined(__AVX__) && \
    !defined(__SSE4_2__) && !defined(__SSE4_1__) && !defined(__SSSE3__) && \
    !defined(__SSE3__) && !defined(__SSE2__) && !defined(__ARM_NEON) && \
    !defined(__wasm_simd128__)
    printf(" None");
#endif
    printf("\n");
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
        .speaker     = "aiden",
    };
}

#include "model/model_weights.h"

// Forward declaration of tts_generate (defined at end of file)
float * tts_generate(
    struct ggml_context * ctx,
    const char * text,
    struct ggml_tensor * embed_weight,
    struct ggml_tensor * talker_codec_embedding,  // CRITICAL: Talker's codec embedding for input combination
    struct ggml_tensor * text_proj_fc1_weight,
    struct ggml_tensor * text_proj_fc1_bias,
    struct ggml_tensor * text_proj_fc2_weight,
    struct ggml_tensor * text_proj_fc2_bias,
    struct ggml_tensor ** layer_weights,
    int n_layers,
    struct ggml_tensor * norm_weight,
    struct ggml_tensor * lm_head_weight,
    struct ggml_tensor ** codec_embeddings,  // Code predictor's embeddings (15 tables)
    struct ggml_tensor ** code_pred_layer_weights,
    struct ggml_tensor * code_pred_norm_weight,
    struct ggml_tensor ** code_pred_output_heads,
    const leaxer_qwen::model::VocoderWeights * vocoder_weights,
    float temperature,
    int top_k,
    float top_p,
    int seed,
    size_t * n_samples_out);

// Forward declarations
namespace leaxer_qwen {
namespace io {
bool load_talker_weights(const char * gguf_path, struct leaxer_qwen::model::TalkerWeights * weights, struct ggml_context * ctx);
bool load_code_predictor_weights(const char * gguf_path, struct leaxer_qwen::model::CodePredictorWeights * weights, struct ggml_context * ctx);
bool load_vocoder_weights(const char * gguf_path, struct leaxer_qwen::model::VocoderWeights * weights, struct ggml_context * ctx);
}

namespace model {
// Full version with talker hidden states (CORRECT)
struct ggml_tensor * code_predictor_forward(
    struct ggml_context * ctx,
    struct ggml_tensor * codebook_0_tokens,
    struct ggml_tensor ** codec_embeddings,
    struct ggml_tensor ** layer_weights,
    struct ggml_tensor * output_norm_weight,
    struct ggml_tensor ** output_heads,
    int hidden_dim,
    int seq_len,
    struct ggml_tensor * talker_hidden_states,   // [seq_len, hidden_dim] from talker
    struct ggml_tensor * talker_codec_embedding); // talker's codec embedding

// Legacy version without hidden states (produces poor results)
struct ggml_tensor * code_predictor_forward(
    struct ggml_context * ctx,
    struct ggml_tensor * semantic_codes,
    struct ggml_tensor ** codec_embeddings,
    struct ggml_tensor ** layer_weights,
    struct ggml_tensor * output_norm_weight,
    struct ggml_tensor ** output_heads,
    int hidden_dim,
    int seq_len);
}
}

// Internal model structure
struct leaxer_qwen_model {
    struct ggml_context * ctx;
    leaxer_qwen::model::TalkerWeights talker;
    leaxer_qwen::model::CodePredictorWeights code_predictor;
    leaxer_qwen::model::VocoderWeights vocoder;
    bool vocoder_loaded;
    int n_threads;
};

// Internal context structure
struct leaxer_qwen_context {
    struct leaxer_qwen_model * model;
    struct ggml_context * ctx;
};

// Implementation

struct leaxer_qwen_model * leaxer_qwen_load_model(
    const char * path,
    struct leaxer_qwen_model_params params
) {
    if (!path) {
        fprintf(stderr, "Error: model path is null\n");
        return nullptr;
    }

    printf("Loading model from: %s\n", path);

    // Allocate model struct
    struct leaxer_qwen_model * model = (struct leaxer_qwen_model *)calloc(1, sizeof(struct leaxer_qwen_model));
    if (!model) {
        fprintf(stderr, "Error: failed to allocate model struct\n");
        return nullptr;
    }

    model->n_threads = params.n_threads;

    // Create ggml context for model weights
    // 0.6B model requires ~2GB for weights
    size_t mem_size = 3ULL * 1024 * 1024 * 1024;  // 3GB
    struct ggml_init_params ggml_params = {
        .mem_size   = mem_size,
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };
    model->ctx = ggml_init(ggml_params);
    if (!model->ctx) {
        fprintf(stderr, "Error: failed to initialize ggml context\n");
        free(model);
        return nullptr;
    }

    // Load model weights from GGUF files
    // For now, assume all weights are in a single file
    // In production, talker and vocoder might be separate files

    printf("Loading talker weights...\n");
    if (!leaxer_qwen::io::load_talker_weights(path, &model->talker, model->ctx)) {
        fprintf(stderr, "Error: failed to load talker weights\n");
        ggml_free(model->ctx);
        free(model);
        return nullptr;
    }

    printf("Loading code predictor weights...\n");
    if (!leaxer_qwen::io::load_code_predictor_weights(path, &model->code_predictor, model->ctx)) {
        fprintf(stderr, "Error: failed to load code predictor weights\n");
        ggml_free(model->ctx);
        free(model);
        return nullptr;
    }

    // Try to load vocoder from same file, or look for separate vocoder file
    // Vocoder is optional - pipeline can output codec tokens without it
    printf("Loading vocoder weights...\n");
    model->vocoder_loaded = false;
    if (!leaxer_qwen::io::load_vocoder_weights(path, &model->vocoder, model->ctx)) {
        // Try vocoder.gguf in same directory
        std::string path_str(path);
        size_t last_slash = path_str.find_last_of("/\\");
        std::string vocoder_path;
        if (last_slash != std::string::npos) {
            vocoder_path = path_str.substr(0, last_slash + 1) + "vocoder.gguf";
        } else {
            vocoder_path = "vocoder.gguf";
        }

        if (!leaxer_qwen::io::load_vocoder_weights(vocoder_path.c_str(), &model->vocoder, model->ctx)) {
            fprintf(stderr, "Warning: vocoder not loaded - will output codec tokens only\n");
        } else {
            model->vocoder_loaded = true;
        }
    } else {
        model->vocoder_loaded = true;
    }

    printf("Model loaded successfully\n");

    // Print model info
    printf("\nModel Info:\n");
    printf("  Architecture: Qwen3-TTS 0.6B\n");
    printf("  Talker layers: 28\n");
    printf("  Code predictor layers: 5\n");
    printf("  Vocoder: %s\n", model->vocoder_loaded ? "loaded" : "not loaded");

    return model;
}

void leaxer_qwen_free_model(struct leaxer_qwen_model * model) {
    if (!model) return;
    if (model->ctx) {
        ggml_free(model->ctx);
    }
    free(model);
}

struct leaxer_qwen_context * leaxer_qwen_new_context(
    struct leaxer_qwen_model * model
) {
    if (!model) {
        fprintf(stderr, "Error: model is null\n");
        return nullptr;
    }

    struct leaxer_qwen_context * ctx = (struct leaxer_qwen_context *)calloc(1, sizeof(struct leaxer_qwen_context));
    if (!ctx) {
        fprintf(stderr, "Error: failed to allocate context struct\n");
        return nullptr;
    }

    ctx->model = model;

    // Create ggml context for inference (temporary tensors)
    // Need enough memory for activations during forward pass
    // Processing one codebook at a time keeps memory reasonable
    size_t mem_size = 8ULL * 1024 * 1024 * 1024;  // 8GB for activations
    struct ggml_init_params ggml_params = {
        .mem_size   = mem_size,
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };
    ctx->ctx = ggml_init(ggml_params);
    if (!ctx->ctx) {
        fprintf(stderr, "Error: failed to initialize ggml context for inference\n");
        free(ctx);
        return nullptr;
    }

    return ctx;
}

void leaxer_qwen_free_context(struct leaxer_qwen_context * ctx) {
    if (!ctx) return;
    if (ctx->ctx) {
        ggml_free(ctx->ctx);
    }
    free(ctx);
}

float * leaxer_qwen_generate(
    struct leaxer_qwen_context * ctx,
    const char * text,
    struct leaxer_qwen_gen_params params,
    size_t * n_samples
) {
    if (!ctx || !text || !n_samples) {
        fprintf(stderr, "Error: invalid parameters to leaxer_qwen_generate\n");
        if (n_samples) *n_samples = 0;
        return nullptr;
    }

    // Prepare layer weights array for tts_generate
    // Each layer needs 11 weight tensors (now includes Q/K normalization)
    // Layout: attn_norm, q, k, v, o, q_norm, k_norm, ffn_norm, gate, up, down
    struct ggml_tensor ** layer_weights = (struct ggml_tensor **)malloc(28 * 11 * sizeof(struct ggml_tensor *));
    if (!layer_weights) {
        fprintf(stderr, "Error: failed to allocate layer weights array\n");
        *n_samples = 0;
        return nullptr;
    }

    for (int i = 0; i < 28; i++) {
        layer_weights[i * 11 + 0] = ctx->model->talker.layers[i].in_ln_weight;
        layer_weights[i * 11 + 1] = ctx->model->talker.layers[i].attn_q_proj_weight;
        layer_weights[i * 11 + 2] = ctx->model->talker.layers[i].attn_k_proj_weight;
        layer_weights[i * 11 + 3] = ctx->model->talker.layers[i].attn_v_proj_weight;
        layer_weights[i * 11 + 4] = ctx->model->talker.layers[i].attn_o_proj_weight;
        layer_weights[i * 11 + 5] = ctx->model->talker.layers[i].attn_q_norm_weight;  // Q norm (Qwen3)
        layer_weights[i * 11 + 6] = ctx->model->talker.layers[i].attn_k_norm_weight;  // K norm (Qwen3)
        layer_weights[i * 11 + 7] = ctx->model->talker.layers[i].post_ln_weight;
        layer_weights[i * 11 + 8] = ctx->model->talker.layers[i].ffn_gate_proj_weight;
        layer_weights[i * 11 + 9] = ctx->model->talker.layers[i].ffn_up_proj_weight;
        layer_weights[i * 11 + 10] = ctx->model->talker.layers[i].ffn_down_proj_weight;
    }

    // Prepare code predictor layer weights (also 11 per layer with Q/K norms)
    struct ggml_tensor ** code_pred_layer_weights = (struct ggml_tensor **)malloc(5 * 11 * sizeof(struct ggml_tensor *));
    if (!code_pred_layer_weights) {
        fprintf(stderr, "Error: failed to allocate code predictor layer weights array\n");
        free(layer_weights);
        *n_samples = 0;
        return nullptr;
    }

    for (int i = 0; i < 5; i++) {
        code_pred_layer_weights[i * 11 + 0] = ctx->model->code_predictor.layers[i].in_ln_weight;
        code_pred_layer_weights[i * 11 + 1] = ctx->model->code_predictor.layers[i].attn_q_proj_weight;
        code_pred_layer_weights[i * 11 + 2] = ctx->model->code_predictor.layers[i].attn_k_proj_weight;
        code_pred_layer_weights[i * 11 + 3] = ctx->model->code_predictor.layers[i].attn_v_proj_weight;
        code_pred_layer_weights[i * 11 + 4] = ctx->model->code_predictor.layers[i].attn_o_proj_weight;
        code_pred_layer_weights[i * 11 + 5] = ctx->model->code_predictor.layers[i].attn_q_norm_weight;  // Q norm
        code_pred_layer_weights[i * 11 + 6] = ctx->model->code_predictor.layers[i].attn_k_norm_weight;  // K norm
        code_pred_layer_weights[i * 11 + 7] = ctx->model->code_predictor.layers[i].post_ln_weight;
        code_pred_layer_weights[i * 11 + 8] = ctx->model->code_predictor.layers[i].ffn_gate_proj_weight;
        code_pred_layer_weights[i * 11 + 9] = ctx->model->code_predictor.layers[i].ffn_up_proj_weight;
        code_pred_layer_weights[i * 11 + 10] = ctx->model->code_predictor.layers[i].ffn_down_proj_weight;
    }

    // Check if vocoder is loaded
    if (!ctx->model->vocoder_loaded) {
        printf("Warning: vocoder not loaded, generating test tone instead of speech\n");
        // Generate a 1-second 440Hz test tone to verify pipeline
        constexpr int SAMPLE_RATE = 24000;
        constexpr int DURATION_SAMPLES = SAMPLE_RATE;  // 1 second
        float * audio = (float *)malloc(DURATION_SAMPLES * sizeof(float));
        if (!audio) {
            fprintf(stderr, "Error: failed to allocate audio buffer\n");
            free(code_pred_layer_weights);
            free(layer_weights);
            *n_samples = 0;
            return nullptr;
        }
        // Generate 440Hz sine wave
        for (int i = 0; i < DURATION_SAMPLES; i++) {
            audio[i] = 0.3f * sinf(2.0f * 3.14159265f * 440.0f * i / SAMPLE_RATE);
        }
        free(code_pred_layer_weights);
        free(layer_weights);
        *n_samples = DURATION_SAMPLES;
        return audio;
    }

    // Call tts_generate with vocoder weights struct
    // CRITICAL: Pass talker's codec_embedding_weight for proper input combination
    // Input to talker should be: text_projection(text_embed) + codec_embed
    float * audio = tts_generate(
        ctx->ctx,
        text,
        ctx->model->talker.emb_weight,
        ctx->model->talker.codec_embedding_weight,  // Talker's codec embedding (3072 × 1024)
        ctx->model->talker.text_proj_fc1_weight,
        ctx->model->talker.text_proj_fc1_bias,
        ctx->model->talker.text_proj_fc2_weight,
        ctx->model->talker.text_proj_fc2_bias,
        layer_weights,
        28,  // n_layers
        ctx->model->talker.norm_weight,
        ctx->model->talker.lm_head_weight,
        ctx->model->code_predictor.codec_embeddings,  // Code predictor's embeddings (15 tables)
        code_pred_layer_weights,
        ctx->model->code_predictor.norm_weight,
        ctx->model->code_predictor.output_heads,
        &ctx->model->vocoder,
        params.temperature,
        params.top_k,
        params.top_p,
        params.seed,
        n_samples
    );

    free(code_pred_layer_weights);
    free(layer_weights);

    return audio;
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
// NEW: Interleaved generation (CORRECT implementation)
// Generates all 16 codebooks properly by interleaving talker and code predictor
int generate_interleaved(
    const int * text_tokens,
    const int * codec_tokens,
    int prefill_len,
    struct ggml_tensor * text_embed_weight,
    struct ggml_tensor * text_proj_fc1_weight,
    struct ggml_tensor * text_proj_fc1_bias,
    struct ggml_tensor * text_proj_fc2_weight,
    struct ggml_tensor * text_proj_fc2_bias,
    struct ggml_tensor * talker_codec_embedding,
    struct ggml_tensor ** talker_layer_weights,
    int n_talker_layers,
    struct ggml_tensor * talker_norm_weight,
    struct ggml_tensor * talker_lm_head_weight,
    struct ggml_tensor ** codec_embeddings,
    struct ggml_tensor ** code_pred_layer_weights,
    struct ggml_tensor * code_pred_norm_weight,
    struct ggml_tensor ** code_pred_output_heads,
    const float * tts_pad_embed,
    int max_frames,
    float temperature,
    int top_k,
    float top_p,
    uint64_t * rng_state,
    int32_t * all_codes_out,
    int * n_frames_out);
}

namespace vocoder {
void vocoder_full_forward(
    float * audio_out,
    const int32_t * codes,
    int64_t seq_len,
    const model::VocoderWeights * weights);
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
// Connects: text → tokenize → LLM → code predictor → vocoder → audio
//
// CRITICAL ARCHITECTURE NOTES (from Python reference):
// 1. Talker input is SUMMED, not concatenated: text_projection(text_embed) + codec_embed
// 2. Code predictor input: concat([talker_hidden_state, codebook_0_embed], dim=1) → [B, 2, 1024]
// 3. After generating all 16 codebooks, their embeddings are SUMMED for next step
//
// Parameters:
//   ctx: ggml context with sufficient memory
//   text: input text to synthesize
//   embed_weight: text token embedding matrix [vocab_size=151936, embedding_dim=2048]
//   talker_codec_embedding: talker's codec embedding [codec_vocab=3072, hidden_dim=1024]
//   text_proj_*: text projection MLP weights (2048 → 2048 → SiLU → 1024)
//   layer_weights: array of pointers to transformer layer weights
//   n_layers: number of transformer layers
//   norm_weight: final layer norm weight
//   lm_head_weight: output projection weight [hidden_dim=1024, codec_vocab=3072]
//   codec_embeddings: code predictor's embeddings (15 tables, indices 0-14)
//   code_pred_layer_weights: code predictor transformer layer weights
//   code_pred_norm_weight: code predictor final layer norm
//   code_pred_output_heads: code predictor output projection heads [15]
//   vocoder_weights: full vocoder weights struct (codebooks, projections, transformer, upsample)
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
    struct ggml_tensor * talker_codec_embedding,  // Talker's codec embedding (3072 × 1024)
    struct ggml_tensor * text_proj_fc1_weight,
    struct ggml_tensor * text_proj_fc1_bias,
    struct ggml_tensor * text_proj_fc2_weight,
    struct ggml_tensor * text_proj_fc2_bias,
    struct ggml_tensor ** layer_weights,
    int n_layers,
    struct ggml_tensor * norm_weight,
    struct ggml_tensor * lm_head_weight,
    struct ggml_tensor ** codec_embeddings,  // Code predictor's embeddings (15 tables)
    struct ggml_tensor ** code_pred_layer_weights,
    struct ggml_tensor * code_pred_norm_weight,
    struct ggml_tensor ** code_pred_output_heads,
    const leaxer_qwen::model::VocoderWeights * vocoder_weights,
    float temperature,
    int top_k,
    float top_p,
    int seed,
    size_t * n_samples_out
) {
    using namespace leaxer_qwen;

    // Model dimensions
    constexpr int HIDDEN_DIM = 1024;
    constexpr int NUM_CODEBOOKS = 16;

    // Special token IDs
    constexpr int IM_START_TOKEN_ID = 151644;
    constexpr int TTS_PAD_TOKEN_ID = 151671;
    constexpr int TTS_EOS_TOKEN_ID = 151673;
    
    // Codec special tokens
    constexpr int CODEC_PAD_ID = 2148;
    constexpr int CODEC_BOS_ID = 2149;
    constexpr int CODEC_NOTHINK_ID = 2155;
    constexpr int CODEC_THINK_BOS_ID = 2156;
    constexpr int CODEC_THINK_EOS_ID = 2157;

    // Step 1: Tokenize input text
    std::string text_str(text);
    std::vector<int32_t> text_tokens = io::tokenize(text_str);

    if (text_tokens.empty()) {
        fprintf(stderr, "Error: tokenization produced empty token sequence\n");
        *n_samples_out = 0;
        return nullptr;
    }

    // Estimate frame count
    int estimated_max_frames = (int)text_tokens.size() * 20 + 50;
    if (estimated_max_frames > 512) estimated_max_frames = 512;
    printf("Text tokens: %zu, estimated max frames: %d\n", text_tokens.size(), estimated_max_frames);

    // Step 2: Build PARALLEL text and codec prompt sequences
    std::vector<int> text_prompt;
    std::vector<int> codec_prompt;
    
    // Codec control prefix
    text_prompt.push_back(IM_START_TOKEN_ID);
    codec_prompt.push_back(CODEC_NOTHINK_ID);
    text_prompt.push_back(TTS_PAD_TOKEN_ID);
    codec_prompt.push_back(CODEC_THINK_BOS_ID);
    text_prompt.push_back(TTS_PAD_TOKEN_ID);
    codec_prompt.push_back(CODEC_THINK_EOS_ID);
    
    // Text content with codec_pad
    for (int32_t tok : text_tokens) {
        text_prompt.push_back((int)tok);
        codec_prompt.push_back(CODEC_PAD_ID);
    }
    
    // End: tts_eos + pad, then tts_pad + codec_bos
    text_prompt.push_back(TTS_EOS_TOKEN_ID);
    codec_prompt.push_back(CODEC_PAD_ID);
    text_prompt.push_back(TTS_PAD_TOKEN_ID);
    codec_prompt.push_back(CODEC_BOS_ID);

    // Step 3: Pre-compute tts_pad_embed
    float * tts_pad_embed = (float *)malloc(HIDDEN_DIM * sizeof(float));
    if (!tts_pad_embed) {
        fprintf(stderr, "Error: failed to allocate tts_pad_embed\n");
        *n_samples_out = 0;
        return nullptr;
    }
    
    {
        size_t embed_ctx_size = 64 * 1024 * 1024;
        struct ggml_init_params params = { embed_ctx_size, nullptr, false };
        struct ggml_context * embed_ctx = ggml_init(params);
        if (!embed_ctx) {
            free(tts_pad_embed);
            *n_samples_out = 0;
            return nullptr;
        }
        
        struct ggml_tensor * pad_token = ggml_new_tensor_1d(embed_ctx, GGML_TYPE_I32, 1);
        ((int32_t *)pad_token->data)[0] = TTS_PAD_TOKEN_ID;
        
        struct ggml_tensor * text_embed = ggml_get_rows(embed_ctx, embed_weight, pad_token);
        struct ggml_tensor * proj = ggml_mul_mat(embed_ctx, text_proj_fc1_weight, text_embed);
        struct ggml_tensor * bias1 = (text_proj_fc1_bias->type == GGML_TYPE_F16) 
            ? ggml_cast(embed_ctx, text_proj_fc1_bias, GGML_TYPE_F32) : text_proj_fc1_bias;
        proj = ggml_add(embed_ctx, proj, bias1);
        proj = ggml_silu(embed_ctx, proj);
        proj = ggml_mul_mat(embed_ctx, text_proj_fc2_weight, proj);
        struct ggml_tensor * bias2 = (text_proj_fc2_bias->type == GGML_TYPE_F16)
            ? ggml_cast(embed_ctx, text_proj_fc2_bias, GGML_TYPE_F32) : text_proj_fc2_bias;
        proj = ggml_add(embed_ctx, proj, bias2);
        
        struct ggml_cgraph * graph = ggml_new_graph(embed_ctx);
        ggml_build_forward_expand(graph, proj);
        ggml_graph_compute_with_ctx(embed_ctx, graph, 1);
        memcpy(tts_pad_embed, proj->data, HIDDEN_DIM * sizeof(float));
        ggml_free(embed_ctx);
    }

    // Step 4: Allocate output buffer
    int32_t * all_codes = (int32_t *)malloc(estimated_max_frames * NUM_CODEBOOKS * sizeof(int32_t));
    if (!all_codes) {
        free(tts_pad_embed);
        *n_samples_out = 0;
        return nullptr;
    }

    // Initialize RNG
    uint64_t rng_state = (seed < 0) ? (uint64_t)time(nullptr) : (uint64_t)seed;

    // Step 5: Run INTERLEAVED generation (correct flow!)
    printf("Running INTERLEAVED generation...\n");
    int n_frames = 0;
    int result = model::generate_interleaved(
        text_prompt.data(),
        codec_prompt.data(),
        (int)text_prompt.size(),
        embed_weight,
        text_proj_fc1_weight, text_proj_fc1_bias,
        text_proj_fc2_weight, text_proj_fc2_bias,
        talker_codec_embedding,
        layer_weights,
        n_layers,
        norm_weight,
        lm_head_weight,
        codec_embeddings,
        code_pred_layer_weights,
        code_pred_norm_weight,
        code_pred_output_heads,
        tts_pad_embed,
        estimated_max_frames,
        temperature, top_k, top_p,
        &rng_state,
        all_codes,
        &n_frames
    );

    free(tts_pad_embed);

    if (result != 0 || n_frames <= 0) {
        fprintf(stderr, "Error: interleaved generation failed\n");
        free(all_codes);
        *n_samples_out = 0;
        return nullptr;
    }

    printf("Generated %d frames × %d codebooks.\n", n_frames, NUM_CODEBOOKS);

    // Step 6: Create codes tensor for vocoder
    struct ggml_tensor * codes = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, NUM_CODEBOOKS, n_frames);
    memcpy(codes->data, all_codes, n_frames * NUM_CODEBOOKS * sizeof(int32_t));
    free(all_codes);

    // Step 7: Run vocoder
    constexpr int UPSAMPLE_FACTOR = 1920;
    size_t audio_len = n_frames * UPSAMPLE_FACTOR;

    float * audio = (float *)malloc(audio_len * sizeof(float));
    if (!audio) {
        *n_samples_out = 0;
        return nullptr;
    }

    printf("Running vocoder (n_frames=%d, audio_len=%zu)...\n", n_frames, audio_len);
    vocoder::vocoder_full_forward(audio, (const int32_t *)codes->data, n_frames, vocoder_weights);

    *n_samples_out = audio_len;
    return audio;
}
