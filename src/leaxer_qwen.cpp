// leaxer-qwen: Pure C++ Qwen3-TTS implementation
// Main CLI entry point

#include "leaxer_qwen.h"
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
            printf("leaxer-qwen version %s\n", leaxer_qwen_version());
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

    printf("leaxer-qwen v%s\n", leaxer_qwen_version());
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
    struct ggml_tensor ** layer_weights,
    int n_layers,
    struct ggml_tensor * norm_weight,
    struct ggml_tensor * lm_head_weight,
    struct ggml_tensor * codec_embedding,
    struct ggml_tensor ** code_pred_layer_weights,
    struct ggml_tensor * code_pred_norm_weight,
    struct ggml_tensor ** code_pred_output_heads,
    struct ggml_tensor * codebooks,
    struct ggml_tensor ** upsample_weights,
    struct ggml_tensor ** upsample_alphas,
    struct ggml_tensor ** upsample_betas,
    struct ggml_tensor * final_conv_weight,
    struct ggml_tensor * final_conv_bias,
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
    printf("Loading vocoder weights...\n");
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

        printf("Trying separate vocoder file: %s\n", vocoder_path.c_str());
        if (!leaxer_qwen::io::load_vocoder_weights(vocoder_path.c_str(), &model->vocoder, model->ctx)) {
            fprintf(stderr, "Error: failed to load vocoder weights\n");
            ggml_free(model->ctx);
            free(model);
            return nullptr;
        }
    }

    printf("Model loaded successfully\n");

    // Print model info
    printf("\nModel Info:\n");
    printf("  Architecture: Qwen3-TTS 0.6B\n");
    printf("  Talker layers: 28 (hidden_dim=1024, intermediate=3072)\n");
    printf("  Code predictor layers: 5\n");
    printf("  Attention: GQA (16 query heads, 8 KV heads)\n");
    printf("  Codebooks: 16 (2048 codes each)\n");
    printf("  Vocab size: ~152k tokens\n");

    // Count parameters
    size_t total_params = 0;

    // Talker parameters
    size_t talker_params = 0;
    if (model->talker.emb_weight) {
        talker_params += ggml_nelements(model->talker.emb_weight);
    }
    for (int i = 0; i < 28; i++) {
        if (model->talker.layers[i].in_ln_weight) talker_params += ggml_nelements(model->talker.layers[i].in_ln_weight);
        if (model->talker.layers[i].attn_q_proj_weight) talker_params += ggml_nelements(model->talker.layers[i].attn_q_proj_weight);
        if (model->talker.layers[i].attn_k_proj_weight) talker_params += ggml_nelements(model->talker.layers[i].attn_k_proj_weight);
        if (model->talker.layers[i].attn_v_proj_weight) talker_params += ggml_nelements(model->talker.layers[i].attn_v_proj_weight);
        if (model->talker.layers[i].attn_o_proj_weight) talker_params += ggml_nelements(model->talker.layers[i].attn_o_proj_weight);
        if (model->talker.layers[i].post_ln_weight) talker_params += ggml_nelements(model->talker.layers[i].post_ln_weight);
        if (model->talker.layers[i].ffn_gate_proj_weight) talker_params += ggml_nelements(model->talker.layers[i].ffn_gate_proj_weight);
        if (model->talker.layers[i].ffn_up_proj_weight) talker_params += ggml_nelements(model->talker.layers[i].ffn_up_proj_weight);
        if (model->talker.layers[i].ffn_down_proj_weight) talker_params += ggml_nelements(model->talker.layers[i].ffn_down_proj_weight);
    }
    if (model->talker.norm_weight) talker_params += ggml_nelements(model->talker.norm_weight);
    if (model->talker.lm_head_weight) talker_params += ggml_nelements(model->talker.lm_head_weight);

    // Code predictor parameters
    size_t code_pred_params = 0;
    if (model->code_predictor.codec_embedding_weight) {
        code_pred_params += ggml_nelements(model->code_predictor.codec_embedding_weight);
    }
    for (int i = 0; i < 5; i++) {
        if (model->code_predictor.layers[i].in_ln_weight) code_pred_params += ggml_nelements(model->code_predictor.layers[i].in_ln_weight);
        if (model->code_predictor.layers[i].attn_q_proj_weight) code_pred_params += ggml_nelements(model->code_predictor.layers[i].attn_q_proj_weight);
        if (model->code_predictor.layers[i].attn_k_proj_weight) code_pred_params += ggml_nelements(model->code_predictor.layers[i].attn_k_proj_weight);
        if (model->code_predictor.layers[i].attn_v_proj_weight) code_pred_params += ggml_nelements(model->code_predictor.layers[i].attn_v_proj_weight);
        if (model->code_predictor.layers[i].attn_o_proj_weight) code_pred_params += ggml_nelements(model->code_predictor.layers[i].attn_o_proj_weight);
        if (model->code_predictor.layers[i].post_ln_weight) code_pred_params += ggml_nelements(model->code_predictor.layers[i].post_ln_weight);
        if (model->code_predictor.layers[i].ffn_gate_proj_weight) code_pred_params += ggml_nelements(model->code_predictor.layers[i].ffn_gate_proj_weight);
        if (model->code_predictor.layers[i].ffn_up_proj_weight) code_pred_params += ggml_nelements(model->code_predictor.layers[i].ffn_up_proj_weight);
        if (model->code_predictor.layers[i].ffn_down_proj_weight) code_pred_params += ggml_nelements(model->code_predictor.layers[i].ffn_down_proj_weight);
    }
    if (model->code_predictor.norm_weight) code_pred_params += ggml_nelements(model->code_predictor.norm_weight);
    for (int i = 0; i < 16; i++) {
        if (model->code_predictor.output_heads[i]) code_pred_params += ggml_nelements(model->code_predictor.output_heads[i]);
    }

    // Vocoder parameters
    size_t vocoder_params = 0;
    if (model->vocoder.codebooks) vocoder_params += ggml_nelements(model->vocoder.codebooks);
    if (model->vocoder.causal_conv_weight) vocoder_params += ggml_nelements(model->vocoder.causal_conv_weight);
    if (model->vocoder.causal_conv_bias) vocoder_params += ggml_nelements(model->vocoder.causal_conv_bias);
    for (int i = 0; i < 4; i++) {
        if (model->vocoder.upsample_weights[i]) vocoder_params += ggml_nelements(model->vocoder.upsample_weights[i]);
        if (model->vocoder.upsample_biases[i]) vocoder_params += ggml_nelements(model->vocoder.upsample_biases[i]);
        if (model->vocoder.upsample_alphas[i]) vocoder_params += ggml_nelements(model->vocoder.upsample_alphas[i]);
        if (model->vocoder.upsample_betas[i]) vocoder_params += ggml_nelements(model->vocoder.upsample_betas[i]);
    }
    if (model->vocoder.final_conv_weight) vocoder_params += ggml_nelements(model->vocoder.final_conv_weight);
    if (model->vocoder.final_conv_bias) vocoder_params += ggml_nelements(model->vocoder.final_conv_bias);

    total_params = talker_params + code_pred_params + vocoder_params;

    printf("  Parameters:\n");
    printf("    Talker: %.1fM\n", talker_params / 1e6);
    printf("    Code Predictor: %.1fM\n", code_pred_params / 1e6);
    printf("    Vocoder: %.1fM\n", vocoder_params / 1e6);
    printf("    Total: %.1fM\n", total_params / 1e6);
    printf("\n");

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
    size_t mem_size = 2ULL * 1024 * 1024 * 1024;  // 2GB for activations
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
    // Each layer needs 9 weight tensors
    struct ggml_tensor ** layer_weights = (struct ggml_tensor **)malloc(28 * 9 * sizeof(struct ggml_tensor *));
    if (!layer_weights) {
        fprintf(stderr, "Error: failed to allocate layer weights array\n");
        *n_samples = 0;
        return nullptr;
    }

    for (int i = 0; i < 28; i++) {
        layer_weights[i * 9 + 0] = ctx->model->talker.layers[i].in_ln_weight;
        layer_weights[i * 9 + 1] = ctx->model->talker.layers[i].attn_q_proj_weight;
        layer_weights[i * 9 + 2] = ctx->model->talker.layers[i].attn_k_proj_weight;
        layer_weights[i * 9 + 3] = ctx->model->talker.layers[i].attn_v_proj_weight;
        layer_weights[i * 9 + 4] = ctx->model->talker.layers[i].attn_o_proj_weight;
        layer_weights[i * 9 + 5] = ctx->model->talker.layers[i].post_ln_weight;
        layer_weights[i * 9 + 6] = ctx->model->talker.layers[i].ffn_gate_proj_weight;
        layer_weights[i * 9 + 7] = ctx->model->talker.layers[i].ffn_up_proj_weight;
        layer_weights[i * 9 + 8] = ctx->model->talker.layers[i].ffn_down_proj_weight;
    }

    // Prepare code predictor layer weights
    struct ggml_tensor ** code_pred_layer_weights = (struct ggml_tensor **)malloc(5 * 9 * sizeof(struct ggml_tensor *));
    if (!code_pred_layer_weights) {
        fprintf(stderr, "Error: failed to allocate code predictor layer weights array\n");
        free(layer_weights);
        *n_samples = 0;
        return nullptr;
    }

    for (int i = 0; i < 5; i++) {
        code_pred_layer_weights[i * 9 + 0] = ctx->model->code_predictor.layers[i].in_ln_weight;
        code_pred_layer_weights[i * 9 + 1] = ctx->model->code_predictor.layers[i].attn_q_proj_weight;
        code_pred_layer_weights[i * 9 + 2] = ctx->model->code_predictor.layers[i].attn_k_proj_weight;
        code_pred_layer_weights[i * 9 + 3] = ctx->model->code_predictor.layers[i].attn_v_proj_weight;
        code_pred_layer_weights[i * 9 + 4] = ctx->model->code_predictor.layers[i].attn_o_proj_weight;
        code_pred_layer_weights[i * 9 + 5] = ctx->model->code_predictor.layers[i].post_ln_weight;
        code_pred_layer_weights[i * 9 + 6] = ctx->model->code_predictor.layers[i].ffn_gate_proj_weight;
        code_pred_layer_weights[i * 9 + 7] = ctx->model->code_predictor.layers[i].ffn_up_proj_weight;
        code_pred_layer_weights[i * 9 + 8] = ctx->model->code_predictor.layers[i].ffn_down_proj_weight;
    }

    // Prepare vocoder upsample weights
    struct ggml_tensor ** upsample_weights = ctx->model->vocoder.upsample_weights;
    struct ggml_tensor ** upsample_alphas = ctx->model->vocoder.upsample_alphas;
    struct ggml_tensor ** upsample_betas = ctx->model->vocoder.upsample_betas;

    // Call tts_generate
    float * audio = tts_generate(
        ctx->ctx,
        text,
        ctx->model->talker.emb_weight,
        layer_weights,
        28,  // n_layers
        ctx->model->talker.norm_weight,
        ctx->model->talker.lm_head_weight,
        ctx->model->code_predictor.codec_embedding_weight,
        code_pred_layer_weights,
        ctx->model->code_predictor.norm_weight,
        ctx->model->code_predictor.output_heads,
        ctx->model->vocoder.codebooks,
        upsample_weights,
        upsample_alphas,
        upsample_betas,
        ctx->model->vocoder.final_conv_weight,
        ctx->model->vocoder.final_conv_bias,
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
struct VocoderTransformerWeights;

void vocoder_decode(
    struct ggml_tensor * dst,
    const struct ggml_tensor * codes,
    const struct ggml_tensor * codebooks,
    const struct ggml_tensor ** upsample_weights,
    const struct ggml_tensor ** upsample_alphas,
    const struct ggml_tensor ** upsample_betas,
    int * kernel_sizes,
    int * paddings,
    const struct ggml_tensor * final_conv_weight,
    const struct ggml_tensor * final_conv_bias,
    const VocoderTransformerWeights * transformer_weights);
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
// Parameters:
//   ctx: ggml context with sufficient memory
//   text: input text to synthesize
//   embed_weight: token embedding matrix
//   layer_weights: array of pointers to transformer layer weights
//   n_layers: number of transformer layers
//   norm_weight: final layer norm weight
//   lm_head_weight: output projection weight
//   codec_embedding: codec embedding table for code predictor [16*2048, hidden_dim]
//   code_pred_layer_weights: code predictor transformer layer weights
//   code_pred_norm_weight: code predictor final layer norm
//   code_pred_output_heads: code predictor output projection heads [16]
//   codebooks: vocoder codebook embeddings [16, 2048, 512]
//   upsample_weights: array of 4 upsample layer weights
//   upsample_alphas: array of 4 alpha parameters
//   upsample_betas: array of 4 beta parameters
//   final_conv_weight: final conv weight (optional, can be nullptr)
//   final_conv_bias: final conv bias (optional, can be nullptr)
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
    struct ggml_tensor * codec_embedding,
    struct ggml_tensor ** code_pred_layer_weights,
    struct ggml_tensor * code_pred_norm_weight,
    struct ggml_tensor ** code_pred_output_heads,
    struct ggml_tensor * codebooks,
    struct ggml_tensor ** upsample_weights,
    struct ggml_tensor ** upsample_alphas,
    struct ggml_tensor ** upsample_betas,
    struct ggml_tensor * final_conv_weight,
    struct ggml_tensor * final_conv_bias,
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

    // Step 4: Run code predictor to generate all 16 codebooks
    // Convert generated tokens to semantic codes (codebook 0)
    int seq_len = codec_len;
    struct ggml_tensor * semantic_codes = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len);
    int32_t * semantic_data = (int32_t *)semantic_codes->data;

    // Convert token IDs to codebook indices (0-2047)
    constexpr int CODEC_VOCAB_START = 151936;
    for (int t = 0; t < seq_len; t++) {
        int token = generated_tokens[codec_start + t];
        int code = token - CODEC_VOCAB_START;
        // Clamp to valid range
        if (code < 0) code = 0;
        if (code >= 2048) code = 2047;
        semantic_data[t] = code;
    }

    free(generated_tokens);

    // Split codec_embedding into 16 separate codebook embeddings
    // codec_embedding: [16*2048, hidden_dim] = [32768, 1024]
    // Each codebook: [2048, 1024]
    constexpr int CODEBOOK_VOCAB = 2048;
    constexpr int HIDDEN_DIM = 1024;
    struct ggml_tensor * codec_embeddings[16];
    for (int cb = 0; cb < 16; cb++) {
        size_t offset = cb * CODEBOOK_VOCAB * HIDDEN_DIM * sizeof(float);
        codec_embeddings[cb] = ggml_view_2d(
            ctx,
            codec_embedding,
            HIDDEN_DIM,
            CODEBOOK_VOCAB,
            codec_embedding->nb[1],
            offset
        );
    }

    // Run code predictor to generate all 16 codebooks autoregressively
    struct ggml_tensor * codes = model::code_predictor_forward(
        ctx,
        semantic_codes,
        codec_embeddings,
        code_pred_layer_weights,
        code_pred_norm_weight,
        code_pred_output_heads,
        HIDDEN_DIM,
        seq_len
    );

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
        paddings,
        final_conv_weight,
        final_conv_bias,
        nullptr  // transformer_weights - not loaded yet
    );

    // Step 6: Return audio samples
    *n_samples_out = audio_len;
    return audio;
}
