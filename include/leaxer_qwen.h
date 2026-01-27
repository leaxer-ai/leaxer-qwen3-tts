#ifndef LEAXER_QWEN_H
#define LEAXER_QWEN_H

#include <stdint.h>
#include <stddef.h>
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Version
#ifndef LEAXER_QWEN_VERSION
#define LEAXER_QWEN_VERSION "0.1.0"
#endif

// Forward declarations
struct leaxer_qwen_context;
struct leaxer_qwen_model;

// Model parameters
struct leaxer_qwen_model_params {
    int32_t n_threads;      // Number of threads for inference
    int32_t n_gpu_layers;   // Number of layers to offload to GPU (0 = CPU only)
    bool    use_mmap;       // Use memory mapping for model loading
    bool    use_mlock;      // Lock model in memory
};

// Generation parameters
struct leaxer_qwen_gen_params {
    float   temperature;    // Sampling temperature (default: 0.9)
    int32_t top_k;          // Top-k sampling (default: 50)
    float   top_p;          // Top-p (nucleus) sampling (default: 0.95)
    int32_t max_tokens;     // Maximum tokens to generate (default: 2048)
    int32_t seed;           // Random seed (-1 for random)
};

// Default parameters
struct leaxer_qwen_model_params leaxer_qwen_model_default_params(void);
struct leaxer_qwen_gen_params   leaxer_qwen_gen_default_params(void);

// Model loading
struct leaxer_qwen_model * leaxer_qwen_load_model(
    const char * path,
    struct leaxer_qwen_model_params params
);
void leaxer_qwen_free_model(struct leaxer_qwen_model * model);

// Context management
struct leaxer_qwen_context * leaxer_qwen_new_context(
    struct leaxer_qwen_model * model
);
void leaxer_qwen_free_context(struct leaxer_qwen_context * ctx);

// Text-to-speech generation
// Returns audio samples (24kHz, float32, mono)
// Caller must free the returned buffer with leaxer_qwen_free_audio()
float * leaxer_qwen_generate(
    struct leaxer_qwen_context * ctx,
    const char * text,
    struct leaxer_qwen_gen_params params,
    size_t * n_samples  // Output: number of audio samples
);
void leaxer_qwen_free_audio(float * audio);

// WAV file output
int leaxer_qwen_write_wav(
    const char * path,
    const float * audio,
    size_t n_samples,
    int sample_rate  // Should be 24000
);

// Utility
const char * leaxer_qwen_version(void);
void leaxer_qwen_print_system_info(void);

#ifdef __cplusplus
}
#endif

#endif // LEAXER_QWEN_H
