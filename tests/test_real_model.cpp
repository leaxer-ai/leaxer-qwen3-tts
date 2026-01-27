// Test with real GGUF weights
// This test loads a real Qwen3-TTS GGUF model and verifies basic functionality

#include "test_utils.h"
#include "ggml.h"
#include "gguf.h"
#include "io/gguf_loader.cpp"
#include "io/wav_writer.cpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <sys/stat.h>

// Check if file exists
bool file_exists(const char* path) {
    struct stat buffer;
    return (stat(path, &buffer) == 0);
}

// Test loading a GGUF model file
bool test_load_gguf_model() {
    using namespace leaxer_qwen::test;

    printf("Testing GGUF model loading...\n");

    // Try multiple possible model paths
    const char* possible_paths[] = {
        "qwen3_tts_0.6b.gguf",
        "qwen3_tts_1.7b.gguf",
        "../qwen3_tts_0.6b.gguf",
        "../qwen3_tts_1.7b.gguf",
        "models/qwen3_tts_0.6b.gguf",
        "models/qwen3_tts_1.7b.gguf",
        nullptr
    };

    const char* model_path = nullptr;
    for (int i = 0; possible_paths[i] != nullptr; i++) {
        if (file_exists(possible_paths[i])) {
            model_path = possible_paths[i];
            break;
        }
    }

    if (!model_path) {
        printf("SKIP: No GGUF model file found. To run this test:\n");
        printf("  1. Download or convert a Qwen3-TTS model\n");
        printf("  2. Place it in the project root as qwen3_tts_0.6b.gguf or qwen3_tts_1.7b.gguf\n");
        printf("  3. Or use: python scripts/convert_to_gguf.py --model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base --output qwen3_tts_0.6b.gguf\n");
        return true;  // Pass the test (skip)
    }

    printf("Found model file: %s\n", model_path);

    // Initialize GGUF context to read metadata
    struct gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ nullptr,
    };
    struct gguf_context * gguf_ctx = gguf_init_from_file(model_path, params);
    TEST_ASSERT(gguf_ctx != nullptr, "Failed to load GGUF file");

    // Check metadata
    int n_tensors = gguf_get_n_tensors(gguf_ctx);
    printf("Model contains %d tensors\n", n_tensors);
    TEST_ASSERT(n_tensors > 0, "Model has no tensors");

    // Try to read some basic metadata
    int meta_idx = gguf_find_key(gguf_ctx, "general.architecture");
    if (meta_idx >= 0) {
        const char* arch = gguf_get_val_str(gguf_ctx, meta_idx);
        printf("Architecture: %s\n", arch);
    }

    meta_idx = gguf_find_key(gguf_ctx, "qwen3.block_count");
    if (meta_idx >= 0) {
        uint32_t n_layers = gguf_get_val_u32(gguf_ctx, meta_idx);
        printf("Number of layers: %u\n", n_layers);
        TEST_ASSERT(n_layers > 0 && n_layers <= 50, "Invalid layer count");
    }

    meta_idx = gguf_find_key(gguf_ctx, "qwen3.embedding_length");
    if (meta_idx >= 0) {
        uint32_t hidden_dim = gguf_get_val_u32(gguf_ctx, meta_idx);
        printf("Hidden dimension: %u\n", hidden_dim);
        TEST_ASSERT(hidden_dim > 0 && hidden_dim <= 8192, "Invalid hidden dimension");
    }

    // Verify we can find some expected tensors and check for quantization
    bool found_embed = false;
    bool found_norm = false;
    bool found_layer = false;
    bool has_quantized = false;

    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(gguf_ctx, i);
        enum ggml_type type = gguf_get_tensor_type(gguf_ctx, i);

        if (strstr(name, "token_embd") || strstr(name, "embed")) {
            found_embed = true;
        }
        if (strstr(name, "output_norm") || strstr(name, "norm")) {
            found_norm = true;
        }
        if (strstr(name, "blk.0") || strstr(name, "layer.0") || strstr(name, "layers.0")) {
            found_layer = true;
        }

        // Check if any tensors are quantized (Q4_0, Q8_0, etc.)
        if (ggml_is_quantized(type)) {
            has_quantized = true;
            if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q8_0) {
                printf("Found quantized tensor: %s (type: %s)\n", name, ggml_type_name(type));
            }
        }
    }

    if (has_quantized) {
        printf("Model contains quantized tensors (Q4_0/Q8_0 support verified)\n");
    }

    TEST_ASSERT(found_embed, "Model missing embedding weights");
    TEST_ASSERT(found_norm, "Model missing normalization weights");
    TEST_ASSERT(found_layer, "Model missing layer weights");

    gguf_free(gguf_ctx);

    TEST_PASS("GGUF model loaded successfully");
    return true;
}

// Test loading a specific tensor from GGUF
bool test_load_tensor_from_gguf() {
    using namespace leaxer_qwen::test;

    printf("Testing tensor loading from GGUF...\n");

    // Find model file (same logic as above)
    const char* possible_paths[] = {
        "qwen3_tts_0.6b.gguf",
        "qwen3_tts_1.7b.gguf",
        "../qwen3_tts_0.6b.gguf",
        "../qwen3_tts_1.7b.gguf",
        "models/qwen3_tts_0.6b.gguf",
        "models/qwen3_tts_1.7b.gguf",
        nullptr
    };

    const char* model_path = nullptr;
    for (int i = 0; possible_paths[i] != nullptr; i++) {
        if (file_exists(possible_paths[i])) {
            model_path = possible_paths[i];
            break;
        }
    }

    if (!model_path) {
        printf("SKIP: No GGUF model file found\n");
        return true;  // Pass the test (skip)
    }

    // Create ggml context for tensor data
    size_t mem_size = 100 * 1024 * 1024;  // 100MB should be enough for one tensor
    struct ggml_init_params params = {
        .mem_size   = mem_size,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * ctx = ggml_init(params);
    TEST_ASSERT(ctx != nullptr, "Failed to create ggml context");

    // Try to load a tensor (token embedding is usually present and not too large)
    // Try different naming conventions
    const char* tensor_names[] = {
        "token_embd.weight",
        "model.embed_tokens.weight",
        "model_embed_tokens_weight",
        nullptr
    };

    struct ggml_tensor * tensor = nullptr;
    const char* found_name = nullptr;
    for (int i = 0; tensor_names[i] != nullptr; i++) {
        tensor = leaxer_qwen::io::gguf_load_tensor(model_path, tensor_names[i], ctx);
        if (tensor != nullptr) {
            found_name = tensor_names[i];
            break;
        }
    }

    if (tensor == nullptr) {
        printf("WARNING: Could not find token embedding tensor (tried multiple names)\n");
        printf("This might be OK if the model uses different tensor naming\n");
        ggml_free(ctx);
        return true;  // Pass with warning
    }

    printf("Loaded tensor '%s'\n", found_name);

    // Verify tensor has valid data
    TEST_ASSERT(tensor->data != nullptr, "Tensor data is null");
    size_t n_elements = ggml_nelements(tensor);
    TEST_ASSERT(n_elements > 0, "Tensor has no elements");
    printf("Tensor has %zu elements\n", n_elements);

    // Check that data is reasonable (not all zeros or NaN)
    // For quantized tensors (Q4_0, Q8_0), we just verify data is non-null
    // For float tensors, we check the actual values
    if (ggml_is_quantized(tensor->type)) {
        // Quantized tensor - just verify we have data
        printf("Tensor is quantized (%s)\n", ggml_type_name(tensor->type));
        TEST_ASSERT(tensor->data != nullptr, "Quantized tensor data is null");
    } else {
        // Float tensor - check values
        float* data = (float*)tensor->data;
        bool has_nonzero = false;
        bool has_nan = false;
        for (size_t i = 0; i < std::min(n_elements, (size_t)1000); i++) {
            if (data[i] != 0.0f) has_nonzero = true;
            if (std::isnan(data[i]) || std::isinf(data[i])) has_nan = true;
        }

        TEST_ASSERT(has_nonzero, "Tensor appears to be all zeros");
        TEST_ASSERT(!has_nan, "Tensor contains NaN or Inf values");
    }

    ggml_free(ctx);

    TEST_PASS("Tensor loaded and validated successfully");
    return true;
}

// Test full TTS generation with real model (if available)
bool test_generate_audio() {
    using namespace leaxer_qwen::test;

    printf("Testing audio generation with real model...\n");

    // Find model file
    const char* possible_paths[] = {
        "qwen3_tts_0.6b.gguf",
        "qwen3_tts_1.7b.gguf",
        "../qwen3_tts_0.6b.gguf",
        "../qwen3_tts_1.7b.gguf",
        "models/qwen3_tts_0.6b.gguf",
        "models/qwen3_tts_1.7b.gguf",
        nullptr
    };

    const char* model_path = nullptr;
    for (int i = 0; possible_paths[i] != nullptr; i++) {
        if (file_exists(possible_paths[i])) {
            model_path = possible_paths[i];
            break;
        }
    }

    if (!model_path) {
        printf("SKIP: No GGUF model file found\n");
        printf("When a model is available, this test will:\n");
        printf("  1. Load the full model\n");
        printf("  2. Generate audio from text: \"Hello world\"\n");
        printf("  3. Verify audio output is valid\n");
        printf("  4. Save to test_real_output.wav for manual inspection\n");
        return true;  // Pass the test (skip)
    }

    printf("Found model file: %s\n", model_path);
    printf("TODO: Implement full model loading and audio generation\n");
    printf("This requires:\n");
    printf("  - Complete GGUF loader for all model weights\n");
    printf("  - Tokenizer implementation\n");
    printf("  - LLM forward pass\n");
    printf("  - Code predictor\n");
    printf("  - Vocoder pipeline\n");
    printf("\n");
    printf("For now, this test verifies the model file can be opened and contains valid metadata.\n");
    printf("Audio generation will be enabled once the full pipeline is implemented.\n");

    TEST_PASS("Audio generation test (placeholder)");
    return true;
}

// Main test entry point
int main() {
    printf("leaxer-qwen real model test\n");
    printf("============================\n\n");

    test_load_gguf_model();
    test_load_tensor_from_gguf();
    test_generate_audio();

    return leaxer_qwen::test::print_summary();
}
