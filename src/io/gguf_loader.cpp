// GGUF Model Loader
// Loads Qwen3-TTS weights from GGUF format
// Supports both float (F32, F16) and quantized (Q4_0, Q8_0) tensor types
// Quantized tensors are loaded as-is and automatically dequantized during ggml operations

#include "ggml.h"
#include "gguf.h"
#include "model/model_weights.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace leaxer_qwen {
namespace io {

// Map GGUF tensor name to model struct field path
// Handles shortened names from conversion:
// - "tk_l_N_*" -> talker layer N fields
// - "talker_cp_l_N_*" -> code predictor layer N fields
// - "talker_model_*" -> talker top-level fields
// Returns empty string if name is not recognized
static const char * map_gguf_tensor_name(const char * gguf_name) {
    if (!gguf_name) {
        return "";
    }

    // Talker model layers: tk_l_N_*
    if (strncmp(gguf_name, "tk_l_", 5) == 0) {
        // Extract layer number
        int layer_num = atoi(gguf_name + 5);
        const char * rest = strchr(gguf_name + 5, '_');
        if (!rest) return "";
        rest++; // Skip the underscore after layer number

        // Map to talker.layers[N].field
        static char path[256];
        snprintf(path, sizeof(path), "talker.layers[%d].", layer_num);

        // Map field names
        if (strncmp(rest, "in_ln_weight", 12) == 0) {
            strcat(path, "in_ln_weight");
        } else if (strncmp(rest, "attn_q_proj_weight", 18) == 0) {
            strcat(path, "attn_q_proj_weight");
        } else if (strncmp(rest, "attn_k_proj_weight", 18) == 0) {
            strcat(path, "attn_k_proj_weight");
        } else if (strncmp(rest, "attn_v_proj_weight", 18) == 0) {
            strcat(path, "attn_v_proj_weight");
        } else if (strncmp(rest, "attn_o_proj_weight", 18) == 0) {
            strcat(path, "attn_o_proj_weight");
        } else if (strncmp(rest, "post_ln_weight", 14) == 0) {
            strcat(path, "post_ln_weight");
        } else if (strncmp(rest, "ffn_gate_proj_weight", 20) == 0) {
            strcat(path, "ffn_gate_proj_weight");
        } else if (strncmp(rest, "ffn_up_proj_weight", 18) == 0) {
            strcat(path, "ffn_up_proj_weight");
        } else if (strncmp(rest, "ffn_down_proj_weight", 20) == 0) {
            strcat(path, "ffn_down_proj_weight");
        } else {
            return "";
        }

        return path;
    }

    // Code predictor layers: talker_cp_l_N_*
    if (strncmp(gguf_name, "talker_cp_l_", 12) == 0) {
        // Extract layer number
        int layer_num = atoi(gguf_name + 12);
        const char * rest = strchr(gguf_name + 12, '_');
        if (!rest) return "";
        rest++; // Skip the underscore after layer number

        // Map to code_predictor.layers[N].field
        static char path[256];
        snprintf(path, sizeof(path), "code_predictor.layers[%d].", layer_num);

        // Map field names (same as talker layer fields)
        if (strncmp(rest, "in_ln_weight", 12) == 0) {
            strcat(path, "in_ln_weight");
        } else if (strncmp(rest, "attn_q_proj_weight", 18) == 0) {
            strcat(path, "attn_q_proj_weight");
        } else if (strncmp(rest, "attn_k_proj_weight", 18) == 0) {
            strcat(path, "attn_k_proj_weight");
        } else if (strncmp(rest, "attn_v_proj_weight", 18) == 0) {
            strcat(path, "attn_v_proj_weight");
        } else if (strncmp(rest, "attn_o_proj_weight", 18) == 0) {
            strcat(path, "attn_o_proj_weight");
        } else if (strncmp(rest, "post_ln_weight", 14) == 0) {
            strcat(path, "post_ln_weight");
        } else if (strncmp(rest, "ffn_gate_proj_weight", 20) == 0) {
            strcat(path, "ffn_gate_proj_weight");
        } else if (strncmp(rest, "ffn_up_proj_weight", 18) == 0) {
            strcat(path, "ffn_up_proj_weight");
        } else if (strncmp(rest, "ffn_down_proj_weight", 20) == 0) {
            strcat(path, "ffn_down_proj_weight");
        } else {
            return "";
        }

        return path;
    }

    // Talker top-level fields: talker_model_*
    if (strncmp(gguf_name, "talker_model_", 13) == 0) {
        const char * rest = gguf_name + 13;
        static char path[256];
        strcpy(path, "talker.");

        if (strncmp(rest, "emb_weight", 10) == 0 ||
            strncmp(rest, "embed_tokens_weight", 19) == 0) {
            strcat(path, "emb_weight");
        } else if (strcmp(rest, "norm_weight") == 0) {
            strcat(path, "norm_weight");
        } else if (strcmp(rest, "lm_head_weight") == 0) {
            strcat(path, "lm_head_weight");
        } else if (strcmp(rest, "codec_embedding_weight") == 0) {
            // This is actually for code_predictor
            return "code_predictor.codec_embedding_weight";
        } else {
            return "";
        }

        return path;
    }

    // Code predictor top-level fields
    if (strncmp(gguf_name, "talker_code_predictor_", 22) == 0) {
        const char * rest = gguf_name + 22;
        static char path[256];
        strcpy(path, "code_predictor.");

        if (strcmp(rest, "norm_weight") == 0) {
            strcat(path, "norm_weight");
        } else if (strncmp(rest, "output_heads_", 13) == 0) {
            // output_heads_N_weight
            int head_num = atoi(rest + 13);
            snprintf(path + 15, 256 - 15, "output_heads[%d]", head_num);
        } else {
            return "";
        }

        return path;
    }

    // Vocoder fields: decoder_* (from Tokenizer-12Hz model)
    if (strncmp(gguf_name, "decoder_", 8) == 0) {
        const char * rest = gguf_name + 8;
        static char path[256];
        strcpy(path, "vocoder.");

        // Parse vocoder component names
        if (strncmp(rest, "codebooks", 9) == 0) {
            strcat(path, "codebooks");
        } else if (strncmp(rest, "causal_conv_weight", 18) == 0) {
            strcat(path, "causal_conv_weight");
        } else if (strncmp(rest, "causal_conv_bias", 16) == 0) {
            strcat(path, "causal_conv_bias");
        } else if (strncmp(rest, "upsample_", 9) == 0) {
            // upsample_N_weight, upsample_N_bias, upsample_N_alpha, upsample_N_beta
            int stage = atoi(rest + 9);
            const char * type = strchr(rest + 9, '_');
            if (!type) return "";
            type++; // Skip underscore

            if (strcmp(type, "weight") == 0) {
                snprintf(path + 8, 256 - 8, "upsample_weights[%d]", stage);
            } else if (strcmp(type, "bias") == 0) {
                snprintf(path + 8, 256 - 8, "upsample_biases[%d]", stage);
            } else if (strcmp(type, "alpha") == 0) {
                snprintf(path + 8, 256 - 8, "upsample_alphas[%d]", stage);
            } else if (strcmp(type, "beta") == 0) {
                snprintf(path + 8, 256 - 8, "upsample_betas[%d]", stage);
            } else {
                return "";
            }
        } else if (strcmp(rest, "final_conv_weight") == 0) {
            strcat(path, "final_conv_weight");
        } else if (strcmp(rest, "final_conv_bias") == 0) {
            strcat(path, "final_conv_bias");
        } else {
            return "";
        }

        return path;
    }

    // Unrecognized tensor name
    return "";
}

// Load a single tensor from GGUF file by name
// Returns nullptr if tensor not found or on error
// The tensor data is allocated in the provided ggml_context
struct ggml_tensor * gguf_load_tensor(
    const char * gguf_path,
    const char * tensor_name,
    struct ggml_context * ctx
) {
    if (!gguf_path || !tensor_name || !ctx) {
        fprintf(stderr, "gguf_load_tensor: invalid parameters\n");
        return nullptr;
    }

    // Open GGUF file for reading tensor data
    FILE * file = fopen(gguf_path, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open GGUF file: %s\n", gguf_path);
        return nullptr;
    }

    // Initialize GGUF context with ggml context to auto-create tensors with correct dimensions
    struct gguf_init_params params = {
        /*.no_alloc =*/ false,
        /*.ctx      =*/ &ctx,
    };
    struct gguf_context * gguf_ctx = gguf_init_from_file(gguf_path, params);
    if (!gguf_ctx) {
        fprintf(stderr, "Failed to initialize GGUF context from: %s\n", gguf_path);
        fclose(file);
        return nullptr;
    }

    // Find tensor by name
    int64_t tensor_id = gguf_find_tensor(gguf_ctx, tensor_name);
    if (tensor_id < 0) {
        fprintf(stderr, "Tensor '%s' not found in GGUF file\n", tensor_name);
        gguf_free(gguf_ctx);
        fclose(file);
        return nullptr;
    }

    // Get tensor metadata
    size_t tensor_offset = gguf_get_tensor_offset(gguf_ctx, tensor_id);
    size_t tensor_size = gguf_get_tensor_size(gguf_ctx, tensor_id);
    size_t data_offset = gguf_get_data_offset(gguf_ctx);

    // Get tensor that was created by gguf_init_from_file with proper multi-dimensional shape
    struct ggml_tensor * tensor = ggml_get_tensor(ctx, tensor_name);
    if (!tensor) {
        fprintf(stderr, "Failed to get tensor '%s' from context\n", tensor_name);
        gguf_free(gguf_ctx);
        fclose(file);
        return nullptr;
    }

    // Seek to tensor data in file and read
    if (fseek(file, data_offset + tensor_offset, SEEK_SET) != 0) {
        fprintf(stderr, "Failed to seek to tensor data\n");
        gguf_free(gguf_ctx);
        fclose(file);
        return nullptr;
    }

    size_t bytes_read = fread(tensor->data, 1, tensor_size, file);
    if (bytes_read != tensor_size) {
        fprintf(stderr, "Failed to read tensor data: expected %zu bytes, got %zu\n",
                tensor_size, bytes_read);
        gguf_free(gguf_ctx);
        fclose(file);
        return nullptr;
    }

    // Cleanup
    gguf_free(gguf_ctx);
    fclose(file);

    return tensor;
}

// Load all tensors from GGUF file into a ggml context
// Returns true on success, false on error
// All tensors are created in the provided context and their data is loaded
bool gguf_load_model(
    const char * gguf_path,
    struct ggml_context * ctx
) {
    if (!gguf_path || !ctx) {
        fprintf(stderr, "gguf_load_model: invalid parameters\n");
        return false;
    }

    // Open GGUF file for reading
    FILE * file = fopen(gguf_path, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open GGUF file: %s\n", gguf_path);
        return false;
    }

    // Initialize GGUF context with the provided ggml context
    // This will create all tensors in the context automatically
    struct gguf_init_params params = {
        /*.no_alloc =*/ false,
        /*.ctx      =*/ &ctx,
    };
    struct gguf_context * gguf_ctx = gguf_init_from_file(gguf_path, params);
    if (!gguf_ctx) {
        fprintf(stderr, "Failed to initialize GGUF context from: %s\n", gguf_path);
        fclose(file);
        return false;
    }

    // Get number of tensors
    int n_tensors = gguf_get_n_tensors(gguf_ctx);
    if (n_tensors <= 0) {
        fprintf(stderr, "No tensors found in GGUF file\n");
        gguf_free(gguf_ctx);
        fclose(file);
        return false;
    }

    // Get data offset
    size_t data_offset = gguf_get_data_offset(gguf_ctx);

    // Load data for each tensor
    for (int i = 0; i < n_tensors; i++) {
        const char * tensor_name = gguf_get_tensor_name(gguf_ctx, i);
        size_t tensor_offset = gguf_get_tensor_offset(gguf_ctx, i);
        size_t tensor_size = gguf_get_tensor_size(gguf_ctx, i);

        // Get the tensor that was created by gguf_init_from_file
        struct ggml_tensor * tensor = ggml_get_tensor(ctx, tensor_name);
        if (!tensor) {
            fprintf(stderr, "Failed to get tensor '%s' from context\n", tensor_name);
            gguf_free(gguf_ctx);
            fclose(file);
            return false;
        }

        // Seek to tensor data and read
        if (fseek(file, data_offset + tensor_offset, SEEK_SET) != 0) {
            fprintf(stderr, "Failed to seek to tensor '%s' data\n", tensor_name);
            gguf_free(gguf_ctx);
            fclose(file);
            return false;
        }

        size_t bytes_read = fread(tensor->data, 1, tensor_size, file);
        if (bytes_read != tensor_size) {
            fprintf(stderr, "Failed to read tensor '%s' data: expected %zu bytes, got %zu\n",
                    tensor_name, tensor_size, bytes_read);
            gguf_free(gguf_ctx);
            fclose(file);
            return false;
        }
    }

    // Cleanup
    gguf_free(gguf_ctx);
    fclose(file);

    return true;
}

// Test wrapper to expose mapping function
const char * test_map_gguf_tensor_name(const char * gguf_name) {
    return map_gguf_tensor_name(gguf_name);
}

// Load talker weights from GGUF file into TalkerWeights struct
// Returns true on success, false on error
// Loads embeddings, all 28 transformer layers, and final norm
bool load_talker_weights(
    const char * gguf_path,
    struct model::TalkerWeights * weights,
    struct ggml_context * ctx
) {
    if (!gguf_path || !weights || !ctx) {
        fprintf(stderr, "load_talker_weights: invalid parameters\n");
        return false;
    }

    // Open GGUF file for reading
    FILE * file = fopen(gguf_path, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open GGUF file: %s\n", gguf_path);
        return false;
    }

    // Initialize GGUF context with ggml context
    // This will create all tensors in the context automatically with proper dimensions
    struct gguf_init_params params = {
        /*.no_alloc =*/ false,
        /*.ctx      =*/ &ctx,
    };
    struct gguf_context * gguf_ctx = gguf_init_from_file(gguf_path, params);
    if (!gguf_ctx) {
        fprintf(stderr, "Failed to initialize GGUF context from: %s\n", gguf_path);
        fclose(file);
        return false;
    }

    size_t data_offset = gguf_get_data_offset(gguf_ctx);

    // Helper macro to load a tensor by name
    #define LOAD_TENSOR(tensor_ptr, name) \
        do { \
            int64_t tid = gguf_find_tensor(gguf_ctx, name); \
            if (tid < 0) { \
                fprintf(stderr, "Tensor '%s' not found in GGUF file\n", name); \
                gguf_free(gguf_ctx); \
                fclose(file); \
                return false; \
            } \
            tensor_ptr = ggml_get_tensor(ctx, name); \
            if (!tensor_ptr) { \
                fprintf(stderr, "Failed to get tensor '%s' from context\n", name); \
                gguf_free(gguf_ctx); \
                fclose(file); \
                return false; \
            } \
            size_t t_offset = gguf_get_tensor_offset(gguf_ctx, tid); \
            size_t t_size = gguf_get_tensor_size(gguf_ctx, tid); \
            if (fseek(file, data_offset + t_offset, SEEK_SET) != 0) { \
                fprintf(stderr, "Failed to seek to tensor '%s' data\n", name); \
                gguf_free(gguf_ctx); \
                fclose(file); \
                return false; \
            } \
            if (fread(tensor_ptr->data, 1, t_size, file) != t_size) { \
                fprintf(stderr, "Failed to read tensor '%s' data\n", name); \
                gguf_free(gguf_ctx); \
                fclose(file); \
                return false; \
            } \
        } while (0)

    // Helper macro to try multiple tensor names (for compatibility)
    #define LOAD_TENSOR_ALT(tensor_ptr, name1, name2) \
        do { \
            int64_t tid = gguf_find_tensor(gguf_ctx, name1); \
            const char * found_name = name1; \
            if (tid < 0) { \
                tid = gguf_find_tensor(gguf_ctx, name2); \
                found_name = name2; \
            } \
            if (tid < 0) { \
                fprintf(stderr, "Tensor '%s' or '%s' not found in GGUF file\n", name1, name2); \
                gguf_free(gguf_ctx); \
                fclose(file); \
                return false; \
            } \
            tensor_ptr = ggml_get_tensor(ctx, found_name); \
            if (!tensor_ptr) { \
                fprintf(stderr, "Failed to get tensor '%s' from context\n", found_name); \
                gguf_free(gguf_ctx); \
                fclose(file); \
                return false; \
            } \
            size_t t_offset = gguf_get_tensor_offset(gguf_ctx, tid); \
            size_t t_size = gguf_get_tensor_size(gguf_ctx, tid); \
            if (fseek(file, data_offset + t_offset, SEEK_SET) != 0) { \
                fprintf(stderr, "Failed to seek to tensor '%s' data\n", found_name); \
                gguf_free(gguf_ctx); \
                fclose(file); \
                return false; \
            } \
            if (fread(tensor_ptr->data, 1, t_size, file) != t_size) { \
                fprintf(stderr, "Failed to read tensor '%s' data\n", found_name); \
                gguf_free(gguf_ctx); \
                fclose(file); \
                return false; \
            } \
        } while (0)

    // Load embedding weight (supports both naming conventions)
    LOAD_TENSOR_ALT(weights->emb_weight, "talker_model_emb_weight", "talker_model_text_embedding_weight");

    // Load text projection weights (embedding_dim → hidden_dim)
    // Flow: input(2048) → fc1 → SiLU → fc2 → output(1024)
    LOAD_TENSOR(weights->text_proj_fc1_weight, "talker_text_projection_linear_fc1_weight");
    LOAD_TENSOR(weights->text_proj_fc1_bias, "talker_text_projection_linear_fc1_bias");
    LOAD_TENSOR(weights->text_proj_fc2_weight, "talker_text_projection_linear_fc2_weight");
    LOAD_TENSOR(weights->text_proj_fc2_bias, "talker_text_projection_linear_fc2_bias");

    // Load all 28 layers
    for (int layer = 0; layer < 28; layer++) {
        char tensor_name[128];
        model::TalkerLayer * layer_weights = &weights->layers[layer];

        // Load input layer norm
        snprintf(tensor_name, sizeof(tensor_name), "tk_l_%d_in_ln_weight", layer);
        LOAD_TENSOR(layer_weights->in_ln_weight, tensor_name);

        // Load attention Q projection
        snprintf(tensor_name, sizeof(tensor_name), "tk_l_%d_attn_q_proj_weight", layer);
        LOAD_TENSOR(layer_weights->attn_q_proj_weight, tensor_name);

        // Load attention K projection
        snprintf(tensor_name, sizeof(tensor_name), "tk_l_%d_attn_k_proj_weight", layer);
        LOAD_TENSOR(layer_weights->attn_k_proj_weight, tensor_name);

        // Load attention V projection
        snprintf(tensor_name, sizeof(tensor_name), "tk_l_%d_attn_v_proj_weight", layer);
        LOAD_TENSOR(layer_weights->attn_v_proj_weight, tensor_name);

        // Load attention O projection
        snprintf(tensor_name, sizeof(tensor_name), "tk_l_%d_attn_o_proj_weight", layer);
        LOAD_TENSOR(layer_weights->attn_o_proj_weight, tensor_name);

        // Load post attention layer norm
        snprintf(tensor_name, sizeof(tensor_name), "tk_l_%d_post_ln_weight", layer);
        LOAD_TENSOR(layer_weights->post_ln_weight, tensor_name);

        // Load FFN/MLP gate projection (supports both naming conventions)
        char tensor_name_alt[128];
        snprintf(tensor_name, sizeof(tensor_name), "tk_l_%d_ffn_gate_proj_weight", layer);
        snprintf(tensor_name_alt, sizeof(tensor_name_alt), "tk_l_%d_mlp_gate_proj_weight", layer);
        LOAD_TENSOR_ALT(layer_weights->ffn_gate_proj_weight, tensor_name, tensor_name_alt);

        // Load FFN/MLP up projection
        snprintf(tensor_name, sizeof(tensor_name), "tk_l_%d_ffn_up_proj_weight", layer);
        snprintf(tensor_name_alt, sizeof(tensor_name_alt), "tk_l_%d_mlp_up_proj_weight", layer);
        LOAD_TENSOR_ALT(layer_weights->ffn_up_proj_weight, tensor_name, tensor_name_alt);

        // Load FFN/MLP down projection
        snprintf(tensor_name, sizeof(tensor_name), "tk_l_%d_ffn_down_proj_weight", layer);
        snprintf(tensor_name_alt, sizeof(tensor_name_alt), "tk_l_%d_mlp_down_proj_weight", layer);
        LOAD_TENSOR_ALT(layer_weights->ffn_down_proj_weight, tensor_name, tensor_name_alt);
    }

    // Load final norm weight
    LOAD_TENSOR(weights->norm_weight, "talker_model_norm_weight");

    // Load lm_head weight (may not exist in all models)
    {
        int64_t tid = gguf_find_tensor(gguf_ctx, "talker_model_lm_head_weight");
        if (tid >= 0) {
            LOAD_TENSOR(weights->lm_head_weight, "talker_model_lm_head_weight");
        } else {
            weights->lm_head_weight = nullptr;
        }
    }

    #undef LOAD_TENSOR
    #undef LOAD_TENSOR_ALT

    // Cleanup
    gguf_free(gguf_ctx);
    fclose(file);

    return true;
}

// Load code predictor weights from GGUF file into CodePredictorWeights struct
// Returns true on success, false on error
// Loads codec embeddings, all 5 transformer layers, norm, and 16 output heads
bool load_code_predictor_weights(
    const char * gguf_path,
    struct model::CodePredictorWeights * weights,
    struct ggml_context * ctx
) {
    if (!gguf_path || !weights || !ctx) {
        fprintf(stderr, "load_code_predictor_weights: invalid parameters\n");
        return false;
    }

    // Open GGUF file for reading
    FILE * file = fopen(gguf_path, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open GGUF file: %s\n", gguf_path);
        return false;
    }

    // Initialize GGUF context with ggml context
    // This will create all tensors in the context automatically with proper dimensions
    struct gguf_init_params params = {
        /*.no_alloc =*/ false,
        /*.ctx      =*/ &ctx,
    };
    struct gguf_context * gguf_ctx = gguf_init_from_file(gguf_path, params);
    if (!gguf_ctx) {
        fprintf(stderr, "Failed to initialize GGUF context from: %s\n", gguf_path);
        fclose(file);
        return false;
    }

    size_t data_offset = gguf_get_data_offset(gguf_ctx);

    // Helper macro to load a tensor by name
    #define LOAD_TENSOR(tensor_ptr, name) \
        do { \
            int64_t tid = gguf_find_tensor(gguf_ctx, name); \
            if (tid < 0) { \
                fprintf(stderr, "Tensor '%s' not found in GGUF file\n", name); \
                gguf_free(gguf_ctx); \
                fclose(file); \
                return false; \
            } \
            tensor_ptr = ggml_get_tensor(ctx, name); \
            if (!tensor_ptr) { \
                fprintf(stderr, "Failed to get tensor '%s' from context\n", name); \
                gguf_free(gguf_ctx); \
                fclose(file); \
                return false; \
            } \
            size_t t_offset = gguf_get_tensor_offset(gguf_ctx, tid); \
            size_t t_size = gguf_get_tensor_size(gguf_ctx, tid); \
            if (fseek(file, data_offset + t_offset, SEEK_SET) != 0) { \
                fprintf(stderr, "Failed to seek to tensor '%s' data\n", name); \
                gguf_free(gguf_ctx); \
                fclose(file); \
                return false; \
            } \
            if (fread(tensor_ptr->data, 1, t_size, file) != t_size) { \
                fprintf(stderr, "Failed to read tensor '%s' data\n", name); \
                gguf_free(gguf_ctx); \
                fclose(file); \
                return false; \
            } \
        } while (0)

    // Helper macro to try multiple tensor names (for compatibility)
    #define LOAD_TENSOR_ALT(tensor_ptr, name1, name2) \
        do { \
            int64_t tid = gguf_find_tensor(gguf_ctx, name1); \
            const char * found_name = name1; \
            if (tid < 0) { \
                tid = gguf_find_tensor(gguf_ctx, name2); \
                found_name = name2; \
            } \
            if (tid < 0) { \
                fprintf(stderr, "Tensor '%s' or '%s' not found in GGUF file\n", name1, name2); \
                gguf_free(gguf_ctx); \
                fclose(file); \
                return false; \
            } \
            tensor_ptr = ggml_get_tensor(ctx, found_name); \
            if (!tensor_ptr) { \
                fprintf(stderr, "Failed to get tensor '%s' from context\n", found_name); \
                gguf_free(gguf_ctx); \
                fclose(file); \
                return false; \
            } \
            size_t t_offset = gguf_get_tensor_offset(gguf_ctx, tid); \
            size_t t_size = gguf_get_tensor_size(gguf_ctx, tid); \
            if (fseek(file, data_offset + t_offset, SEEK_SET) != 0) { \
                fprintf(stderr, "Failed to seek to tensor '%s' data\n", found_name); \
                gguf_free(gguf_ctx); \
                fclose(file); \
                return false; \
            } \
            if (fread(tensor_ptr->data, 1, t_size, file) != t_size) { \
                fprintf(stderr, "Failed to read tensor '%s' data\n", found_name); \
                gguf_free(gguf_ctx); \
                fclose(file); \
                return false; \
            } \
        } while (0)

    // Load codec embedding weight
    LOAD_TENSOR(weights->codec_embedding_weight, "talker_model_codec_embedding_weight");

    // Load all 5 layers
    for (int layer = 0; layer < 5; layer++) {
        char tensor_name[128];
        char tensor_name_alt[128];
        model::CodePredictorLayer * layer_weights = &weights->layers[layer];

        // Load input layer norm
        snprintf(tensor_name, sizeof(tensor_name), "talker_cp_l_%d_in_ln_weight", layer);
        LOAD_TENSOR(layer_weights->in_ln_weight, tensor_name);

        // Load attention Q projection
        snprintf(tensor_name, sizeof(tensor_name), "talker_cp_l_%d_attn_q_proj_weight", layer);
        LOAD_TENSOR(layer_weights->attn_q_proj_weight, tensor_name);

        // Load attention K projection
        snprintf(tensor_name, sizeof(tensor_name), "talker_cp_l_%d_attn_k_proj_weight", layer);
        LOAD_TENSOR(layer_weights->attn_k_proj_weight, tensor_name);

        // Load attention V projection
        snprintf(tensor_name, sizeof(tensor_name), "talker_cp_l_%d_attn_v_proj_weight", layer);
        LOAD_TENSOR(layer_weights->attn_v_proj_weight, tensor_name);

        // Load attention O projection
        snprintf(tensor_name, sizeof(tensor_name), "talker_cp_l_%d_attn_o_proj_weight", layer);
        LOAD_TENSOR(layer_weights->attn_o_proj_weight, tensor_name);

        // Load post attention layer norm
        snprintf(tensor_name, sizeof(tensor_name), "talker_cp_l_%d_post_ln_weight", layer);
        LOAD_TENSOR(layer_weights->post_ln_weight, tensor_name);

        // Load FFN/MLP gate projection (supports both naming conventions)
        snprintf(tensor_name, sizeof(tensor_name), "talker_cp_l_%d_ffn_gate_proj_weight", layer);
        snprintf(tensor_name_alt, sizeof(tensor_name_alt), "talker_cp_l_%d_mlp_gate_proj_weight", layer);
        LOAD_TENSOR_ALT(layer_weights->ffn_gate_proj_weight, tensor_name, tensor_name_alt);

        // Load FFN/MLP up projection
        snprintf(tensor_name, sizeof(tensor_name), "talker_cp_l_%d_ffn_up_proj_weight", layer);
        snprintf(tensor_name_alt, sizeof(tensor_name_alt), "talker_cp_l_%d_mlp_up_proj_weight", layer);
        LOAD_TENSOR_ALT(layer_weights->ffn_up_proj_weight, tensor_name, tensor_name_alt);

        // Load FFN/MLP down projection
        snprintf(tensor_name, sizeof(tensor_name), "talker_cp_l_%d_ffn_down_proj_weight", layer);
        snprintf(tensor_name_alt, sizeof(tensor_name_alt), "talker_cp_l_%d_mlp_down_proj_weight", layer);
        LOAD_TENSOR_ALT(layer_weights->ffn_down_proj_weight, tensor_name, tensor_name_alt);
    }

    // Load final norm weight (try both naming conventions)
    {
        int64_t tid = gguf_find_tensor(gguf_ctx, "talker_code_predictor_norm_weight");
        if (tid >= 0) {
            LOAD_TENSOR(weights->norm_weight, "talker_code_predictor_norm_weight");
        } else {
            LOAD_TENSOR(weights->norm_weight, "talker_code_predictor_model_norm_weight");
        }
    }

    // Load all 15 output heads (acoustic codebooks; semantic is handled by main LLM)
    for (int head = 0; head < 15; head++) {
        char tensor_name[128];
        char tensor_name_alt[128];
        snprintf(tensor_name, sizeof(tensor_name), "talker_code_predictor_output_heads_%d_weight", head);
        snprintf(tensor_name_alt, sizeof(tensor_name_alt), "talker_code_predictor_lm_head_%d_weight", head);
        LOAD_TENSOR_ALT(weights->output_heads[head], tensor_name, tensor_name_alt);
    }

    #undef LOAD_TENSOR
    #undef LOAD_TENSOR_ALT

    // Cleanup
    gguf_free(gguf_ctx);
    fclose(file);

    return true;
}

// Load vocoder weights from GGUF file into VocoderWeights struct
// Returns true on success, false on error
// Loads decoder weights from Qwen3-TTS-Tokenizer-12Hz model
// Includes: codebooks, causal conv, 4 upsample stages, final conv
bool load_vocoder_weights(
    const char * gguf_path,
    struct model::VocoderWeights * weights,
    struct ggml_context * ctx
) {
    if (!gguf_path || !weights || !ctx) {
        fprintf(stderr, "load_vocoder_weights: invalid parameters\n");
        return false;
    }

    // Open GGUF file for reading
    FILE * file = fopen(gguf_path, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open GGUF file: %s\n", gguf_path);
        return false;
    }

    // First, do a lightweight check to see if this file has vocoder tensors
    // Use no_alloc=true to avoid modifying the ggml context
    struct gguf_init_params check_params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ nullptr,
    };
    struct gguf_context * check_ctx = gguf_init_from_file(gguf_path, check_params);
    if (!check_ctx) {
        fprintf(stderr, "Failed to read GGUF file: %s\n", gguf_path);
        fclose(file);
        return false;
    }

    // Check if this file has the required vocoder tensor
    if (gguf_find_tensor(check_ctx, "decoder_codebooks") < 0) {
        // Silently fail - vocoder is optional
        gguf_free(check_ctx);
        fclose(file);
        return false;
    }
    gguf_free(check_ctx);

    // Now do the actual loading with tensor allocation
    struct gguf_init_params params = {
        /*.no_alloc =*/ false,
        /*.ctx      =*/ &ctx,
    };
    struct gguf_context * gguf_ctx = gguf_init_from_file(gguf_path, params);
    if (!gguf_ctx) {
        fprintf(stderr, "Failed to initialize GGUF context from: %s\n", gguf_path);
        fclose(file);
        return false;
    }

    size_t data_offset = gguf_get_data_offset(gguf_ctx);

    // Helper macro to load a tensor by name
    #define LOAD_TENSOR(tensor_ptr, name) \
        do { \
            int64_t tid = gguf_find_tensor(gguf_ctx, name); \
            if (tid < 0) { \
                fprintf(stderr, "Tensor '%s' not found in GGUF file\n", name); \
                gguf_free(gguf_ctx); \
                fclose(file); \
                return false; \
            } \
            tensor_ptr = ggml_get_tensor(ctx, name); \
            if (!tensor_ptr) { \
                fprintf(stderr, "Failed to get tensor '%s' from context\n", name); \
                gguf_free(gguf_ctx); \
                fclose(file); \
                return false; \
            } \
            size_t t_offset = gguf_get_tensor_offset(gguf_ctx, tid); \
            size_t t_size = gguf_get_tensor_size(gguf_ctx, tid); \
            if (fseek(file, data_offset + t_offset, SEEK_SET) != 0) { \
                fprintf(stderr, "Failed to seek to tensor '%s' data\n", name); \
                gguf_free(gguf_ctx); \
                fclose(file); \
                return false; \
            } \
            if (fread(tensor_ptr->data, 1, t_size, file) != t_size) { \
                fprintf(stderr, "Failed to read tensor '%s' data\n", name); \
                gguf_free(gguf_ctx); \
                fclose(file); \
                return false; \
            } \
        } while (0)

    // Load codebooks
    LOAD_TENSOR(weights->codebooks, "decoder_codebooks");

    // Load RVQ output projections
    LOAD_TENSOR(weights->rvq_first_output_proj, "decoder_rvq_first_output_proj_weight");
    LOAD_TENSOR(weights->rvq_rest_output_proj, "decoder_rvq_rest_output_proj_weight");

    // Load pre-transformer input/output projections
    LOAD_TENSOR(weights->pre_transformer_input_proj_weight, "decoder_pre_transformer_input_proj_weight");
    LOAD_TENSOR(weights->pre_transformer_input_proj_bias, "decoder_pre_transformer_input_proj_bias");
    LOAD_TENSOR(weights->pre_transformer_output_proj_weight, "decoder_pre_transformer_output_proj_weight");
    LOAD_TENSOR(weights->pre_transformer_output_proj_bias, "decoder_pre_transformer_output_proj_bias");

    // Load 8 pre-transformer layers
    for (int layer = 0; layer < 8; layer++) {
        char tensor_name[128];
        model::PreTransformerLayer * layer_weights = &weights->pre_transformer_layers[layer];

        // Layer norms
        snprintf(tensor_name, sizeof(tensor_name), "decoder_pre_transformer_l%d_input_ln_weight", layer);
        LOAD_TENSOR(layer_weights->input_ln_weight, tensor_name);

        snprintf(tensor_name, sizeof(tensor_name), "decoder_pre_transformer_l%d_post_ln_weight", layer);
        LOAD_TENSOR(layer_weights->post_ln_weight, tensor_name);

        // Self-attention
        snprintf(tensor_name, sizeof(tensor_name), "decoder_pre_transformer_l%d_attn_q_weight", layer);
        LOAD_TENSOR(layer_weights->attn_q_weight, tensor_name);

        snprintf(tensor_name, sizeof(tensor_name), "decoder_pre_transformer_l%d_attn_k_weight", layer);
        LOAD_TENSOR(layer_weights->attn_k_weight, tensor_name);

        snprintf(tensor_name, sizeof(tensor_name), "decoder_pre_transformer_l%d_attn_v_weight", layer);
        LOAD_TENSOR(layer_weights->attn_v_weight, tensor_name);

        snprintf(tensor_name, sizeof(tensor_name), "decoder_pre_transformer_l%d_attn_o_weight", layer);
        LOAD_TENSOR(layer_weights->attn_o_weight, tensor_name);

        // FFN
        snprintf(tensor_name, sizeof(tensor_name), "decoder_pre_transformer_l%d_ffn_gate_weight", layer);
        LOAD_TENSOR(layer_weights->ffn_gate_weight, tensor_name);

        snprintf(tensor_name, sizeof(tensor_name), "decoder_pre_transformer_l%d_ffn_up_weight", layer);
        LOAD_TENSOR(layer_weights->ffn_up_weight, tensor_name);

        snprintf(tensor_name, sizeof(tensor_name), "decoder_pre_transformer_l%d_ffn_down_weight", layer);
        LOAD_TENSOR(layer_weights->ffn_down_weight, tensor_name);

        // Layer scales
        snprintf(tensor_name, sizeof(tensor_name), "decoder_pre_transformer_l%d_attn_scale", layer);
        LOAD_TENSOR(layer_weights->attn_scale, tensor_name);

        snprintf(tensor_name, sizeof(tensor_name), "decoder_pre_transformer_l%d_ffn_scale", layer);
        LOAD_TENSOR(layer_weights->ffn_scale, tensor_name);
    }

    // Load causal conv
    LOAD_TENSOR(weights->causal_conv_weight, "decoder_causal_conv_weight");
    LOAD_TENSOR(weights->causal_conv_bias, "decoder_causal_conv_bias");

    // Load 4 upsample stages
    for (int stage = 0; stage < 4; stage++) {
        char tensor_name[128];

        // Load SnakeBeta alpha/beta (before upsample conv)
        snprintf(tensor_name, sizeof(tensor_name), "decoder_upsample_%d_alpha", stage);
        LOAD_TENSOR(weights->upsample_alphas[stage], tensor_name);

        snprintf(tensor_name, sizeof(tensor_name), "decoder_upsample_%d_beta", stage);
        LOAD_TENSOR(weights->upsample_betas[stage], tensor_name);

        // Load upsample conv weight/bias
        snprintf(tensor_name, sizeof(tensor_name), "decoder_upsample_%d_weight", stage);
        LOAD_TENSOR(weights->upsample_weights[stage], tensor_name);

        snprintf(tensor_name, sizeof(tensor_name), "decoder_upsample_%d_bias", stage);
        LOAD_TENSOR(weights->upsample_biases[stage], tensor_name);
    }

    // Load final SnakeBeta
    LOAD_TENSOR(weights->final_snake_alpha, "decoder_final_snake_alpha");
    LOAD_TENSOR(weights->final_snake_beta, "decoder_final_snake_beta");

    // Load final conv
    LOAD_TENSOR(weights->final_conv_weight, "decoder_final_conv_weight");
    LOAD_TENSOR(weights->final_conv_bias, "decoder_final_conv_bias");

    #undef LOAD_TENSOR

    // Cleanup
    gguf_free(gguf_ctx);
    fclose(file);

    return true;
}

} // namespace io
} // namespace leaxer_qwen
