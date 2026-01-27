// GGUF Model Loader
// Loads Qwen3-TTS weights from GGUF format

#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>

namespace leaxer_qwen {
namespace io {

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

    // Initialize GGUF context (metadata only, no tensor allocation)
    struct gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ nullptr,
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
    enum ggml_type tensor_type = gguf_get_tensor_type(gguf_ctx, tensor_id);
    size_t tensor_offset = gguf_get_tensor_offset(gguf_ctx, tensor_id);
    size_t tensor_size = gguf_get_tensor_size(gguf_ctx, tensor_id);
    size_t data_offset = gguf_get_data_offset(gguf_ctx);

    // Create tensor in provided context
    // Note: This is a simplified version that creates a 1D tensor
    // A full implementation would parse dimensions from GGUF metadata
    size_t n_elements = tensor_size / ggml_type_size(tensor_type);
    struct ggml_tensor * tensor = ggml_new_tensor_1d(ctx, tensor_type, n_elements);
    if (!tensor) {
        fprintf(stderr, "Failed to create tensor in context\n");
        gguf_free(gguf_ctx);
        fclose(file);
        return nullptr;
    }

    // Set tensor name
    ggml_set_name(tensor, tensor_name);

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

} // namespace io
} // namespace leaxer_qwen
