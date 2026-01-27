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

} // namespace io
} // namespace leaxer_qwen
