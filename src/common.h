// Common ggml context helpers
// Provides simplified context creation/destruction for consistent memory management

#ifndef LEAXER_QWEN_COMMON_H
#define LEAXER_QWEN_COMMON_H

#include "ggml.h"
#include <cstddef>

namespace leaxer_qwen {

// Create ggml context with specified memory size
// Returns nullptr on failure
inline ggml_context* create_ggml_context(size_t mem_size) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    return ggml_init(params);
}

// Free ggml context
// Safe to call with nullptr
inline void free_ggml_context(ggml_context* ctx) {
    if (ctx) {
        ggml_free(ctx);
    }
}

} // namespace leaxer_qwen

#endif // LEAXER_QWEN_COMMON_H
