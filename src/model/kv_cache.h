// KV Cache for efficient autoregressive generation
// Stores key and value tensors from previous forward passes
// Enables O(n) instead of O(n²) generation

#ifndef LEAXER_QWEN_KV_CACHE_H
#define LEAXER_QWEN_KV_CACHE_H

#include "ggml.h"
#include <cstdlib>
#include <cstring>

namespace leaxer_qwen {
namespace model {

// KV Cache for a single layer
struct LayerKVCache {
    float * k_cache;  // [num_kv_heads, head_dim, max_seq_len]
    float * v_cache;  // [num_kv_heads, head_dim, max_seq_len]
    int seq_len;      // Current cached sequence length
};

// KV Cache for all layers
struct KVCache {
    LayerKVCache * layers;
    int n_layers;
    int num_kv_heads;
    int head_dim;
    int max_seq_len;

    // Allocate cache for all layers
    static KVCache * create(int n_layers, int num_kv_heads, int head_dim, int max_seq_len) {
        KVCache * cache = (KVCache *)calloc(1, sizeof(KVCache));
        if (!cache) return nullptr;

        cache->n_layers = n_layers;
        cache->num_kv_heads = num_kv_heads;
        cache->head_dim = head_dim;
        cache->max_seq_len = max_seq_len;

        cache->layers = (LayerKVCache *)calloc(n_layers, sizeof(LayerKVCache));
        if (!cache->layers) {
            free(cache);
            return nullptr;
        }

        // Allocate K and V buffers for each layer
        size_t kv_size = (size_t)num_kv_heads * head_dim * max_seq_len * sizeof(float);
        for (int i = 0; i < n_layers; i++) {
            cache->layers[i].k_cache = (float *)calloc(1, kv_size);
            cache->layers[i].v_cache = (float *)calloc(1, kv_size);
            cache->layers[i].seq_len = 0;

            if (!cache->layers[i].k_cache || !cache->layers[i].v_cache) {
                // Cleanup on failure
                for (int j = 0; j <= i; j++) {
                    free(cache->layers[j].k_cache);
                    free(cache->layers[j].v_cache);
                }
                free(cache->layers);
                free(cache);
                return nullptr;
            }
        }

        return cache;
    }

    // Free all cache memory
    static void destroy(KVCache * cache) {
        if (!cache) return;
        if (cache->layers) {
            for (int i = 0; i < cache->n_layers; i++) {
                free(cache->layers[i].k_cache);
                free(cache->layers[i].v_cache);
            }
            free(cache->layers);
        }
        free(cache);
    }

    // Reset cache (clear all stored KV)
    void reset() {
        for (int i = 0; i < n_layers; i++) {
            layers[i].seq_len = 0;
        }
    }

    // Append new K/V values to layer cache
    // new_k, new_v: [head_dim, new_tokens, num_kv_heads] in F32
    void append(int layer_idx, const float * new_k, const float * new_v, int new_tokens) {
        if (layer_idx < 0 || layer_idx >= n_layers) return;

        LayerKVCache * lc = &layers[layer_idx];
        int old_len = lc->seq_len;
        int new_len = old_len + new_tokens;

        if (new_len > max_seq_len) {
            // Would overflow - truncate (drop oldest tokens)
            int keep = max_seq_len - new_tokens;
            if (keep > 0) {
                // Shift existing data
                size_t shift_size = (size_t)num_kv_heads * head_dim * keep * sizeof(float);
                size_t drop_offset = (size_t)num_kv_heads * head_dim * (old_len - keep) * sizeof(float);
                memmove(lc->k_cache, lc->k_cache + drop_offset / sizeof(float), shift_size);
                memmove(lc->v_cache, lc->v_cache + drop_offset / sizeof(float), shift_size);
                old_len = keep;
            } else {
                old_len = 0;
            }
            new_len = old_len + new_tokens;
        }

        // Copy new K/V to cache
        // Input layout: [head_dim, new_tokens, num_kv_heads]
        // Cache layout: [num_kv_heads, head_dim, seq_len] - need to transpose
        // Actually, let's keep same layout for simplicity: [head_dim, seq_len, num_kv_heads]
        size_t copy_size = (size_t)head_dim * new_tokens * num_kv_heads * sizeof(float);
        size_t offset = (size_t)head_dim * old_len * num_kv_heads;

        memcpy(lc->k_cache + offset, new_k, copy_size);
        memcpy(lc->v_cache + offset, new_v, copy_size);

        lc->seq_len = new_len;
    }

    // Get current sequence length for a layer
    int get_seq_len(int layer_idx) const {
        if (layer_idx < 0 || layer_idx >= n_layers) return 0;
        return layers[layer_idx].seq_len;
    }

    // Create ggml tensors from cached K/V for attention computation
    // Returns tensors with shape [head_dim, cached_seq_len, num_kv_heads]
    struct ggml_tensor * get_k_tensor(struct ggml_context * ctx, int layer_idx) const {
        if (layer_idx < 0 || layer_idx >= n_layers) return nullptr;

        const LayerKVCache * lc = &layers[layer_idx];
        if (lc->seq_len == 0) return nullptr;

        struct ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                                     head_dim, lc->seq_len, num_kv_heads);
        memcpy(k->data, lc->k_cache, (size_t)head_dim * lc->seq_len * num_kv_heads * sizeof(float));
        return k;
    }

    struct ggml_tensor * get_v_tensor(struct ggml_context * ctx, int layer_idx) const {
        if (layer_idx < 0 || layer_idx >= n_layers) return nullptr;

        const LayerKVCache * lc = &layers[layer_idx];
        if (lc->seq_len == 0) return nullptr;

        struct ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                                     head_dim, lc->seq_len, num_kv_heads);
        memcpy(v->data, lc->v_cache, (size_t)head_dim * lc->seq_len * num_kv_heads * sizeof(float));
        return v;
    }
};

} // namespace model
} // namespace leaxer_qwen

#endif // LEAXER_QWEN_KV_CACHE_H
