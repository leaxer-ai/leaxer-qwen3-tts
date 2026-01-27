// Test speaker embedding lookup for CustomVoice model
// Verifies that speaker names map to correct embeddings with fallback

#include "test_utils.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "model/qwen_tts.cpp"
#include <cstdio>
#include <cstring>

bool test_speaker_lookup() {
    using namespace leaxer_qwen::test;

    printf("Testing speaker embedding lookup...\n");

    // Create ggml context
    size_t mem_size = 10 * 1024 * 1024;  // 10MB
    struct ggml_init_params params = {
        .mem_size   = mem_size,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * ctx = ggml_init(params);
    TEST_ASSERT(ctx != nullptr, "Failed to create ggml context");

    // Create mock speaker embeddings tensor
    // Shape: [embedding_dim, n_speakers]
    // For testing, use 256-dim embeddings and 7 speakers
    int embedding_dim = 256;
    int n_speakers = 7;

    struct ggml_tensor * speaker_embeddings = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, embedding_dim, n_speakers);

    // Fill with unique values per speaker (speaker_idx * 1000 + embedding_idx)
    float * data = (float *)speaker_embeddings->data;
    for (int s = 0; s < n_speakers; s++) {
        for (int e = 0; e < embedding_dim; e++) {
            data[s * embedding_dim + e] = (float)(s * 1000 + e);
        }
    }

    // Test 1: Valid speaker name (aiden -> index 0)
    struct ggml_tensor * emb_aiden = leaxer_qwen::model::get_speaker_embedding(
        ctx, "aiden", speaker_embeddings);
    TEST_ASSERT(emb_aiden != nullptr, "Failed to get aiden embedding");
    TEST_ASSERT(emb_aiden->ne[0] == embedding_dim, "Wrong embedding dimension");

    // Verify it's from speaker 0
    float * emb_data = (float *)emb_aiden->data;
    TEST_ASSERT(fabs(emb_data[0] - 0.0f) < 1e-6f, "aiden embedding[0] should be 0");
    TEST_ASSERT(fabs(emb_data[1] - 1.0f) < 1e-6f, "aiden embedding[1] should be 1");
    TEST_ASSERT(fabs(emb_data[10] - 10.0f) < 1e-6f, "aiden embedding[10] should be 10");

    // Test 2: Another valid speaker (ryan -> index 1)
    struct ggml_tensor * emb_ryan = leaxer_qwen::model::get_speaker_embedding(
        ctx, "ryan", speaker_embeddings);
    TEST_ASSERT(emb_ryan != nullptr, "Failed to get ryan embedding");
    emb_data = (float *)emb_ryan->data;
    TEST_ASSERT(fabs(emb_data[0] - 1000.0f) < 1e-6f, "ryan embedding[0] should be 1000");
    TEST_ASSERT(fabs(emb_data[1] - 1001.0f) < 1e-6f, "ryan embedding[1] should be 1001");

    // Test 3: Case-insensitive (SERENA -> index 2)
    struct ggml_tensor * emb_serena = leaxer_qwen::model::get_speaker_embedding(
        ctx, "SERENA", speaker_embeddings);
    TEST_ASSERT(emb_serena != nullptr, "Failed to get SERENA embedding");
    emb_data = (float *)emb_serena->data;
    TEST_ASSERT(fabs(emb_data[0] - 2000.0f) < 1e-6f, "SERENA embedding[0] should be 2000");

    // Test 4: Mixed case (ViViAn -> index 3)
    struct ggml_tensor * emb_vivian = leaxer_qwen::model::get_speaker_embedding(
        ctx, "ViViAn", speaker_embeddings);
    TEST_ASSERT(emb_vivian != nullptr, "Failed to get ViViAn embedding");
    emb_data = (float *)emb_vivian->data;
    TEST_ASSERT(fabs(emb_data[0] - 3000.0f) < 1e-6f, "ViViAn embedding[0] should be 3000");

    // Test 5: aria (index 4)
    struct ggml_tensor * emb_aria = leaxer_qwen::model::get_speaker_embedding(
        ctx, "aria", speaker_embeddings);
    TEST_ASSERT(emb_aria != nullptr, "Failed to get aria embedding");
    emb_data = (float *)emb_aria->data;
    TEST_ASSERT(fabs(emb_data[0] - 4000.0f) < 1e-6f, "aria embedding[0] should be 4000");

    // Test 6: Unknown speaker (should fallback to default speaker 0)
    struct ggml_tensor * emb_unknown = leaxer_qwen::model::get_speaker_embedding(
        ctx, "unknown_speaker", speaker_embeddings);
    TEST_ASSERT(emb_unknown != nullptr, "Failed to get unknown speaker embedding");
    emb_data = (float *)emb_unknown->data;
    TEST_ASSERT(fabs(emb_data[0] - 0.0f) < 1e-6f, "Unknown speaker should fallback to speaker 0");
    TEST_ASSERT(fabs(emb_data[1] - 1.0f) < 1e-6f, "Unknown speaker embedding[1] should be 1");

    // Test 7: Empty string (should fallback to default)
    struct ggml_tensor * emb_empty = leaxer_qwen::model::get_speaker_embedding(
        ctx, "", speaker_embeddings);
    TEST_ASSERT(emb_empty != nullptr, "Failed to get empty string embedding");
    emb_data = (float *)emb_empty->data;
    TEST_ASSERT(fabs(emb_data[0] - 0.0f) < 1e-6f, "Empty string should fallback to speaker 0");

    ggml_free(ctx);

    TEST_PASS("All speaker embedding lookups succeeded");
    return true;
}

int main() {
    printf("leaxer-qwen speaker embedding test\n");
    printf("===================================\n\n");

    test_speaker_lookup();

    return leaxer_qwen::test::print_summary();
}
