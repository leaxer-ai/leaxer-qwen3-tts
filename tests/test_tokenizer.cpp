// BPE Tokenizer test

#include "test_utils.h"
#include <string>
#include <vector>
#include <cstdint>
#include <cstdio>

// Forward declaration
namespace leaxer_qwen {
namespace io {
    bool load_vocab(const std::string& vocab_path);
    bool load_merges(const std::string& merges_path);
    std::vector<int32_t> tokenize(const std::string& text);
    std::string token_to_string(int32_t id);
    int32_t string_to_token(const std::string& token);
}
}

bool test_load_vocab() {
    const char* vocab_path = "../models/Qwen3-TTS-12Hz-0.6B-CustomVoice/vocab.json";
    bool loaded = leaxer_qwen::io::load_vocab(vocab_path);
    TEST_ASSERT(loaded, "should load vocab.json");
    TEST_PASS("vocab loading");
    return true;
}

bool test_load_merges() {
    const char* merges_path = "../models/Qwen3-TTS-12Hz-0.6B-CustomVoice/merges.txt";
    bool loaded = leaxer_qwen::io::load_merges(merges_path);
    TEST_ASSERT(loaded, "should load merges.txt");
    TEST_PASS("merges loading");
    return true;
}

bool test_token_to_id() {
    // Test common tokens from vocab
    int32_t id_h = leaxer_qwen::io::string_to_token("h");
    TEST_ASSERT(id_h == 71, "h should map to token ID 71");

    int32_t id_e = leaxer_qwen::io::string_to_token("e");
    TEST_ASSERT(id_e == 68, "e should map to token ID 68");

    int32_t id_hello = leaxer_qwen::io::string_to_token("hello");
    TEST_ASSERT(id_hello > 0, "hello should exist as a token");

    TEST_PASS("token to ID");
    return true;
}

bool test_id_to_token() {
    // Test reverse mapping
    std::string token_h = leaxer_qwen::io::token_to_string(71);
    TEST_ASSERT(token_h == "h", "ID 71 should map to h");

    std::string token_e = leaxer_qwen::io::token_to_string(68);
    TEST_ASSERT(token_e == "e", "ID 68 should map to e");

    TEST_PASS("ID to token");
    return true;
}

bool test_tokenize_empty() {
    auto tokens = leaxer_qwen::io::tokenize("");
    TEST_ASSERT(tokens.empty(), "empty string should produce no tokens");
    TEST_PASS("empty string");
    return true;
}

bool test_tokenize_hello() {
    auto tokens = leaxer_qwen::io::tokenize("hello");

    TEST_ASSERT(!tokens.empty(), "should produce tokens");

    // With vocab loaded, "hello" should be a single token or multiple BPE tokens
    // The exact tokenization depends on the vocab
    printf("    'hello' tokenized to %zu tokens: ", tokens.size());
    for (size_t i = 0; i < tokens.size() && i < 10; i++) {
        printf("%d ", tokens[i]);
    }
    printf("\n");

    TEST_PASS("hello tokenization");
    return true;
}

bool test_tokenize_unicode() {
    // Test basic ASCII
    auto tokens = leaxer_qwen::io::tokenize("Hi!");
    TEST_ASSERT(!tokens.empty(), "Hi! should produce tokens");
    printf("    'Hi!' tokenized to %zu tokens\n", tokens.size());
    TEST_PASS("basic ASCII");
    return true;
}

int main() {
    printf("leaxer-qwen tokenizer test\n");
    printf("============================\n\n");

    test_load_vocab();
    test_load_merges();
    test_token_to_id();
    test_id_to_token();
    test_tokenize_empty();
    test_tokenize_hello();
    test_tokenize_unicode();

    return leaxer_qwen::test::print_summary();
}
