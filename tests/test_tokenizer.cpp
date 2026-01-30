// BPE Tokenizer test

#include "test_utils.h"
#include "io/tokenizer.h"
#include <string>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <filesystem>

namespace fs = std::filesystem;

// Search paths for vocab/merges files (relative to build directory)
const std::vector<std::string> MODEL_SEARCH_PATHS = {
    "../hf_onnx_bundle/models/Qwen3-TTS-12Hz-0.6B-Base",
    "../../hf_onnx_bundle/models/Qwen3-TTS-12Hz-0.6B-Base",
    "../models/Qwen3-TTS-12Hz-0.6B-Base",
};

std::string findModelDir() {
    for (const auto& dir : MODEL_SEARCH_PATHS) {
        std::string vocab_path = dir + "/vocab.json";
        if (fs::exists(vocab_path)) {
            return dir;
        }
    }
    return "";
}

bool test_load_vocab() {
    std::string model_dir = findModelDir();
    if (model_dir.empty()) {
        printf("[SKIP] Vocab file not found in search paths\n");
        return true;  // Skip, not fail
    }
    
    std::string vocab_path = model_dir + "/vocab.json";
    bool loaded = leaxer_qwen::io::load_vocab(vocab_path);
    TEST_ASSERT(loaded, "should load vocab.json");
    TEST_PASS("vocab loading");
    return true;
}

bool test_load_merges() {
    std::string model_dir = findModelDir();
    if (model_dir.empty()) {
        printf("[SKIP] Merges file not found in search paths\n");
        return true;
    }
    
    std::string merges_path = model_dir + "/merges.txt";
    bool loaded = leaxer_qwen::io::load_merges(merges_path);
    TEST_ASSERT(loaded, "should load merges.txt");
    TEST_PASS("merges loading");
    return true;
}

bool test_tokenizer_ready() {
    TEST_ASSERT(leaxer_qwen::io::is_tokenizer_ready(), "tokenizer should be ready after loading");
    TEST_PASS("tokenizer ready check");
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

    // With BPE properly loaded, "hello" should produce fewer tokens than raw bytes
    printf("    'hello' tokenized to %zu tokens: ", tokens.size());
    for (size_t i = 0; i < tokens.size() && i < 10; i++) {
        printf("%d ", tokens[i]);
    }
    printf("\n");

    // Basic sanity: BPE should compress - "hello" (5 chars) should produce <= 5 tokens
    TEST_ASSERT(tokens.size() <= 5, "BPE should not produce more tokens than characters");

    TEST_PASS("hello tokenization");
    return true;
}

bool test_tokenize_world() {
    auto tokens = leaxer_qwen::io::tokenize("world");
    
    printf("    'world' tokenized to %zu tokens: ", tokens.size());
    for (size_t i = 0; i < tokens.size() && i < 10; i++) {
        printf("%d ", tokens[i]);
    }
    printf("\n");

    // BPE should compress - "world" (5 chars) should typically be 1 token
    TEST_ASSERT(tokens.size() <= 5, "BPE should not produce more tokens than characters");
    
    TEST_PASS("world tokenization");
    return true;
}

bool test_tokenize_hello_world() {
    auto tokens = leaxer_qwen::io::tokenize("Hello world");
    
    printf("    'Hello world' tokenized to %zu tokens: ", tokens.size());
    for (size_t i = 0; i < tokens.size() && i < 10; i++) {
        printf("%d ", tokens[i]);
    }
    printf("\n");

    // "Hello world" (11 chars including space) should compress with BPE
    // Typically becomes 2 tokens: "Hello" and " world"
    TEST_ASSERT(tokens.size() <= 11, "BPE should not produce more tokens than characters");
    
    TEST_PASS("Hello world tokenization");
    return true;
}

bool test_tokenize_sentence() {
    auto tokens = leaxer_qwen::io::tokenize("The quick brown fox jumps over the lazy dog.");
    
    printf("    Sentence tokenized to %zu tokens: ", tokens.size());
    for (size_t i = 0; i < std::min(tokens.size(), (size_t)10); i++) {
        printf("%d ", tokens[i]);
    }
    if (tokens.size() > 10) printf("...");
    printf("\n");

    // Should compress well - 44 chars should produce fewer tokens
    TEST_ASSERT(tokens.size() < 44, "BPE should compress English text");
    TEST_ASSERT(tokens.size() >= 8, "Should produce at least some tokens");
    
    TEST_PASS("sentence tokenization");
    return true;
}

bool test_roundtrip() {
    // Test that tokenizing and then looking up tokens makes sense
    std::string text = "test";
    auto tokens = leaxer_qwen::io::tokenize(text);
    
    printf("    '%s' -> tokens: ", text.c_str());
    for (auto t : tokens) {
        printf("%d ", t);
    }
    printf("\n");
    
    // Look up each token string
    printf("    Token strings: ");
    for (auto t : tokens) {
        std::string tok_str = leaxer_qwen::io::token_to_string(t);
        printf("'%s' ", tok_str.c_str());
    }
    printf("\n");
    
    TEST_ASSERT(!tokens.empty(), "Should produce tokens");
    TEST_PASS("roundtrip tokenization");
    return true;
}

int main() {
    printf("leaxer-qwen tokenizer test\n");
    printf("============================\n\n");

    // First load tokenizer
    if (!test_load_vocab()) return 1;
    if (!test_load_merges()) return 1;
    
    // Check it's ready
    if (!leaxer_qwen::io::is_tokenizer_ready()) {
        printf("[SKIP] Tokenizer not loaded, skipping tokenization tests\n");
        return 0;
    }
    
    test_tokenizer_ready();
    test_tokenize_empty();
    test_tokenize_hello();
    test_tokenize_world();
    test_tokenize_hello_world();
    test_tokenize_sentence();
    test_roundtrip();

    return leaxer_qwen::test::print_summary();
}
