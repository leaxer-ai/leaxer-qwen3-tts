// BPE Tokenizer test

#include "test_utils.h"
#include <string>
#include <vector>
#include <cstdint>

// Forward declaration
namespace leaxer_qwen {
namespace io {
    std::vector<int32_t> tokenize(const std::string& text);
}
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
    TEST_ASSERT(tokens.size() == 5, "hello should produce 5 tokens");

    // Check that we get valid token IDs (basic byte-level encoding)
    // 'h' = 104, 'e' = 101, 'l' = 108, 'l' = 108, 'o' = 111
    TEST_ASSERT(tokens[0] == 104, "h should map to token ID 104");
    TEST_ASSERT(tokens[1] == 101, "e should map to token ID 101");
    TEST_ASSERT(tokens[2] == 108, "l should map to token ID 108");
    TEST_ASSERT(tokens[3] == 108, "l should map to token ID 108");
    TEST_ASSERT(tokens[4] == 111, "o should map to token ID 111");

    TEST_PASS("hello tokenization");
    return true;
}

bool test_tokenize_unicode() {
    // Test basic ASCII
    auto tokens = leaxer_qwen::io::tokenize("Hi!");
    TEST_ASSERT(tokens.size() == 3, "Hi! should produce 3 tokens");
    TEST_PASS("basic ASCII");
    return true;
}

int main() {
    printf("leaxer-qwen tokenizer test\n");
    printf("============================\n\n");

    test_tokenize_empty();
    test_tokenize_hello();
    test_tokenize_unicode();

    return leaxer_qwen::test::print_summary();
}
