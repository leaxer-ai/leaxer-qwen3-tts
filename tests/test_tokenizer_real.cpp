// Real tokenizer test - compares against Python transformers fixtures

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
    "../onnx/models/Qwen3-TTS-12Hz-0.6B-Base",
    "../../onnx/models/Qwen3-TTS-12Hz-0.6B-Base",
    "../onnx/models/Qwen3-TTS-12Hz-0.6B-Base",
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

// Test data: Must match oracle.py test strings
// These test strings are tokenized using our C++ BPE implementation
// with regex pre-tokenization and compared against Python transformers output.
struct TestCase {
    const char* text;
    const char* fixture_name;
};

static const TestCase TEST_CASES[] = {
    {"hello", "tokenizer_test0.bin"},
    {"world", "tokenizer_test1.bin"},
    {"speech", "tokenizer_test2.bin"},
    {"synthesis", "tokenizer_test3.bin"},
    {"testing", "tokenizer_test4.bin"},
};

static const int NUM_TEST_CASES = sizeof(TEST_CASES) / sizeof(TEST_CASES[0]);

bool test_load_tokenizer() {
    std::string model_dir = findModelDir();
    if (model_dir.empty()) {
        printf("[SKIP] Model files not found in search paths\n");
        printf("  Searched:\n");
        for (const auto& path : MODEL_SEARCH_PATHS) {
            printf("    - %s\n", path.c_str());
        }
        return false;  // Cannot continue without tokenizer
    }
    
    std::string vocab_path = model_dir + "/vocab.json";
    bool loaded_vocab = leaxer_qwen::io::load_vocab(vocab_path);
    TEST_ASSERT(loaded_vocab, "should load vocab.json");

    std::string merges_path = model_dir + "/merges.txt";
    bool loaded_merges = leaxer_qwen::io::load_merges(merges_path);
    TEST_ASSERT(loaded_merges, "should load merges.txt");

    TEST_PASS("tokenizer loading");
    return true;
}

bool test_tokenize_against_fixture(const TestCase& test_case) {
    printf("\nTest: '%s'\n", test_case.text);

    // Load expected tokens from fixture
    auto expected_tokens = leaxer_qwen::test::load_fixture_i32(test_case.fixture_name);
    if (expected_tokens.empty()) {
        printf("[SKIP] Failed to load fixture %s\n", test_case.fixture_name);
        return true;  // Skip, not fail - fixtures may not exist
    }

    // Tokenize using C++ implementation
    auto got_tokens = leaxer_qwen::io::tokenize(test_case.text);

    // Compare lengths
    if (got_tokens.size() != expected_tokens.size()) {
        printf("[FAIL] Token count mismatch\n");
        printf("  Expected %zu tokens, got %zu tokens\n",
               expected_tokens.size(), got_tokens.size());
        printf("  Expected: ");
        for (size_t i = 0; i < expected_tokens.size() && i < 20; i++) {
            printf("%d ", expected_tokens[i]);
        }
        printf("\n  Got:      ");
        for (size_t i = 0; i < got_tokens.size() && i < 20; i++) {
            printf("%d ", got_tokens[i]);
        }
        printf("\n");
        leaxer_qwen::test::g_tests_failed++;
        return false;
    }

    // Compare token values
    bool all_match = true;
    for (size_t i = 0; i < got_tokens.size(); i++) {
        if (got_tokens[i] != expected_tokens[i]) {
            printf("[FAIL] Token mismatch at position %zu\n", i);
            printf("  Expected: %d\n", expected_tokens[i]);
            printf("  Got:      %d\n", got_tokens[i]);
            all_match = false;
            break;
        }
    }

    if (!all_match) {
        printf("  Full expected: ");
        for (size_t i = 0; i < expected_tokens.size(); i++) {
            printf("%d ", expected_tokens[i]);
        }
        printf("\n  Full got:      ");
        for (size_t i = 0; i < got_tokens.size(); i++) {
            printf("%d ", got_tokens[i]);
        }
        printf("\n");
        leaxer_qwen::test::g_tests_failed++;
        return false;
    }

    // Success
    printf("[PASS] Tokenization matches exactly (%zu tokens)\n", got_tokens.size());
    printf("  Tokens: ");
    for (size_t i = 0; i < got_tokens.size() && i < 20; i++) {
        printf("%d ", got_tokens[i]);
    }
    if (got_tokens.size() > 20) {
        printf("...");
    }
    printf("\n");

    leaxer_qwen::test::g_tests_passed++;
    return true;
}

int main() {
    printf("leaxer-qwen tokenizer real test\n");
    printf("Comparing C++ tokenizer against Python transformers\n");
    printf("====================================================\n\n");

    // Load tokenizer
    if (!test_load_tokenizer()) {
        printf("Failed to load tokenizer, aborting tests\n");
        return 1;
    }

    // Run tests against fixtures
    for (int i = 0; i < NUM_TEST_CASES; i++) {
        test_tokenize_against_fixture(TEST_CASES[i]);
    }

    return leaxer_qwen::test::print_summary();
}
