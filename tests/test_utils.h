#ifndef LEAXER_QWEN_TEST_UTILS_H
#define LEAXER_QWEN_TEST_UTILS_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

// Tolerance levels for different test types
#define TOL_EXACT    1e-6f
#define TOL_TIGHT    1e-5f
#define TOL_RELAXED  1e-4f
#define TOL_AUDIO    1e-3f

namespace leaxer_qwen {
namespace test {

// Test result tracking
static int g_tests_passed = 0;
static int g_tests_failed = 0;

// Fixture path helper
inline std::string fixture_path(const char* name) {
    // Assumes running from build directory
    return std::string("../tests/fixtures/") + name;
}

// Load binary fixture (float32)
inline std::vector<float> load_fixture_f32(const char* name) {
    std::string path = fixture_path(name);
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "Failed to open fixture: %s\n", path.c_str());
        return {};
    }

    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);

    size_t n_floats = size / sizeof(float);
    std::vector<float> data(n_floats);
    fread(data.data(), sizeof(float), n_floats, f);
    fclose(f);

    printf("Loaded fixture %s: %zu floats\n", name, n_floats);
    return data;
}

// Load binary fixture (int32)
inline std::vector<int32_t> load_fixture_i32(const char* name) {
    std::string path = fixture_path(name);
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "Failed to open fixture: %s\n", path.c_str());
        return {};
    }

    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);

    size_t n_ints = size / sizeof(int32_t);
    std::vector<int32_t> data(n_ints);
    fread(data.data(), sizeof(int32_t), n_ints, f);
    fclose(f);

    printf("Loaded fixture %s: %zu ints\n", name, n_ints);
    return data;
}

// Compare tensors with tolerance
struct CompareResult {
    bool passed;
    float max_diff;
    float mean_diff;
    size_t first_mismatch_idx;
    float first_mismatch_expected;
    float first_mismatch_got;
};

inline CompareResult compare_tensors(
    const float* got,
    const float* expected,
    size_t n,
    float tolerance
) {
    CompareResult result = {true, 0.0f, 0.0f, 0, 0.0f, 0.0f};
    double sum_diff = 0.0;
    bool first_mismatch_found = false;

    for (size_t i = 0; i < n; i++) {
        float diff = std::fabs(got[i] - expected[i]);
        sum_diff += diff;

        if (diff > result.max_diff) {
            result.max_diff = diff;
        }

        if (diff > tolerance && !first_mismatch_found) {
            result.passed = false;
            first_mismatch_found = true;
            result.first_mismatch_idx = i;
            result.first_mismatch_expected = expected[i];
            result.first_mismatch_got = got[i];
        }
    }

    result.mean_diff = static_cast<float>(sum_diff / n);
    return result;
}

// Assert tensor close
inline bool assert_tensor_close(
    const float* got,
    const float* expected,
    size_t n,
    float tolerance,
    const char* test_name
) {
    CompareResult result = compare_tensors(got, expected, n, tolerance);

    if (result.passed) {
        printf("[PASS] %s (max_diff=%.2e, mean_diff=%.2e)\n",
               test_name, result.max_diff, result.mean_diff);
        g_tests_passed++;
        return true;
    } else {
        printf("[FAIL] %s\n", test_name);
        printf("  max_diff=%.2e, mean_diff=%.2e, tolerance=%.2e\n",
               result.max_diff, result.mean_diff, tolerance);
        printf("  First mismatch at index %zu:\n", result.first_mismatch_idx);
        printf("    expected: %f\n", result.first_mismatch_expected);
        printf("    got:      %f\n", result.first_mismatch_got);

        // Print first 10 values for debugging
        printf("  First 10 values:\n");
        printf("    expected: ");
        for (size_t i = 0; i < std::min(n, (size_t)10); i++) {
            printf("%.4f ", expected[i]);
        }
        printf("\n    got:      ");
        for (size_t i = 0; i < std::min(n, (size_t)10); i++) {
            printf("%.4f ", got[i]);
        }
        printf("\n");

        g_tests_failed++;
        return false;
    }
}

// Assert tensor close (vector version)
inline bool assert_tensor_close(
    const std::vector<float>& got,
    const std::vector<float>& expected,
    float tolerance,
    const char* test_name
) {
    if (got.size() != expected.size()) {
        printf("[FAIL] %s - size mismatch (got %zu, expected %zu)\n",
               test_name, got.size(), expected.size());
        g_tests_failed++;
        return false;
    }
    return assert_tensor_close(got.data(), expected.data(), got.size(), tolerance, test_name);
}

// Print test summary
inline int print_summary() {
    printf("\n========================================\n");
    printf("Test Summary:\n");
    printf("  Passed: %d\n", g_tests_passed);
    printf("  Failed: %d\n", g_tests_failed);
    printf("========================================\n");

    return g_tests_failed > 0 ? 1 : 0;
}

// Macro for simple assertions
#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("[FAIL] %s: %s\n", __func__, msg); \
        leaxer_qwen::test::g_tests_failed++; \
        return false; \
    } \
} while(0)

#define TEST_PASS(msg) do { \
    printf("[PASS] %s: %s\n", __func__, msg); \
    leaxer_qwen::test::g_tests_passed++; \
} while(0)

} // namespace test
} // namespace leaxer_qwen

#endif // LEAXER_QWEN_TEST_UTILS_H
