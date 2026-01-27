// Placeholder test to verify test infrastructure works
// This test always passes

#include "test_utils.h"

bool test_infrastructure() {
    // Test that we can use the test utilities
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {1.0f, 2.0f, 3.0f};

    return leaxer_qwen::test::assert_tensor_close(a, b, TOL_EXACT, "infrastructure_self_compare");
}

bool test_tolerance() {
    // Test that tolerance works correctly
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {1.0f + 1e-7f, 2.0f + 1e-7f, 3.0f + 1e-7f};

    return leaxer_qwen::test::assert_tensor_close(a, b, TOL_EXACT, "tolerance_small_diff");
}

int main() {
    printf("leaxer-qwen test infrastructure\n");
    printf("================================\n\n");

    test_infrastructure();
    test_tolerance();

    return leaxer_qwen::test::print_summary();
}
