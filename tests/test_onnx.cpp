/**
 * Test ONNX Runtime integration
 * 
 * Tests:
 * 1. Loading an ONNX model (speaker_encoder.onnx)
 * 2. Model introspection (input/output names, shapes)
 * 3. Basic inference with dummy data
 */

#include "onnx/onnx_session.h"
#include "onnx/tensor_utils.h"

#include <iostream>
#include <filesystem>
#include <cassert>
#include <cmath>

namespace fs = std::filesystem;

// Test model paths (relative to build directory or project root)
const std::vector<std::string> MODEL_SEARCH_PATHS = {
    "hf_onnx_bundle/onnx_kv_06b/speaker_encoder.onnx",
    "../hf_onnx_bundle/onnx_kv_06b/speaker_encoder.onnx",
    "../../hf_onnx_bundle/onnx_kv_06b/speaker_encoder.onnx",
};

std::string findModel() {
    for (const auto& path : MODEL_SEARCH_PATHS) {
        if (fs::exists(path)) {
            return path;
        }
    }
    return "";
}

void testModelLoading() {
    std::cout << "=== Test: Model Loading ===\n";
    
    std::string model_path = findModel();
    if (model_path.empty()) {
        std::cout << "SKIP: speaker_encoder.onnx not found\n";
        std::cout << "  Searched paths:\n";
        for (const auto& path : MODEL_SEARCH_PATHS) {
            std::cout << "    - " << path << "\n";
        }
        return;
    }
    
    std::cout << "Loading model: " << model_path << "\n";
    
    leaxer::OnnxSession session(model_path);
    session.printModelInfo();
    
    // Verify we can get input/output names
    auto inputs = session.getInputNames();
    auto outputs = session.getOutputNames();
    
    assert(!inputs.empty() && "Model should have inputs");
    assert(!outputs.empty() && "Model should have outputs");
    
    std::cout << "PASS: Model loaded successfully\n\n";
}

void testTensorUtilities() {
    std::cout << "=== Test: Tensor Utilities ===\n";
    
    // Test float tensor creation
    std::vector<float> float_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<int64_t> shape = {2, 3};
    
    auto float_tensor = leaxer::tensor::createFloat(float_data, shape);
    
    // Verify shape
    auto result_shape = leaxer::tensor::getShape(float_tensor);
    assert(result_shape.size() == 2);
    assert(result_shape[0] == 2);
    assert(result_shape[1] == 3);
    
    // Verify data
    auto extracted = leaxer::tensor::extractFloat(float_tensor);
    assert(extracted.size() == 6);
    for (size_t i = 0; i < 6; ++i) {
        assert(std::abs(extracted[i] - float_data[i]) < 1e-6);
    }
    
    std::cout << "PASS: Float tensor creation/extraction\n";
    
    // Test int64 tensor creation
    std::vector<int64_t> int_data = {100, 200, 300};
    std::vector<int64_t> int_shape = {1, 3};
    
    auto int_tensor = leaxer::tensor::createInt64(int_data, int_shape);
    auto int_extracted = leaxer::tensor::extractInt64(int_tensor);
    
    assert(int_extracted.size() == 3);
    for (size_t i = 0; i < 3; ++i) {
        assert(int_extracted[i] == int_data[i]);
    }
    
    std::cout << "PASS: Int64 tensor creation/extraction\n\n";
}

void testInference() {
    std::cout << "=== Test: Inference ===\n";
    
    std::string model_path = findModel();
    if (model_path.empty()) {
        std::cout << "SKIP: speaker_encoder.onnx not found\n";
        return;
    }
    
    leaxer::OnnxSession session(model_path);
    
    // Get input shape to create proper dummy data
    auto input_names = session.getInputNames();
    auto input_shape = session.getInputShape(0);
    
    std::cout << "Input: " << input_names[0] << " shape [";
    for (size_t i = 0; i < input_shape.size(); ++i) {
        if (i > 0) std::cout << ", ";
        if (input_shape[i] < 0) {
            std::cout << "?";
        } else {
            std::cout << input_shape[i];
        }
    }
    std::cout << "]\n";
    
    // speaker_encoder.onnx expects [batch, frames, 128] mel spectrograms
    // Use small test values: batch=1, frames=100, mel_bins=128
    std::vector<int64_t> test_shape = {1, 100, 128};
    int64_t total_size = 1;
    for (auto dim : test_shape) {
        total_size *= dim;
    }
    
    std::cout << "Using test shape: [" << test_shape[0] << ", " 
              << test_shape[1] << ", " << test_shape[2] << "] = "
              << total_size << " elements\n";
    
    // Create dummy mel spectrogram data
    std::vector<float> dummy_input(static_cast<size_t>(total_size), 0.0f);
    
    // Fill with realistic mel spectrogram values (log scale, typically -10 to 0)
    for (size_t i = 0; i < dummy_input.size(); ++i) {
        dummy_input[i] = -5.0f + std::sin(static_cast<float>(i) * 0.01f) * 2.0f;
    }
    
    // Create input tensor
    std::vector<Ort::Value> inputs;
    inputs.push_back(leaxer::tensor::createFloat(dummy_input, test_shape));
    
    // Run inference
    std::cout << "Running inference...\n";
    auto outputs = session.run(inputs);
    
    // Check outputs
    assert(!outputs.empty() && "Should have output");
    
    auto output_shape = leaxer::tensor::getShape(outputs[0]);
    std::cout << "Output shape: [";
    for (size_t i = 0; i < output_shape.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << output_shape[i];
    }
    std::cout << "]\n";
    
    // Verify output contains valid data (not NaN/Inf)
    auto output_data = leaxer::tensor::extractFloat(outputs[0]);
    bool valid = true;
    for (float v : output_data) {
        if (std::isnan(v) || std::isinf(v)) {
            valid = false;
            break;
        }
    }
    assert(valid && "Output should not contain NaN/Inf");
    
    std::cout << "Output embedding size: " << output_data.size() << "\n";
    std::cout << "First 5 values: ";
    for (size_t i = 0; i < std::min(size_t(5), output_data.size()); ++i) {
        std::cout << output_data[i] << " ";
    }
    std::cout << "\n";
    
    std::cout << "PASS: Inference completed successfully\n\n";
}

int main() {
    std::cout << "========================================\n";
    std::cout << "ONNX Runtime Integration Tests\n";
    std::cout << "========================================\n\n";
    
    try {
        testTensorUtilities();
        testModelLoading();
        testInference();
        
        std::cout << "========================================\n";
        std::cout << "All ONNX tests passed!\n";
        std::cout << "========================================\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "TEST FAILED: " << e.what() << "\n";
        return 1;
    }
}
