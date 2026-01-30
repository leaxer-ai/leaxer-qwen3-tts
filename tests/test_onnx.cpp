/**
 * Test ONNX Runtime integration
 * 
 * Tests:
 * 1. Tensor utilities (create/extract tensors)
 * 2. Model loading and introspection
 * 3. Inference with dummy data
 * 4. TTS-specific model tests (codec_embed, speaker_encoder)
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
    "hf_onnx_bundle/onnx_kv_06b",
    "../hf_onnx_bundle/onnx_kv_06b",
    "../../hf_onnx_bundle/onnx_kv_06b",
};

std::string findModelDir() {
    for (const auto& path : MODEL_SEARCH_PATHS) {
        if (fs::exists(path + "/speaker_encoder.onnx")) {
            return path;
        }
    }
    return "";
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
    
    std::cout << "PASS: Int64 tensor creation/extraction\n";
    
    // Test int32 tensor creation
    std::vector<int32_t> int32_data = {10, 20, 30, 40};
    std::vector<int64_t> int32_shape = {2, 2};
    
    auto int32_tensor = leaxer::tensor::createInt32(int32_data.data(), int32_shape);
    auto int32_extracted = leaxer::tensor::extractInt32(int32_tensor);
    
    assert(int32_extracted.size() == 4);
    for (size_t i = 0; i < 4; ++i) {
        assert(int32_extracted[i] == int32_data[i]);
    }
    
    std::cout << "PASS: Int32 tensor creation/extraction\n\n";
}

void testModelLoading() {
    std::cout << "=== Test: Model Loading ===\n";
    
    std::string model_dir = findModelDir();
    if (model_dir.empty()) {
        std::cout << "SKIP: ONNX models not found\n";
        std::cout << "  Searched paths:\n";
        for (const auto& path : MODEL_SEARCH_PATHS) {
            std::cout << "    - " << path << "\n";
        }
        return;
    }
    
    std::string model_path = model_dir + "/speaker_encoder.onnx";
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

void testSpeakerEncoderInference() {
    std::cout << "=== Test: Speaker Encoder Inference ===\n";
    
    std::string model_dir = findModelDir();
    if (model_dir.empty()) {
        std::cout << "SKIP: speaker_encoder.onnx not found\n";
        return;
    }
    
    std::string model_path = model_dir + "/speaker_encoder.onnx";
    leaxer::OnnxSession session(model_path);
    
    // speaker_encoder.onnx expects [batch, frames, 128] mel spectrograms
    // Use small test values: batch=1, frames=100, mel_bins=128
    std::vector<int64_t> test_shape = {1, 100, 128};
    int64_t total_size = 1;
    for (auto dim : test_shape) {
        total_size *= dim;
    }
    
    std::cout << "Input shape: [" << test_shape[0] << ", " 
              << test_shape[1] << ", " << test_shape[2] << "]\n";
    
    // Create dummy mel spectrogram data
    std::vector<float> dummy_input(static_cast<size_t>(total_size), 0.0f);
    for (size_t i = 0; i < dummy_input.size(); ++i) {
        dummy_input[i] = -5.0f + std::sin(static_cast<float>(i) * 0.01f) * 2.0f;
    }
    
    // Create input tensor
    std::vector<Ort::Value> inputs;
    inputs.push_back(leaxer::tensor::createFloat(dummy_input, test_shape));
    
    // Run inference
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
    
    // Verify output is [1, 1024] speaker embedding
    assert(output_shape.size() == 2);
    assert(output_shape[0] == 1);
    assert(output_shape[1] == 1024);
    
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
    
    std::cout << "First 5 values: ";
    for (size_t i = 0; i < std::min(size_t(5), output_data.size()); ++i) {
        std::cout << output_data[i] << " ";
    }
    std::cout << "\n";
    
    std::cout << "PASS: Speaker encoder inference\n\n";
}

void testCodecEmbedInference() {
    std::cout << "=== Test: Codec Embed Inference ===\n";
    
    std::string model_dir = findModelDir();
    if (model_dir.empty()) {
        std::cout << "SKIP: codec_embed.onnx not found\n";
        return;
    }
    
    std::string model_path = model_dir + "/codec_embed.onnx";
    if (!fs::exists(model_path)) {
        std::cout << "SKIP: codec_embed.onnx not found at " << model_path << "\n";
        return;
    }
    
    leaxer::OnnxSession session(model_path);
    session.printModelInfo();
    
    // codec_embed.onnx expects [batch, seq] int64 token IDs
    // and outputs [batch, seq, 1024] embeddings
    
    // Test with some codec token IDs (these are special tokens from config)
    // CODEC_BOS=2149, CODEC_PAD=2148, CODEC_NOTHINK=2155
    std::vector<int64_t> codec_ids = {2155, 2156, 2157, 2148, 2149};  // nothink, think_bos, think_eos, pad, bos
    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(codec_ids.size())};
    
    std::cout << "Input codec IDs: [";
    for (size_t i = 0; i < codec_ids.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << codec_ids[i];
    }
    std::cout << "]\n";
    
    // Create input tensor
    std::vector<Ort::Value> inputs;
    inputs.push_back(leaxer::tensor::createInt64(codec_ids, input_shape));
    
    // Run inference
    auto outputs = session.run(inputs);
    
    assert(!outputs.empty() && "Should have output");
    
    auto output_shape = leaxer::tensor::getShape(outputs[0]);
    std::cout << "Output shape: [";
    for (size_t i = 0; i < output_shape.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << output_shape[i];
    }
    std::cout << "]\n";
    
    // Verify output is [1, 5, 1024]
    assert(output_shape.size() == 3);
    assert(output_shape[0] == 1);
    assert(output_shape[1] == static_cast<int64_t>(codec_ids.size()));
    assert(output_shape[2] == 1024);
    
    // Verify output contains valid data
    auto output_data = leaxer::tensor::extractFloat(outputs[0]);
    bool valid = true;
    for (float v : output_data) {
        if (std::isnan(v) || std::isinf(v)) {
            valid = false;
            break;
        }
    }
    assert(valid && "Output should not contain NaN/Inf");
    
    std::cout << "First 5 values of first embedding: ";
    for (size_t i = 0; i < std::min(size_t(5), output_data.size()); ++i) {
        std::cout << output_data[i] << " ";
    }
    std::cout << "\n";
    
    std::cout << "PASS: Codec embed inference\n\n";
}

void testMultipleModelSessions() {
    std::cout << "=== Test: Multiple Model Sessions ===\n";
    
    std::string model_dir = findModelDir();
    if (model_dir.empty()) {
        std::cout << "SKIP: Models not found\n";
        return;
    }
    
    // Load multiple models simultaneously (as TTS engine would)
    std::string speaker_path = model_dir + "/speaker_encoder.onnx";
    std::string codec_path = model_dir + "/codec_embed.onnx";
    
    if (!fs::exists(speaker_path) || !fs::exists(codec_path)) {
        std::cout << "SKIP: Not all models available\n";
        return;
    }
    
    std::cout << "Loading speaker_encoder and codec_embed simultaneously...\n";
    
    leaxer::OnnxSession speaker_session(speaker_path);
    leaxer::OnnxSession codec_session(codec_path);
    
    // Run inference on both
    std::vector<float> mel_data(1 * 50 * 128, -3.0f);
    std::vector<Ort::Value> mel_inputs;
    mel_inputs.push_back(leaxer::tensor::createFloat(mel_data, {1, 50, 128}));
    auto speaker_output = speaker_session.run(mel_inputs);
    
    std::vector<int64_t> codec_ids = {2149, 2150};
    std::vector<Ort::Value> codec_inputs;
    codec_inputs.push_back(leaxer::tensor::createInt64(codec_ids, {1, 2}));
    auto codec_output = codec_session.run(codec_inputs);
    
    assert(!speaker_output.empty());
    assert(!codec_output.empty());
    
    auto spk_shape = leaxer::tensor::getShape(speaker_output[0]);
    auto codec_shape = leaxer::tensor::getShape(codec_output[0]);
    
    std::cout << "Speaker output shape: [" << spk_shape[0] << ", " << spk_shape[1] << "]\n";
    std::cout << "Codec output shape: [" << codec_shape[0] << ", " << codec_shape[1] << ", " << codec_shape[2] << "]\n";
    
    std::cout << "PASS: Multiple model sessions work correctly\n\n";
}

int main() {
    std::cout << "========================================\n";
    std::cout << "ONNX Runtime Integration Tests\n";
    std::cout << "========================================\n\n";
    
    try {
        testTensorUtilities();
        testModelLoading();
        testSpeakerEncoderInference();
        testCodecEmbedInference();
        testMultipleModelSessions();
        
        std::cout << "========================================\n";
        std::cout << "All ONNX tests passed!\n";
        std::cout << "========================================\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "TEST FAILED: " << e.what() << "\n";
        return 1;
    }
}
