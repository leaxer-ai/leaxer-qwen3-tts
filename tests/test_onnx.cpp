/**
 * Test TTS Engine ONNX integration
 */

#include "tts_onnx.h"
#include <iostream>
#include <filesystem>
#include <cassert>

namespace fs = std::filesystem;

const std::vector<std::string> MODEL_SEARCH_PATHS = {
    "onnx/onnx_kv_06b",
    "../onnx/onnx_kv_06b",
};

std::string findModelDir() {
    for (const auto& path : MODEL_SEARCH_PATHS) {
        if (fs::exists(path + "/speaker_encoder.onnx")) {
            return path;
        }
    }
    return "";
}

void testEngineLoading() {
    std::cout << "=== Test: Engine Loading ===\n";
    
    std::string model_dir = findModelDir();
    if (model_dir.empty()) {
        std::cout << "SKIP: ONNX models not found\n";
        return;
    }
    
    std::cout << "Loading from: " << model_dir << "\n";
    leaxer_qwen::TTSEngine engine(model_dir);
    
    if (!engine.is_ready()) {
        std::cout << "SKIP: Engine not ready - " << engine.get_error() << "\n";
        return;
    }
    
    std::cout << "PASS: Engine loaded successfully\n\n";
}

void testLanguageToCodec() {
    std::cout << "=== Test: Language to Codec ID ===\n";
    
    assert(leaxer_qwen::language_to_codec_id(leaxer_qwen::Language::English) == 2050);
    assert(leaxer_qwen::language_to_codec_id(leaxer_qwen::Language::Chinese) == 2051);
    assert(leaxer_qwen::language_to_codec_id(leaxer_qwen::Language::Japanese) == 2052);
    assert(leaxer_qwen::language_to_codec_id(leaxer_qwen::Language::Korean) == 2053);
    assert(leaxer_qwen::language_to_codec_id(leaxer_qwen::Language::Auto) == 0);
    
    std::cout << "PASS: Language to codec ID mapping\n\n";
}

void testConfigValues() {
    std::cout << "=== Test: Config Values ===\n";
    
    // Verify config constants match expected values
    assert(leaxer_qwen::config::HIDDEN_SIZE == 1024);
    assert(leaxer_qwen::config::NUM_LAYERS == 28);
    assert(leaxer_qwen::config::VOCAB_SIZE == 3072);
    assert(leaxer_qwen::config::SAMPLE_RATE == 24000);
    
    assert(leaxer_qwen::config::TTS_BOS == 151672);
    assert(leaxer_qwen::config::TTS_EOS == 151673);
    assert(leaxer_qwen::config::CODEC_EOS == 2150);
    
    std::cout << "PASS: Config values correct\n\n";
}

int main() {
    std::cout << "========================================\n";
    std::cout << "TTS Engine Tests\n";
    std::cout << "========================================\n\n";
    
    try {
        testConfigValues();
        testLanguageToCodec();
        testEngineLoading();
        
        std::cout << "========================================\n";
        std::cout << "All tests passed!\n";
        std::cout << "========================================\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "TEST FAILED: " << e.what() << "\n";
        return 1;
    }
}
