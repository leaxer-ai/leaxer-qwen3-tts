// Mel Spectrogram Extractor test

#include "test_utils.h"
#include "io/mel.h"
#include <cmath>
#include <cstdio>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Helper: Generate a sine wave
std::vector<float> generate_sine(float freq, float duration, int sample_rate) {
    size_t num_samples = static_cast<size_t>(duration * sample_rate);
    std::vector<float> audio(num_samples);
    for (size_t i = 0; i < num_samples; i++) {
        audio[i] = std::sin(2.0f * M_PI * freq * i / sample_rate);
    }
    return audio;
}

// Test MelExtractor construction with default config
bool test_mel_extractor_construction() {
    leaxer_qwen::io::MelConfig config;
    leaxer_qwen::io::MelExtractor extractor(config);

    // Should construct without throwing
    TEST_ASSERT(config.num_mels == 128, "default num_mels should be 128");
    TEST_ASSERT(config.sample_rate == 24000, "default sample_rate should be 24000");
    TEST_ASSERT(config.n_fft == 1024, "default n_fft should be 1024");
    TEST_ASSERT(config.hop_size == 256, "default hop_size should be 256");

    TEST_PASS("MelExtractor construction");
    return true;
}

// Test MelExtractor construction with custom config
bool test_mel_extractor_custom_config() {
    leaxer_qwen::io::MelConfig config;
    config.sample_rate = 16000;
    config.num_mels = 80;
    config.n_fft = 512;
    config.hop_size = 128;
    config.fmax = 8000.0f;

    leaxer_qwen::io::MelExtractor extractor(config);

    // Should construct with custom values
    TEST_ASSERT(extractor.num_mels() == 80, "custom num_mels should be 80");

    TEST_PASS("MelExtractor custom config");
    return true;
}

// Test extract produces correct shape
bool test_mel_extract_shape() {
    leaxer_qwen::io::MelConfig config;
    config.sample_rate = 24000;
    config.n_fft = 1024;
    config.hop_size = 256;
    config.win_size = 1024;
    config.num_mels = 128;

    leaxer_qwen::io::MelExtractor extractor(config);

    // Generate 0.5 seconds of audio at 24000 Hz = 12000 samples
    std::vector<float> audio = generate_sine(440.0f, 0.5f, 24000);
    printf("    Generated %zu samples of audio\n", audio.size());

    std::vector<float> mel = extractor.extract(audio);

    size_t num_frames = extractor.num_frames();
    size_t num_mels = extractor.num_mels();

    printf("    Extracted mel spectrogram: %zu mels x %zu frames\n", num_mels, num_frames);

    // Check shape
    TEST_ASSERT(num_mels == 128, "should have 128 mel bands");
    
    // Expected frames: (audio_len - win_size) / hop_size + 1
    // (12000 - 1024) / 256 + 1 = 43.85... -> 43 frames
    size_t expected_frames = (audio.size() - config.win_size) / config.hop_size + 1;
    TEST_ASSERT(num_frames == expected_frames, "should have correct number of frames");

    // Total size should be num_mels * num_frames
    TEST_ASSERT(mel.size() == num_mels * num_frames, "output size should be num_mels * num_frames");

    TEST_PASS("mel extraction shape");
    return true;
}

// Test extract produces valid values (not NaN/inf)
bool test_mel_extract_valid_values() {
    leaxer_qwen::io::MelConfig config;
    leaxer_qwen::io::MelExtractor extractor(config);

    std::vector<float> audio = generate_sine(440.0f, 0.1f, 24000);
    std::vector<float> mel = extractor.extract(audio);

    for (size_t i = 0; i < mel.size(); i++) {
        TEST_ASSERT(!std::isnan(mel[i]), "mel values should not be NaN");
        TEST_ASSERT(!std::isinf(mel[i]), "mel values should not be inf");
    }

    // Log mel values should be negative (since energy < 1 for normalized audio)
    // or close to a reasonable range
    float min_val = mel[0], max_val = mel[0];
    for (size_t i = 0; i < mel.size(); i++) {
        min_val = std::min(min_val, mel[i]);
        max_val = std::max(max_val, mel[i]);
    }
    printf("    Mel value range: [%.2f, %.2f]\n", min_val, max_val);

    // Log mel values should be in a reasonable range (log scale)
    TEST_ASSERT(max_val < 50.0f, "max mel value should be reasonable");
    TEST_ASSERT(min_val > -100.0f, "min mel value should be reasonable");

    TEST_PASS("mel extraction valid values");
    return true;
}

// Test extract with empty audio
bool test_mel_extract_empty() {
    leaxer_qwen::io::MelExtractor extractor(leaxer_qwen::io::MelConfig{});

    std::vector<float> empty_audio;
    std::vector<float> mel = extractor.extract(empty_audio);

    TEST_ASSERT(mel.empty(), "empty audio should produce empty mel spectrogram");
    TEST_PASS("mel extraction empty audio");
    return true;
}

// Test extract with very short audio (shorter than window)
bool test_mel_extract_short_audio() {
    leaxer_qwen::io::MelConfig config;
    config.win_size = 1024;
    leaxer_qwen::io::MelExtractor extractor(config);

    // Audio shorter than window size
    std::vector<float> short_audio(100, 0.5f);
    std::vector<float> mel = extractor.extract(short_audio);

    // Should still produce at least 1 frame (padded)
    TEST_ASSERT(extractor.num_frames() >= 1, "short audio should produce at least 1 frame");
    TEST_ASSERT(!mel.empty(), "short audio should produce some output");

    printf("    Short audio (%zu samples) produced %zu frames\n", 
           short_audio.size(), extractor.num_frames());

    TEST_PASS("mel extraction short audio");
    return true;
}

// Test that different frequencies produce different spectrograms
bool test_mel_extract_frequency_sensitivity() {
    leaxer_qwen::io::MelConfig config;
    config.sample_rate = 24000;
    leaxer_qwen::io::MelExtractor extractor(config);

    // Generate two different frequencies
    std::vector<float> audio_low = generate_sine(200.0f, 0.1f, 24000);   // Low frequency
    std::vector<float> audio_high = generate_sine(4000.0f, 0.1f, 24000); // High frequency

    std::vector<float> mel_low = extractor.extract(audio_low);
    std::vector<float> mel_high = extractor.extract(audio_high);

    // Both should have same shape
    TEST_ASSERT(mel_low.size() == mel_high.size(), "same length audio should produce same size mel");

    // But different content - compute average difference
    float diff = 0.0f;
    for (size_t i = 0; i < mel_low.size(); i++) {
        diff += std::fabs(mel_low[i] - mel_high[i]);
    }
    diff /= mel_low.size();

    printf("    Average mel difference between 200Hz and 4000Hz: %.4f\n", diff);
    TEST_ASSERT(diff > 0.1f, "different frequencies should produce different mel spectrograms");

    TEST_PASS("mel extraction frequency sensitivity");
    return true;
}

// Test consistency (same input -> same output)
bool test_mel_extract_consistency() {
    leaxer_qwen::io::MelExtractor extractor(leaxer_qwen::io::MelConfig{});

    std::vector<float> audio = generate_sine(440.0f, 0.1f, 24000);

    std::vector<float> mel1 = extractor.extract(audio);
    std::vector<float> mel2 = extractor.extract(audio);

    TEST_ASSERT(mel1.size() == mel2.size(), "same input should produce same size output");

    for (size_t i = 0; i < mel1.size(); i++) {
        TEST_ASSERT(mel1[i] == mel2[i], "same input should produce identical output");
    }

    TEST_PASS("mel extraction consistency");
    return true;
}

int main() {
    printf("leaxer-qwen mel spectrogram test\n");
    printf("=================================\n\n");

    test_mel_extractor_construction();
    test_mel_extractor_custom_config();
    test_mel_extract_shape();
    test_mel_extract_valid_values();
    test_mel_extract_empty();
    test_mel_extract_short_audio();
    test_mel_extract_frequency_sensitivity();
    test_mel_extract_consistency();

    return leaxer_qwen::test::print_summary();
}
