// WAV Reader test

#include "test_utils.h"
#include "io/wav_reader.h"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Helper: Generate a sine wave
std::vector<float> generate_sine_wave(float freq, float duration, int sample_rate) {
    size_t num_samples = static_cast<size_t>(duration * sample_rate);
    std::vector<float> audio(num_samples);
    for (size_t i = 0; i < num_samples; i++) {
        audio[i] = std::sin(2.0f * M_PI * freq * i / sample_rate);
    }
    return audio;
}

// Helper: Write a minimal valid WAV file (16-bit PCM mono)
bool write_test_wav(const std::string& path, const std::vector<float>& audio, int sample_rate) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;

    uint16_t num_channels = 1;
    uint16_t bits_per_sample = 16;
    uint32_t byte_rate = sample_rate * num_channels * bits_per_sample / 8;
    uint16_t block_align = num_channels * bits_per_sample / 8;
    uint32_t data_size = static_cast<uint32_t>(audio.size() * sizeof(int16_t));
    uint32_t file_size = 36 + data_size;

    // RIFF header
    fwrite("RIFF", 1, 4, f);
    fwrite(&file_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f);

    // fmt chunk
    fwrite("fmt ", 1, 4, f);
    uint32_t fmt_size = 16;
    fwrite(&fmt_size, 4, 1, f);
    uint16_t audio_format = 1;  // PCM
    fwrite(&audio_format, 2, 1, f);
    fwrite(&num_channels, 2, 1, f);
    uint32_t sr = static_cast<uint32_t>(sample_rate);
    fwrite(&sr, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    fwrite(&block_align, 2, 1, f);
    fwrite(&bits_per_sample, 2, 1, f);

    // data chunk
    fwrite("data", 1, 4, f);
    fwrite(&data_size, 4, 1, f);

    // Write samples as 16-bit PCM
    for (size_t i = 0; i < audio.size(); i++) {
        float clamped = std::max(-1.0f, std::min(1.0f, audio[i]));
        int16_t sample = static_cast<int16_t>(clamped * 32767.0f);
        fwrite(&sample, sizeof(int16_t), 1, f);
    }

    fclose(f);
    return true;
}

// Test reading a valid WAV file
bool test_read_valid_wav() {
    // Generate test audio: 440 Hz sine wave, 0.1 seconds, 16000 Hz
    int sample_rate = 16000;
    std::vector<float> original = generate_sine_wave(440.0f, 0.1f, sample_rate);

    // Write to temp file
    std::string test_path = "/tmp/test_wav_reader.wav";
    TEST_ASSERT(write_test_wav(test_path, original, sample_rate), "should write test WAV file");

    // Read it back
    int out_sr = 0;
    std::vector<float> audio = leaxer_qwen::io::read_wav(test_path, out_sr);

    TEST_ASSERT(!audio.empty(), "should read audio samples");
    TEST_ASSERT(out_sr == sample_rate, "should read correct sample rate");
    TEST_ASSERT(audio.size() == original.size(), "should read correct number of samples");

    // Check samples are approximately correct (16-bit quantization adds noise)
    float max_error = 0.0f;
    for (size_t i = 0; i < audio.size(); i++) {
        float error = std::fabs(audio[i] - original[i]);
        max_error = std::max(max_error, error);
    }
    TEST_ASSERT(max_error < 0.001f, "samples should match within quantization error");

    // Clean up
    remove(test_path.c_str());

    TEST_PASS("read valid WAV file");
    return true;
}

// Test reading non-existent file
bool test_read_missing_file() {
    int out_sr = 0;
    std::vector<float> audio = leaxer_qwen::io::read_wav("/nonexistent/path/to/file.wav", out_sr);

    TEST_ASSERT(audio.empty(), "missing file should return empty vector");
    TEST_PASS("handle missing file");
    return true;
}

// Test reading invalid file (not a WAV)
bool test_read_invalid_file() {
    // Write a non-WAV file
    std::string test_path = "/tmp/test_invalid.wav";
    FILE* f = fopen(test_path.c_str(), "wb");
    if (f) {
        fwrite("This is not a WAV file", 1, 22, f);
        fclose(f);
    }

    int out_sr = 0;
    std::vector<float> audio = leaxer_qwen::io::read_wav(test_path, out_sr);

    TEST_ASSERT(audio.empty(), "invalid file should return empty vector");

    // Clean up
    remove(test_path.c_str());

    TEST_PASS("handle invalid file");
    return true;
}

// Test resampling to higher sample rate (upsampling)
bool test_resample_upsample() {
    // Generate 100 samples at 8000 Hz -> resample to 16000 Hz
    std::vector<float> audio = generate_sine_wave(440.0f, 0.0125f, 8000);  // 100 samples
    size_t original_size = audio.size();

    std::vector<float> resampled = leaxer_qwen::io::resample(audio, 8000, 16000);

    // Should roughly double the number of samples
    TEST_ASSERT(resampled.size() == original_size * 2, "upsampling 2x should double samples");

    // Resampled audio should still be valid (no NaN/inf)
    for (size_t i = 0; i < resampled.size(); i++) {
        TEST_ASSERT(!std::isnan(resampled[i]) && !std::isinf(resampled[i]),
                   "resampled audio should be valid numbers");
    }

    // Check values are in valid range
    for (size_t i = 0; i < resampled.size(); i++) {
        TEST_ASSERT(resampled[i] >= -1.1f && resampled[i] <= 1.1f,
                   "resampled audio should be in valid range");
    }

    TEST_PASS("upsampling");
    return true;
}

// Test resampling to lower sample rate (downsampling)
bool test_resample_downsample() {
    // Generate 200 samples at 16000 Hz -> resample to 8000 Hz
    std::vector<float> audio = generate_sine_wave(440.0f, 0.0125f, 16000);  // 200 samples
    size_t original_size = audio.size();

    std::vector<float> resampled = leaxer_qwen::io::resample(audio, 16000, 8000);

    // Should roughly halve the number of samples
    TEST_ASSERT(resampled.size() == original_size / 2, "downsampling 2x should halve samples");

    // Resampled audio should still be valid
    for (size_t i = 0; i < resampled.size(); i++) {
        TEST_ASSERT(!std::isnan(resampled[i]) && !std::isinf(resampled[i]),
                   "resampled audio should be valid numbers");
    }

    TEST_PASS("downsampling");
    return true;
}

// Test resampling with same sample rate (no-op)
bool test_resample_same_rate() {
    std::vector<float> audio = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};

    std::vector<float> resampled = leaxer_qwen::io::resample(audio, 16000, 16000);

    TEST_ASSERT(resampled.size() == audio.size(), "same rate should preserve size");
    for (size_t i = 0; i < audio.size(); i++) {
        TEST_ASSERT(resampled[i] == audio[i], "same rate should preserve samples");
    }

    TEST_PASS("resample with same rate");
    return true;
}

// Test resampling empty audio
bool test_resample_empty() {
    std::vector<float> audio;
    std::vector<float> resampled = leaxer_qwen::io::resample(audio, 16000, 8000);

    TEST_ASSERT(resampled.empty(), "empty audio should resample to empty");
    TEST_PASS("resample empty audio");
    return true;
}

int main() {
    printf("leaxer-qwen WAV reader test\n");
    printf("============================\n\n");

    test_read_valid_wav();
    test_read_missing_file();
    test_read_invalid_file();
    test_resample_upsample();
    test_resample_downsample();
    test_resample_same_rate();
    test_resample_empty();

    return leaxer_qwen::test::print_summary();
}
