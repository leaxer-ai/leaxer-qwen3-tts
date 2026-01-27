// Test WAV writer with 1-second sine wave

#include <cstdio>
#include <cmath>
#include <vector>

// Forward declaration
namespace leaxer_qwen {
namespace io {
int write_wav(const char* path, const float* audio, size_t n_samples, int sample_rate);
}
}

int main() {
    // Generate 1-second sine wave at 440 Hz (A4 note)
    const int sample_rate = 24000;
    const int duration_sec = 1;
    const int n_samples = sample_rate * duration_sec;
    const float frequency = 440.0f;  // Hz
    const float amplitude = 0.5f;    // Keep modest to avoid clipping

    std::vector<float> audio(n_samples);

    for (int i = 0; i < n_samples; i++) {
        float t = static_cast<float>(i) / sample_rate;
        audio[i] = amplitude * sinf(2.0f * M_PI * frequency * t);
    }

    // Write to output/test.wav
    int result = leaxer_qwen::io::write_wav("output/test_wav.wav", audio.data(), n_samples, sample_rate);

    if (result != 0) {
        fprintf(stderr, "Failed to write output/test_wav.wav\n");
        return 1;
    }

    printf("Successfully wrote output/test_wav.wav (1-second 440 Hz sine wave)\n");
    return 0;
}
