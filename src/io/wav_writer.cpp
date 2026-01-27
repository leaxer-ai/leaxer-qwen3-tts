// WAV File Writer
// Writes float32 audio to 16-bit PCM WAV file

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <algorithm>

namespace leaxer_qwen {
namespace io {

// WAV header structure
#pragma pack(push, 1)
struct wav_header {
    char     riff[4];        // "RIFF"
    uint32_t file_size;      // File size - 8
    char     wave[4];        // "WAVE"
    char     fmt[4];         // "fmt "
    uint32_t fmt_size;       // 16 for PCM
    uint16_t audio_format;   // 1 for PCM
    uint16_t num_channels;   // 1 for mono
    uint32_t sample_rate;    // 24000
    uint32_t byte_rate;      // sample_rate * num_channels * bytes_per_sample
    uint16_t block_align;    // num_channels * bytes_per_sample
    uint16_t bits_per_sample;// 16
    char     data[4];        // "data"
    uint32_t data_size;      // num_samples * num_channels * bytes_per_sample
};
#pragma pack(pop)

int write_wav(const char* path, const float* audio, size_t n_samples, int sample_rate) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        return -1;
    }

    // Find peak amplitude for normalization
    float peak = 0.0f;
    for (size_t i = 0; i < n_samples; i++) {
        float abs_val = audio[i] < 0 ? -audio[i] : audio[i];
        if (abs_val > peak) peak = abs_val;
    }

    // Calculate normalization factor (target peak at 0.95 to leave headroom)
    float norm_factor = 1.0f;
    if (peak > 0.0001f) {
        norm_factor = 0.95f / peak;
    }

    // Debug: print normalization info
    printf("  wav: peak=%.4f, norm_factor=%.2f\n", peak, norm_factor);

    // Prepare header
    wav_header header;
    std::memcpy(header.riff, "RIFF", 4);
    std::memcpy(header.wave, "WAVE", 4);
    std::memcpy(header.fmt, "fmt ", 4);
    std::memcpy(header.data, "data", 4);

    header.fmt_size = 16;
    header.audio_format = 1;  // PCM
    header.num_channels = 1;  // Mono
    header.sample_rate = sample_rate;
    header.bits_per_sample = 16;
    header.block_align = header.num_channels * (header.bits_per_sample / 8);
    header.byte_rate = header.sample_rate * header.block_align;
    header.data_size = n_samples * header.block_align;
    header.file_size = sizeof(wav_header) - 8 + header.data_size;

    // Write header
    fwrite(&header, sizeof(wav_header), 1, f);

    // Convert float32 to int16 with normalization
    for (size_t i = 0; i < n_samples; i++) {
        float sample = audio[i] * norm_factor;
        // Clamp to [-1, 1]
        sample = std::max(-1.0f, std::min(1.0f, sample));
        // Convert to int16
        int16_t sample_int = static_cast<int16_t>(sample * 32767.0f);
        fwrite(&sample_int, sizeof(int16_t), 1, f);
    }

    fclose(f);
    return 0;
}

} // namespace io
} // namespace leaxer_qwen
