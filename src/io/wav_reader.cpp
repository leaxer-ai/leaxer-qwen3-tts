// WAV File Reader Implementation
#include "wav_reader.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace leaxer_qwen {
namespace io {

// WAV header structure
#pragma pack(push, 1)
struct wav_header {
    char     riff[4];
    uint32_t file_size;
    char     wave[4];
    char     fmt[4];
    uint32_t fmt_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
};
#pragma pack(pop)

std::vector<float> read_wav(const std::string& path, int& out_sample_rate) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        return {};
    }

    // Read RIFF header
    char riff[4];
    uint32_t file_size;
    char wave[4];
    
    if (fread(riff, 1, 4, f) != 4 || std::memcmp(riff, "RIFF", 4) != 0) {
        fclose(f);
        return {};
    }
    fread(&file_size, 4, 1, f);
    if (fread(wave, 1, 4, f) != 4 || std::memcmp(wave, "WAVE", 4) != 0) {
        fclose(f);
        return {};
    }

    // Find and parse chunks
    uint16_t audio_format = 0;
    uint16_t num_channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    
    std::vector<uint8_t> audio_data;
    
    while (!feof(f)) {
        char chunk_id[4];
        uint32_t chunk_size;
        
        if (fread(chunk_id, 1, 4, f) != 4) break;
        if (fread(&chunk_size, 4, 1, f) != 1) break;
        
        if (std::memcmp(chunk_id, "fmt ", 4) == 0) {
            // Format chunk
            fread(&audio_format, 2, 1, f);
            fread(&num_channels, 2, 1, f);
            fread(&sample_rate, 4, 1, f);
            uint32_t byte_rate;
            fread(&byte_rate, 4, 1, f);
            uint16_t block_align;
            fread(&block_align, 2, 1, f);
            fread(&bits_per_sample, 2, 1, f);
            
            // Skip extra format bytes if present
            if (chunk_size > 16) {
                fseek(f, chunk_size - 16, SEEK_CUR);
            }
        } else if (std::memcmp(chunk_id, "data", 4) == 0) {
            // Data chunk
            audio_data.resize(chunk_size);
            fread(audio_data.data(), 1, chunk_size, f);
        } else {
            // Skip unknown chunk
            fseek(f, chunk_size, SEEK_CUR);
        }
    }
    
    fclose(f);
    
    // Validate format
    if (audio_format != 1 && audio_format != 3) {  // PCM or float
        return {};
    }
    if (num_channels == 0 || sample_rate == 0 || bits_per_sample == 0) {
        return {};
    }
    
    out_sample_rate = static_cast<int>(sample_rate);
    
    // Convert to float32 mono
    std::vector<float> samples;
    size_t bytes_per_sample = bits_per_sample / 8;
    size_t num_samples = audio_data.size() / (num_channels * bytes_per_sample);
    samples.reserve(num_samples);
    
    for (size_t i = 0; i < num_samples; i++) {
        float sample = 0.0f;
        
        // Average all channels to mono
        for (uint16_t ch = 0; ch < num_channels; ch++) {
            size_t offset = (i * num_channels + ch) * bytes_per_sample;
            
            if (audio_format == 3) {  // Float
                if (bits_per_sample == 32) {
                    float val;
                    std::memcpy(&val, &audio_data[offset], 4);
                    sample += val;
                }
            } else if (bits_per_sample == 16) {
                int16_t val;
                std::memcpy(&val, &audio_data[offset], 2);
                sample += val / 32768.0f;
            } else if (bits_per_sample == 24) {
                int32_t val = 0;
                std::memcpy(&val, &audio_data[offset], 3);
                if (val & 0x800000) val |= 0xFF000000;  // Sign extend
                sample += val / 8388608.0f;
            } else if (bits_per_sample == 32) {
                int32_t val;
                std::memcpy(&val, &audio_data[offset], 4);
                sample += val / 2147483648.0f;
            } else if (bits_per_sample == 8) {
                uint8_t val = audio_data[offset];
                sample += (val - 128) / 128.0f;
            }
        }
        
        samples.push_back(sample / num_channels);
    }
    
    return samples;
}

std::vector<float> resample(const std::vector<float>& audio, int src_sr, int dst_sr) {
    if (src_sr == dst_sr || audio.empty()) {
        return audio;
    }
    
    double ratio = static_cast<double>(dst_sr) / src_sr;
    size_t out_len = static_cast<size_t>(audio.size() * ratio);
    std::vector<float> result(out_len);
    
    for (size_t i = 0; i < out_len; i++) {
        double src_pos = i / ratio;
        size_t idx0 = static_cast<size_t>(src_pos);
        size_t idx1 = std::min(idx0 + 1, audio.size() - 1);
        double frac = src_pos - idx0;
        
        result[i] = static_cast<float>(audio[idx0] * (1.0 - frac) + audio[idx1] * frac);
    }
    
    return result;
}

} // namespace io
} // namespace leaxer_qwen
