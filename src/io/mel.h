// Mel Spectrogram Extraction
#ifndef LEAXER_QWEN_IO_MEL_H
#define LEAXER_QWEN_IO_MEL_H

#include <vector>
#include <cstddef>

namespace leaxer_qwen {
namespace io {

// Mel spectrogram configuration
struct MelConfig {
    int sample_rate = 24000;
    int n_fft = 1024;
    int hop_size = 256;
    int win_size = 1024;
    int num_mels = 128;
    float fmin = 0.0f;
    float fmax = 12000.0f;
};

// Mel spectrogram extractor
class MelExtractor {
public:
    explicit MelExtractor(const MelConfig& config);
    
    // Extract mel spectrogram from audio
    // Returns [num_mels, num_frames] in row-major order
    std::vector<float> extract(const std::vector<float>& audio);
    
    // Get dimensions of last extraction
    size_t num_frames() const { return num_frames_; }
    size_t num_mels() const { return config_.num_mels; }

private:
    MelConfig config_;
    std::vector<float> window_;
    std::vector<std::vector<float>> mel_filterbank_;
    size_t num_frames_ = 0;
    
    // FFT helpers
    void rfft(const std::vector<float>& input, std::vector<float>& real, std::vector<float>& imag);
    void build_mel_filterbank();
    static float hz_to_mel(float hz);
    static float mel_to_hz(float mel);
};

} // namespace io
} // namespace leaxer_qwen

#endif // LEAXER_QWEN_IO_MEL_H
