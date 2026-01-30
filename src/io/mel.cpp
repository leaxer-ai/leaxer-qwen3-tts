// Mel Spectrogram Extraction Implementation
#include "mel.h"
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace leaxer_qwen {
namespace io {

MelExtractor::MelExtractor(const MelConfig& config) : config_(config) {
    // Build Hann window
    window_.resize(config_.win_size);
    for (int i = 0; i < config_.win_size; i++) {
        window_[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (config_.win_size - 1)));
    }
    
    // Build mel filterbank
    build_mel_filterbank();
}

float MelExtractor::hz_to_mel(float hz) {
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

float MelExtractor::mel_to_hz(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

void MelExtractor::build_mel_filterbank() {
    int n_fft_bins = config_.n_fft / 2 + 1;
    
    // Compute mel frequencies
    float mel_min = hz_to_mel(config_.fmin);
    float mel_max = hz_to_mel(config_.fmax);
    
    std::vector<float> mel_points(config_.num_mels + 2);
    for (int i = 0; i < config_.num_mels + 2; i++) {
        mel_points[i] = mel_min + (mel_max - mel_min) * i / (config_.num_mels + 1);
    }
    
    // Convert to Hz
    std::vector<float> hz_points(config_.num_mels + 2);
    for (int i = 0; i < config_.num_mels + 2; i++) {
        hz_points[i] = mel_to_hz(mel_points[i]);
    }
    
    // Convert to FFT bin indices
    std::vector<int> bin_points(config_.num_mels + 2);
    for (int i = 0; i < config_.num_mels + 2; i++) {
        bin_points[i] = static_cast<int>(std::floor((config_.n_fft + 1) * hz_points[i] / config_.sample_rate));
        bin_points[i] = std::min(bin_points[i], n_fft_bins - 1);
    }
    
    // Build filterbank matrix
    mel_filterbank_.resize(config_.num_mels);
    for (int m = 0; m < config_.num_mels; m++) {
        mel_filterbank_[m].resize(n_fft_bins, 0.0f);
        
        int f_left = bin_points[m];
        int f_center = bin_points[m + 1];
        int f_right = bin_points[m + 2];
        
        // Rising slope
        for (int k = f_left; k < f_center; k++) {
            if (f_center > f_left) {
                mel_filterbank_[m][k] = static_cast<float>(k - f_left) / (f_center - f_left);
            }
        }
        
        // Falling slope
        for (int k = f_center; k < f_right; k++) {
            if (f_right > f_center) {
                mel_filterbank_[m][k] = static_cast<float>(f_right - k) / (f_right - f_center);
            }
        }
    }
}

// Cooley-Tukey FFT (radix-2)
static void fft_recursive(std::vector<float>& real, std::vector<float>& imag, int n, int stride, int offset) {
    if (n <= 1) return;
    
    int half = n / 2;
    
    // Recursive calls for even and odd
    fft_recursive(real, imag, half, stride * 2, offset);
    fft_recursive(real, imag, half, stride * 2, offset + stride);
    
    // Combine
    for (int k = 0; k < half; k++) {
        float angle = -2.0f * M_PI * k / n;
        float wr = std::cos(angle);
        float wi = std::sin(angle);
        
        int even_idx = offset + k * stride * 2;
        int odd_idx = offset + k * stride * 2 + stride;
        
        float tr = wr * real[odd_idx] - wi * imag[odd_idx];
        float ti = wr * imag[odd_idx] + wi * real[odd_idx];
        
        float er = real[even_idx];
        float ei = imag[even_idx];
        
        real[even_idx] = er + tr;
        imag[even_idx] = ei + ti;
        real[odd_idx] = er - tr;
        imag[odd_idx] = ei - ti;
    }
}

// Bit-reversal permutation
static void bit_reverse_permute(std::vector<float>& real, std::vector<float>& imag, int n) {
    int bits = 0;
    while ((1 << bits) < n) bits++;
    
    for (int i = 0; i < n; i++) {
        int j = 0;
        for (int b = 0; b < bits; b++) {
            if (i & (1 << b)) j |= (1 << (bits - 1 - b));
        }
        if (j > i) {
            std::swap(real[i], real[j]);
            std::swap(imag[i], imag[j]);
        }
    }
}

// Iterative FFT (more efficient)
static void fft_iterative(std::vector<float>& real, std::vector<float>& imag, int n) {
    bit_reverse_permute(real, imag, n);
    
    for (int size = 2; size <= n; size *= 2) {
        int half = size / 2;
        float angle_step = -2.0f * M_PI / size;
        
        for (int i = 0; i < n; i += size) {
            for (int k = 0; k < half; k++) {
                float angle = angle_step * k;
                float wr = std::cos(angle);
                float wi = std::sin(angle);
                
                int even_idx = i + k;
                int odd_idx = i + k + half;
                
                float tr = wr * real[odd_idx] - wi * imag[odd_idx];
                float ti = wr * imag[odd_idx] + wi * real[odd_idx];
                
                real[odd_idx] = real[even_idx] - tr;
                imag[odd_idx] = imag[even_idx] - ti;
                real[even_idx] = real[even_idx] + tr;
                imag[even_idx] = imag[even_idx] + ti;
            }
        }
    }
}

void MelExtractor::rfft(const std::vector<float>& input, std::vector<float>& real, std::vector<float>& imag) {
    int n = static_cast<int>(input.size());
    
    // Pad to power of 2
    int n_padded = 1;
    while (n_padded < n) n_padded *= 2;
    
    real.assign(n_padded, 0.0f);
    imag.assign(n_padded, 0.0f);
    
    for (int i = 0; i < n; i++) {
        real[i] = input[i];
    }
    
    fft_iterative(real, imag, n_padded);
    
    // Keep only positive frequencies (n/2 + 1 bins)
    int n_bins = n_padded / 2 + 1;
    real.resize(n_bins);
    imag.resize(n_bins);
}

std::vector<float> MelExtractor::extract(const std::vector<float>& audio) {
    if (audio.empty()) return {};
    
    int audio_len = static_cast<int>(audio.size());
    num_frames_ = (audio_len - config_.win_size) / config_.hop_size + 1;
    if (num_frames_ <= 0) {
        num_frames_ = 1;
    }
    
    int n_fft_bins = config_.n_fft / 2 + 1;
    std::vector<float> mel_spec(config_.num_mels * num_frames_, 0.0f);
    
    std::vector<float> frame(config_.win_size);
    std::vector<float> padded(config_.n_fft);
    std::vector<float> real, imag;
    
    for (size_t t = 0; t < num_frames_; t++) {
        int start = static_cast<int>(t) * config_.hop_size;
        
        // Extract and window frame
        for (int i = 0; i < config_.win_size; i++) {
            int idx = start + i;
            if (idx >= 0 && idx < audio_len) {
                frame[i] = audio[idx] * window_[i];
            } else {
                frame[i] = 0.0f;
            }
        }
        
        // Pad to n_fft
        std::fill(padded.begin(), padded.end(), 0.0f);
        for (int i = 0; i < config_.win_size && i < config_.n_fft; i++) {
            padded[i] = frame[i];
        }
        
        // Compute FFT
        rfft(padded, real, imag);
        
        // Compute power spectrum and apply mel filterbank
        for (int m = 0; m < config_.num_mels; m++) {
            float mel_energy = 0.0f;
            for (int k = 0; k < n_fft_bins && k < static_cast<int>(real.size()); k++) {
                float power = real[k] * real[k] + imag[k] * imag[k];
                mel_energy += mel_filterbank_[m][k] * power;
            }
            
            // Log mel energy (add small epsilon to avoid log(0))
            mel_spec[m * num_frames_ + t] = std::log(mel_energy + 1e-10f);
        }
    }
    
    return mel_spec;
}

} // namespace io
} // namespace leaxer_qwen
