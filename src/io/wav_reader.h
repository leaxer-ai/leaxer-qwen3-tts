// WAV File Reader
#ifndef LEAXER_QWEN_IO_WAV_READER_H
#define LEAXER_QWEN_IO_WAV_READER_H

#include <string>
#include <vector>
#include <cstdint>

namespace leaxer_qwen {
namespace io {

// Read WAV file to float32 samples
// Returns empty vector on failure
// out_sample_rate is set to the file's sample rate
std::vector<float> read_wav(const std::string& path, int& out_sample_rate);

// Resample audio from src_sr to dst_sr using linear interpolation
std::vector<float> resample(const std::vector<float>& audio, int src_sr, int dst_sr);

} // namespace io
} // namespace leaxer_qwen

#endif // LEAXER_QWEN_IO_WAV_READER_H
