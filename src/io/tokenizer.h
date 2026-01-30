// Qwen2 BPE Tokenizer Header
#ifndef LEAXER_QWEN_IO_TOKENIZER_H
#define LEAXER_QWEN_IO_TOKENIZER_H

#include <string>
#include <vector>
#include <cstdint>

namespace leaxer_qwen {
namespace io {

// Load vocab.json (token -> id mapping)
bool load_vocab(const std::string& vocab_path);

// Load merges.txt (BPE merge rules)
bool load_merges(const std::string& merges_path);

// Check if tokenizer is initialized
bool is_tokenizer_ready();

// Tokenize text to token IDs
std::vector<int32_t> tokenize(const std::string& text);

// Convert token ID to string
std::string token_to_string(int32_t id);

// Convert string to token ID (exact match)
int32_t string_to_token(const std::string& token);

} // namespace io
} // namespace leaxer_qwen

#endif // LEAXER_QWEN_IO_TOKENIZER_H
