// Qwen2 BPE Tokenizer
// Tokenizes input text to token IDs

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cctype>
#include <cstring>

namespace leaxer_qwen {
namespace io {

// Hash function for string pairs (for merge_rank_)
struct PairHash {
    std::size_t operator()(const std::pair<std::string, std::string>& p) const {
        return std::hash<std::string>{}(p.first) ^ (std::hash<std::string>{}(p.second) << 1);
    }
};

// Helper to parse hex digit
static int hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
    if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
    return -1;
}

// Minimal JSON parser for vocab.json
// Parses flat dictionary: {"token": id, ...}
static bool parse_vocab_json(const std::string& path,
                             std::unordered_map<std::string, int32_t>& token_to_id,
                             std::unordered_map<int32_t, std::string>& id_to_token) {
    FILE* file = fopen(path.c_str(), "rb");
    if (!file) {
        fprintf(stderr, "Failed to open vocab file: %s\n", path.c_str());
        return false;
    }

    // Get file size
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    if (file_size <= 0 || file_size > 100 * 1024 * 1024) {
        fprintf(stderr, "Invalid file size: %ld\n", file_size);
        fclose(file);
        return false;
    }

    // Read file into buffer
    char* buffer = (char*)malloc(file_size + 1);
    if (!buffer) {
        fprintf(stderr, "Failed to allocate memory\n");
        fclose(file);
        return false;
    }

    size_t read_size = fread(buffer, 1, file_size, file);
    fclose(file);

    if (read_size != (size_t)file_size) {
        fprintf(stderr, "Failed to read file\n");
        free(buffer);
        return false;
    }

    buffer[file_size] = '\0';
    const char* content = buffer;
    size_t pos = 0;
    size_t len = file_size;

    // Skip opening '{'
    while (pos < len && std::isspace(static_cast<unsigned char>(content[pos]))) pos++;
    if (pos >= len || content[pos] != '{') {
        fprintf(stderr, "Expected '{' at start of JSON\n");
        free(buffer);
        return false;
    }
    pos++;

    int count = 0;
    while (pos < len) {
        // Skip whitespace
        while (pos < len && std::isspace(static_cast<unsigned char>(content[pos]))) pos++;
        if (pos >= len) break;

        // Check for closing '}'
        if (content[pos] == '}') break;

        // Skip comma
        if (content[pos] == ',') {
            pos++;
            continue;
        }

        // Parse key (token string)
        if (content[pos] != '"') {
            fprintf(stderr, "Expected '\"' at position %zu\n", pos);
            free(buffer);
            return false;
        }
        pos++; // Skip opening quote

        std::string token;
        while (pos < len && content[pos] != '"') {
            if (content[pos] == '\\') {
                pos++;
                if (pos >= len) {
                    fprintf(stderr, "Unexpected end of file in escape sequence\n");
                    free(buffer);
                    return false;
                }

                // Handle escape sequences
                switch (content[pos]) {
                    case 'n': token += '\n'; break;
                    case 't': token += '\t'; break;
                    case 'r': token += '\r'; break;
                    case '\\': token += '\\'; break;
                    case '"': token += '"'; break;
                    case 'u': {
                        // Unicode escape: \uXXXX
                        if (pos + 4 >= len) {
                            fprintf(stderr, "Invalid unicode escape\n");
                            free(buffer);
                            return false;
                        }
                        pos++;

                        // Parse hex digits manually
                        int codepoint = 0;
                        for (int i = 0; i < 4; i++) {
                            int digit = hex_digit(content[pos + i]);
                            if (digit < 0) {
                                fprintf(stderr, "Invalid hex digit in unicode escape\n");
                                free(buffer);
                                return false;
                            }
                            codepoint = (codepoint << 4) | digit;
                        }

                        // Simple UTF-8 encoding (BMP only)
                        if (codepoint < 0x80) {
                            token += static_cast<char>(codepoint);
                        } else if (codepoint < 0x800) {
                            token += static_cast<char>(0xC0 | (codepoint >> 6));
                            token += static_cast<char>(0x80 | (codepoint & 0x3F));
                        } else {
                            token += static_cast<char>(0xE0 | (codepoint >> 12));
                            token += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
                            token += static_cast<char>(0x80 | (codepoint & 0x3F));
                        }
                        pos += 3; // Will be incremented by 1 at end of loop
                        break;
                    }
                    default:
                        token += content[pos];
                        break;
                }
            } else {
                token += content[pos];
            }
            pos++;
        }
        if (pos >= len) {
            fprintf(stderr, "Unexpected end of file in token string\n");
            free(buffer);
            return false;
        }
        pos++; // Skip closing quote

        // Skip whitespace and colon
        while (pos < len && std::isspace(static_cast<unsigned char>(content[pos]))) pos++;
        if (pos >= len || content[pos] != ':') {
            fprintf(stderr, "Expected ':' after token\n");
            free(buffer);
            return false;
        }
        pos++;
        while (pos < len && std::isspace(static_cast<unsigned char>(content[pos]))) pos++;

        // Parse value (token ID)
        if (pos >= len || !std::isdigit(static_cast<unsigned char>(content[pos]))) {
            fprintf(stderr, "Expected digit for token ID\n");
            free(buffer);
            return false;
        }

        int32_t id = 0;
        while (pos < len && std::isdigit(static_cast<unsigned char>(content[pos]))) {
            id = id * 10 + (content[pos] - '0');
            pos++;
        }

        // Store in maps
        token_to_id[token] = id;
        id_to_token[id] = token;
        count++;

        // Progress indicator for large files
        if (count % 10000 == 0) {
            fprintf(stderr, "Loaded %d tokens...\n", count);
        }
    }

    free(buffer);
    fprintf(stderr, "Successfully loaded %d tokens\n", count);
    return !token_to_id.empty();
}

class BPETokenizer {
public:
    BPETokenizer() : vocab_loaded_(false), merges_loaded_(false) {}

    bool load_vocab(const std::string& vocab_path) {
        token_to_id_.clear();
        id_to_token_.clear();

        if (!parse_vocab_json(vocab_path, token_to_id_, id_to_token_)) {
            return false;
        }

        vocab_loaded_ = true;
        return true;
    }

    bool load_merges(const std::string& merges_path) {
        merges_.clear();
        merge_rank_.clear();

        FILE* file = fopen(merges_path.c_str(), "r");
        if (!file) {
            fprintf(stderr, "Failed to open merges file: %s\n", merges_path.c_str());
            return false;
        }

        char line[1024];
        int rank = 0;
        int count = 0;

        while (fgets(line, sizeof(line), file)) {
            // Remove newline
            size_t len = strlen(line);
            while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) {
                line[--len] = '\0';
            }

            if (len == 0) {
                continue; // Skip empty lines
            }

            // Parse space-separated pair
            char* space = strchr(line, ' ');
            if (!space) {
                fprintf(stderr, "Invalid merge line (no space): %s\n", line);
                continue;
            }

            *space = '\0'; // Split at space
            std::string token1(line);
            std::string token2(space + 1);

            // Store merge pair with its rank (order)
            merges_.push_back({token1, token2});
            merge_rank_[{token1, token2}] = rank;
            rank++;
            count++;

            if (count % 10000 == 0) {
                fprintf(stderr, "Loaded %d merge rules...\n", count);
            }
        }

        fclose(file);
        fprintf(stderr, "Successfully loaded %d merge rules\n", count);
        merges_loaded_ = true;
        return !merges_.empty();
    }

    std::vector<int32_t> tokenize(const std::string& text) {
        std::vector<int32_t> tokens;

        if (text.empty()) {
            return tokens;
        }

        if (!vocab_loaded_) {
            // Fallback: byte-level tokenization
            for (unsigned char c : text) {
                tokens.push_back(static_cast<int32_t>(c));
            }
            return tokens;
        }

        // BPE encoding algorithm:
        // 1. Split text into UTF-8 bytes and convert to token strings
        std::vector<std::string> word;
        for (unsigned char c : text) {
            // Create byte token string (e.g., "a" for 'a', or base vocabulary token)
            std::string byte_token(1, static_cast<char>(c));

            // Check if byte exists as token in vocab
            auto it = token_to_id_.find(byte_token);
            if (it != token_to_id_.end()) {
                word.push_back(byte_token);
            } else {
                // If single byte not in vocab, use as-is
                word.push_back(byte_token);
            }
        }

        // 2. Iteratively apply BPE merges until no more can be applied
        if (merges_loaded_ && !word.empty()) {
            while (true) {
                // Find the pair with the lowest merge rank (highest priority)
                int best_rank = INT_MAX;
                int best_pos = -1;

                for (size_t i = 0; i + 1 < word.size(); i++) {
                    auto pair = std::make_pair(word[i], word[i + 1]);
                    auto it = merge_rank_.find(pair);
                    if (it != merge_rank_.end()) {
                        if (it->second < best_rank) {
                            best_rank = it->second;
                            best_pos = static_cast<int>(i);
                        }
                    }
                }

                // If no valid merge found, we're done
                if (best_pos == -1) {
                    break;
                }

                // Merge the best pair
                std::string merged = word[best_pos] + word[best_pos + 1];
                word[best_pos] = merged;
                word.erase(word.begin() + best_pos + 1);
            }
        }

        // 3. Convert final tokens to IDs
        for (const auto& token : word) {
            auto it = token_to_id_.find(token);
            if (it != token_to_id_.end()) {
                tokens.push_back(it->second);
            } else {
                // Token not in vocab - try byte-level fallback
                if (token.length() == 1) {
                    tokens.push_back(static_cast<int32_t>(static_cast<unsigned char>(token[0])));
                } else {
                    // Multi-byte token not in vocab - split into bytes
                    for (unsigned char c : token) {
                        tokens.push_back(static_cast<int32_t>(c));
                    }
                }
            }
        }

        return tokens;
    }

    std::string token_to_string(int32_t id) const {
        auto it = id_to_token_.find(id);
        if (it != id_to_token_.end()) {
            return it->second;
        }
        return "";
    }

    int32_t string_to_token(const std::string& token) const {
        auto it = token_to_id_.find(token);
        if (it != token_to_id_.end()) {
            return it->second;
        }
        return -1; // Not found
    }

    bool is_loaded() const {
        return vocab_loaded_;
    }

    bool merges_loaded() const {
        return merges_loaded_;
    }

    size_t vocab_size() const {
        return token_to_id_.size();
    }

    size_t merges_size() const {
        return merges_.size();
    }

private:
    bool vocab_loaded_;
    bool merges_loaded_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_map<int32_t, std::string> id_to_token_;

    // BPE merge rules
    std::vector<std::pair<std::string, std::string>> merges_;
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> merge_rank_;
};

// Global tokenizer instance - use function-local static for lazy initialization
static BPETokenizer& get_tokenizer() {
    static BPETokenizer tokenizer;
    return tokenizer;
}

// Public API functions
bool load_vocab(const std::string& vocab_path) {
    return get_tokenizer().load_vocab(vocab_path);
}

bool load_merges(const std::string& merges_path) {
    return get_tokenizer().load_merges(merges_path);
}

std::vector<int32_t> tokenize(const std::string& text) {
    return get_tokenizer().tokenize(text);
}

std::string token_to_string(int32_t id) {
    return get_tokenizer().token_to_string(id);
}

int32_t string_to_token(const std::string& token) {
    return get_tokenizer().string_to_token(token);
}

} // namespace io
} // namespace leaxer_qwen
