# 🚧 Work in Progress 🚧

Pure C++ implementation of Qwen3-TTS using ggml tensor library.

## Goal

Single binary `leaxer-qwen` that converts text to speech without Python runtime dependency.

```bash
leaxer-qwen -m qwen3-tts-1.7b.gguf -p "Hello world" -o output.wav
```

## Building

```bash
# Initialize ggml submodule
git submodule add https://github.com/ggerganov/ggml extern/ggml
git submodule update --init --recursive

# Build
cmake -B build
cmake --build build

# Run tests
ctest --test-dir build
```

## Architecture

```
Text → Tokenizer → Qwen3 LLM → Code Predictor → Split RVQ → Vocoder → 24kHz WAV
```

### Components

- **ggml_ops/**: Custom tensor operations (SnakeBeta, RoPE, etc.)
- **vocoder/**: Audio decoder (RVQ + Upsample stages)
- **model/**: Transformer blocks, attention, FFN
- **io/**: GGUF loading, tokenization, WAV output

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for:
- Audio testing workflow with validation scripts
- Debugging tips and common issues
- Architecture details

## Reference

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) - Original Python implementation
- [ggml](https://github.com/ggerganov/ggml) - Tensor library
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Reference for ggml patterns

## License

MIT
