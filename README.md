# leaxer-qwen3-tts

Pure C++ implementation of Qwen3-TTS text-to-speech.

## Goal

Single binary that converts text to speech without Python runtime dependency.

```bash
leaxer-qwen3-tts -m models/ -p "Hello world" -o output.wav
```

## Status

🚧 **Work in Progress** — Refactoring to ONNX Runtime

## Building

```bash
# Dependencies
# - ONNX Runtime (onnxruntime-cpp)
# - CMake 3.16+

cmake -B build
cmake --build build

# Run
./build/leaxer-qwen3-tts -m models/ -p "Hello world" -o output.wav
```

## Architecture

```
Text → Tokenizer → Talker ONNX → Code Predictor ONNX → Vocoder ONNX → 24kHz WAV
                   (prefill/decode)   (codebooks 1-15)    (codes→audio)
```

### ONNX Models

| Model | Purpose |
|-------|---------|
| `talker_prefill.onnx` | Process input prompt |
| `talker_decode.onnx` | Generate tokens with KV-cache |
| `code_predictor.onnx` | Predict sub-codebooks 1-15 |
| `tokenizer12hz_decode.onnx` | Decode codes to audio |
| `speaker_encoder.onnx` | Extract speaker embedding |

## Credits

**ONNX models from:**
- [zukky/Qwen3-TTS-ONNX-DLL](https://huggingface.co/zukky/Qwen3-TTS-ONNX-DLL) — Apache-2.0
- Huge thanks to zukky for the ONNX export work!

**Original model:**
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba — Apache-2.0

## References

- [ONNX Runtime](https://onnxruntime.ai/) — Inference engine
- [Qwen3-TTS Paper](https://arxiv.org/abs/2505.XXXXX) — Model architecture

## License

Apache 2.0

---

*This project uses ONNX models derived from Qwen3-TTS. See [LICENSE](LICENSE) for details.*
