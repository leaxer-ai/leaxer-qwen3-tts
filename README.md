# leaxer-qwen3-tts

C++ implementation of Qwen3-TTS text-to-speech using ONNX runtime.

## Goal

Single binary that converts text to speech without Python runtime dependency.

```bash
leaxer-qwen3-tts -m models/ -p "Hello world" -o output.wav
```

## Status

🚧 **Work in Progress** — Refactoring to ONNX Runtime

## Dependencies
- ONNX Runtime (onnxruntime-cpp)
- CMake 3.16+

## Building

```bash
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
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba — Apache-2.0 
- [zukky/Qwen3-TTS-ONNX-DLL](https://huggingface.co/zukky/Qwen3-TTS-ONNX-DLL) — Apache-2.0

## License

MIT

---

*This project uses ONNX models derived from Qwen3-TTS. See [LICENSE](LICENSE) for details.*
