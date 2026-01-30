# leaxer-qwen3-tts

Single binary, C++ implementation of [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) running on top of ONNX Runtime.

## Usage

```bash
leaxer-qwen3-tts -m <model_dir> -p "Hello world" -o output.wav

# With language hint
leaxer-qwen3-tts -m onnx_kv_06b -p "你好世界" --lang zh -o chinese.wav

# Sampling controls
leaxer-qwen3-tts -m onnx_kv_06b -p "Hello" --temp 0.7 --top-k 30 --top-p 0.9
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-m, --model` | ONNX model directory | required |
| `-p, --prompt` | Text to synthesize | required |
| `-o, --output` | Output WAV path | `output.wav` |
| `--lang` | Language: `auto`, `en`, `zh`, `ja`, `ko` | `auto` |
| `--temp` | Sampling temperature | `0.8` |
| `--top-k` | Top-k sampling | `50` |
| `--top-p` | Top-p (nucleus) sampling | `0.95` |
| `--max-tokens` | Max generation tokens | `2048` |

## Building

### Requirements
- CMake 3.14+
- C++17 compiler
- ONNX Runtime

### macOS (Homebrew)
```bash
brew install onnxruntime cmake

git clone https://github.com/user/leaxer-qwen3-tts
cd leaxer-qwen3-tts
cmake -B build
cmake --build build -j

./build/leaxer-qwen3-tts --help
```

### Linux
```bash
# Install ONNX Runtime (see https://onnxruntime.ai/)
sudo apt install cmake

cmake -B build
cmake --build build -j
```

## Models

Download ONNX models from [zukky/Qwen3-TTS-ONNX-DLL](https://huggingface.co/zukky/Qwen3-TTS-ONNX-DLL):

```bash
# Clone model repo (or download manually)
git lfs install
git clone https://huggingface.co/zukky/Qwen3-TTS-ONNX-DLL models

# Use the 0.6B model
./build/leaxer-qwen3-tts -m onnx/onnx_kv_06b -p "Hello"
```

### Required files in model directory:
- `text_project.onnx` — text token embeddings
- `codec_embed.onnx` — codec token embeddings  
- `code_predictor_embed.onnx` — sub-codec embeddings
- `talker_prefill.onnx` — transformer prefill
- `talker_decode.onnx` — transformer decode (with KV cache)
- `code_predictor.onnx` — predict codebooks 1-15
- `tokenizer12hz_decode.onnx` — vocoder (codes → audio)

Also needs tokenizer files in `../models/Qwen3-TTS-12Hz-0.6B-Base/`:
- `vocab.json`
- `merges.txt`

## Architecture

```
Text → BPE Tokenizer → Talker (prefill/decode) → Code Predictor → Vocoder → WAV
                              ↓                        ↓
                           KV Cache              Codebooks 1-15
```

The model generates 16 audio codebooks per frame at 12Hz, then the vocoder upsamples to 24kHz audio.

## Voice Clone

Clone any voice from a 3-second reference audio:

```bash
leaxer-qwen3-tts -m onnx/onnx_kv_06b -p "Hello world" --ref voice_sample.wav -o output.wav
```

The reference audio should be:
- WAV format (8/16/24/32-bit PCM or float)
- At least 3 seconds of clear speech
- Will be resampled to 24kHz internally

## Roadmap

| Feature | Requires | Status |
|---------|----------|--------|
| Voice clone (`--ref`) | 0.6B-Base | Done |
| GPU acceleration | ONNX Runtime + CoreML/CUDA | Done |
| Preset speakers (`--speaker`) | 0.6B-CustomVoice | Waiting for ONNX export |
| Voice instructions (`--instruct`) | 1.7B-VoiceDesign | Waiting for ONNX export |
| Single static binary | Static ONNX Runtime | Planned |

Currently only 0.6B-Base ONNX models are available from [zukky's exports](https://huggingface.co/zukky/Qwen3-TTS-ONNX-DLL).

## Credits

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba
- [zukky/Qwen3-TTS-ONNX-DLL](https://huggingface.co/zukky/Qwen3-TTS-ONNX-DLL) for ONNX exports, big thanks to Mr. Daishi Suzuki (zukky)!

## License

Apache 2.0 — see [LICENSE](LICENSE)
