# leaxer-qwen3-tts Development Guide

## Building

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

## Running Tests

```bash
# Core functionality tests
./test_e2e          # End-to-end pipeline with dummy weights
./test_wav          # WAV file writing

# Component tests (require fixture files)
./test_tokenizer    # BPE tokenizer
./test_snakebeta    # SnakeBeta activation
./test_rmsnorm      # RMSNorm operation
./test_conv1d       # Causal conv1d
```

## Audio Testing Workflow

**IMPORTANT**: After each successful TTS generation, validate the audio output using the analysis scripts.

### Quick Validation

```bash
# After generating audio
./leaxer-qwen -m model.gguf -p "Hello world" -o output.wav

# Analyze for speech characteristics
python3 ../scripts/analyze_audio.py output.wav
```

### Key Metrics to Check

| Metric | Expected Range | Meaning |
|--------|---------------|---------|
| RMS | > 0.02 | Not silent |
| Envelope variation | > 0.5 | Voice-like dynamics |
| Zero-crossing rate | 3000-8000 Hz | Speech frequency range |
| Spectral energy | 200-4000 Hz | Voice frequencies present |

### Analysis Scripts

1. **`analyze_audio.py`** - Primary validation
   ```bash
   python3 scripts/analyze_audio.py output.wav
   ```
   Checks for:
   - Duration and sample count
   - RMS levels (silence detection)
   - Envelope variation (speech dynamics)
   - Zero-crossing rate (frequency content)
   - Spectral characteristics

2. **`test_vocoder_patterns.py`** - Vocoder verification
   ```bash
   python3 scripts/test_vocoder_patterns.py
   ```
   Tests if different input patterns produce different audio outputs.
   Use this to verify the vocoder is actually processing codes, not just outputting a fixed waveform.

3. **`detailed_analysis.py`** - Debugging
   ```bash
   python3 scripts/detailed_analysis.py output.wav
   ```
   Segment-by-segment analysis for debugging specific issues.

4. **`visualize_audio.py`** - Visual inspection
   ```bash
   python3 scripts/visualize_audio.py output.wav
   ```
   Generates waveform and spectrogram plots.

### Interpreting Results

**Good speech output:**
```
RMS: 0.08
Envelope variation: 1.2
ZCR: 4500 Hz
Speech-like: YES
```

**Bad output (noise/silence/tone):**
```
RMS: 0.01          # Too quiet or silent
Envelope variation: 0.1  # Monotone, no dynamics
ZCR: 15000 Hz      # Too high, likely noise
Speech-like: NO
```

### Common Issues

1. **Silent output** (RMS < 0.01)
   - Vocoder not receiving valid codes
   - Codebook lookup returning zeros
   - Check code predictor output

2. **Pure tone** (envelope variation < 0.2)
   - Vocoder stuck in test mode
   - Same codes being fed repeatedly
   - Check LLM token generation

3. **Noise** (ZCR > 10000)
   - Weight loading issue
   - Numerical instability
   - Check for NaN/Inf in tensors

4. **Clicks/pops**
   - Discontinuities at frame boundaries
   - Check overlap-add in vocoder
   - Verify upsample stages

## Model Files

Required for full operation:
- `*.gguf` - Main model weights (talker + code predictor)
- `vocoder.gguf` - Vocoder weights (or combined in main GGUF)
- `vocab.json` - BPE vocabulary
- `merges.txt` - BPE merge rules

## Architecture Overview

```
Text Input
    ↓
Tokenizer (BPE)
    ↓
Talker LLM (28 layers, RoPE attention)
    ↓
Semantic Codes (codebook 0)
    ↓
Code Predictor (5 layers)
    ↓
16 Codebooks
    ↓
Vocoder:
  ├─ RVQ Decode (codebooks → 256-dim)
  ├─ Output Projection (→ 1024-dim)
  ├─ Pre-Transformer (8 layers, 512-dim)
  ├─ Causal Conv (1024 → 1536)
  ├─ Upsample Stages (8x, 5x, 4x, 3x = 480x)
  └─ Final Conv (→ 1 channel)
    ↓
Audio (24kHz mono WAV)
```

## Debugging Tips

1. **Check intermediate outputs**
   - Use `printf` to show token counts at each stage
   - Verify codec token range (should be 0-2047)

2. **Test vocoder in isolation**
   ```bash
   ./test_vocoder_patterns  # If available
   ```

3. **Compare with Python reference**
   ```bash
   python3 scripts/oracle.py "Hello world" --output reference.wav
   python3 scripts/analyze_audio.py reference.wav
   # Compare metrics with C++ output
   ```

4. **Visualize**
   ```bash
   python3 scripts/visualize_audio.py output.wav --save output_analysis.png
   ```
