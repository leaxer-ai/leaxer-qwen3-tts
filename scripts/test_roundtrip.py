#!/usr/bin/env python3
"""
Test vocoder by encoding real audio and decoding it
This bypasses the talker/code predictor and tests only the vocoder
"""

import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from safetensors.torch import load_file as safe_load


def load_encoder(model_path):
    """Load the encoder (speech tokenizer) weights"""
    st_path = Path(model_path) / "speech_tokenizer" / "model.safetensors"
    sd = safe_load(st_path)
    return sd


def simple_sine_wave(freq=440, duration=1.0, sr=24000):
    """Generate a simple sine wave for testing"""
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    return np.sin(2 * np.pi * freq * t)


def encode_audio_simple(audio, sd):
    """
    Simple encoding using just the encoder codebooks
    This is a placeholder - the real encoder is complex
    """
    # The encoder uses VQ-VAE to encode audio to discrete codes
    # We can't easily replicate this without the full model
    # Instead, let's generate random but structured codes
    
    # Calculate expected code length (12 tokens per second at 12Hz)
    sr = 24000
    duration = len(audio) / sr
    code_len = int(duration * 12)  # 12 Hz tokenizer
    
    print(f"Audio duration: {duration:.2f}s, code length: {code_len}")
    
    # Generate structured codes (not random)
    # Real speech has structure - repeating patterns, limited vocabulary
    codes = np.zeros((code_len, 16), dtype=np.int32)
    
    # Use a simple pattern based on audio energy
    chunk_size = len(audio) // code_len
    for t in range(code_len):
        start = t * chunk_size
        end = min(start + chunk_size, len(audio))
        energy = np.sqrt(np.mean(audio[start:end] ** 2))
        
        # Map energy to code (0-2047)
        base_code = int(energy * 2000) % 2048
        
        for q in range(16):
            # Add some variation per codebook
            codes[t, q] = (base_code + q * 100) % 2048
    
    return codes


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input", help="Input audio file (optional)")
    parser.add_argument("--output", default="roundtrip.wav")
    args = parser.parse_args()
    
    # Load encoder weights (for reference)
    sd = load_encoder(args.model_path)
    
    # Load or generate test audio
    if args.input:
        audio, sr = sf.read(args.input)
        if sr != 24000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
        print(f"Loaded {args.input}")
    else:
        # Generate test signal
        print("Generating test sine wave...")
        audio = simple_sine_wave(440, 1.0, 24000)
    
    # Encode audio to codes (simplified)
    codes = encode_audio_simple(audio, sd)
    print(f"Codes shape: {codes.shape}")
    
    # Save codes for testing with C++
    codes.astype(np.int32).tofile("test_codes.bin")
    print("Saved test_codes.bin")
    
    # Now let's see what range of codes the actual vocoder codebooks cover
    print("\n=== Codebook Analysis ===")
    for i in range(16):
        key = f"decoder.quantizer.rvq_{'first' if i == 0 else 'rest'}.vq.layers.{0 if i == 0 else i-1}._codebook.embedding_sum"
        if key in sd:
            emb = sd[key]
            print(f"CB{i:2d}: shape={list(emb.shape)}, range=[{emb.float().min():.4f}, {emb.float().max():.4f}]")


if __name__ == "__main__":
    main()
