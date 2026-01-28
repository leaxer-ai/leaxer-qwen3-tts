#!/usr/bin/env python3
"""
Test vocoder with synthetic code patterns.
If the vocoder works, different patterns should produce different audio.
"""

import subprocess
import wave
import numpy as np
import os

def analyze_segment(samples, start_sec, duration_sec, sample_rate):
    """Analyze a segment of audio"""
    start = int(start_sec * sample_rate)
    end = int((start_sec + duration_sec) * sample_rate)
    segment = samples[start:end]

    rms = np.sqrt(np.mean(segment ** 2))
    zcr = np.sum(np.abs(np.diff(np.sign(segment))) > 0) / len(segment) * sample_rate

    return {'rms': rms, 'zcr': zcr}

def load_wav(filepath):
    with wave.open(filepath, 'rb') as wf:
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)
        if sample_width == 2:
            samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            samples = np.frombuffer(raw_data, dtype=np.float32)
        return samples, sample_rate

def compare_audio_files(files):
    """Compare multiple audio files to see if they're different"""
    results = []

    for f in files:
        if not os.path.exists(f):
            print(f"  File not found: {f}")
            continue

        samples, sr = load_wav(f)

        # Analyze first second
        stats = analyze_segment(samples, 0, 1.0, sr)

        # FFT of first second
        fft = np.abs(np.fft.rfft(samples[:sr]))
        dominant_freq = np.argmax(fft) * sr / len(samples[:sr]) / 2

        results.append({
            'file': f,
            'rms': stats['rms'],
            'zcr': stats['zcr'],
            'dominant_freq': dominant_freq,
            'samples_hash': hash(samples[:1000].tobytes())  # Quick identity check
        })

        print(f"\n{os.path.basename(f)}:")
        print(f"  RMS: {stats['rms']:.4f}")
        print(f"  ZCR: {stats['zcr']:.0f}/sec")
        print(f"  Dominant freq: {dominant_freq:.1f} Hz")

    return results

def main():
    print("=" * 60)
    print("Vocoder Pattern Test")
    print("=" * 60)

    # Generate with different prompts to see if output changes
    prompts = [
        ("Hello", "test_hello.wav"),
        ("World", "test_world.wav"),
        ("The", "test_the.wav"),
        ("A B C D E F G", "test_abc.wav"),
    ]

    exe = "C:/Users/afterlab/leaxer-qwen/build/leaxer-qwen.exe"
    model = "C:/Users/afterlab/leaxer-qwen/models/qwen3_tts_customvoice.gguf"
    out_dir = "C:/Users/afterlab/leaxer-qwen/output"

    files = []
    for prompt, filename in prompts:
        out_path = os.path.join(out_dir, filename)
        files.append(out_path)

        print(f"\nGenerating: '{prompt}' -> {filename}")
        cmd = [exe, "-m", model, "-p", prompt, "-o", out_path, "--speaker", "aiden"]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                print(f"  ERROR: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            print("  TIMEOUT")
        except Exception as e:
            print(f"  Exception: {e}")

    print("\n" + "=" * 60)
    print("Comparing outputs:")
    print("=" * 60)

    results = compare_audio_files(files)

    # Check if outputs are actually different
    if len(results) >= 2:
        hashes = [r['samples_hash'] for r in results]
        rms_values = [r['rms'] for r in results]

        print("\n" + "-" * 60)
        if len(set(hashes)) == 1:
            print("WARNING: All files have IDENTICAL first 1000 samples!")
            print("This suggests the vocoder output is independent of input!")
        elif max(rms_values) - min(rms_values) < 0.01:
            print("WARNING: All files have very similar RMS (diff < 0.01)")
            print("Different prompts should produce different amplitudes.")
        else:
            print("Good: Files appear to have different content.")

    print("-" * 60)

if __name__ == '__main__':
    main()
