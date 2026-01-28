#!/usr/bin/env python3
"""Analyze audio waveform to detect speech-like characteristics."""

import sys
import numpy as np
import wave

def analyze_audio(wav_path):
    """Analyze WAV file for speech characteristics."""
    with wave.open(wav_path, 'rb') as wf:
        n_frames = wf.getnframes()
        sample_rate = wf.getframerate()
        audio_data = wf.readframes(n_frames)
        samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

    duration = len(samples) / sample_rate

    print(f"=== Audio Analysis ===")
    print(f"Duration: {duration:.2f}s ({len(samples)} samples @ {sample_rate}Hz)")
    print(f"Range: [{samples.min():.4f}, {samples.max():.4f}]")
    print(f"Mean: {samples.mean():.6f}")
    print(f"RMS: {np.sqrt(np.mean(samples**2)):.4f}")

    # Compute envelope (amplitude over time)
    window = int(sample_rate * 0.05)  # 50ms windows
    n_windows = len(samples) // window
    envelope = []
    for i in range(n_windows):
        chunk = samples[i*window:(i+1)*window]
        envelope.append(np.sqrt(np.mean(chunk**2)))
    envelope = np.array(envelope)

    env_mean = envelope.mean()
    env_std = envelope.std()
    env_variation = env_std / env_mean if env_mean > 0.001 else 0

    print(f"\nEnvelope (50ms windows, n={n_windows}):")
    print(f"  Mean: {env_mean:.4f}")
    print(f"  Std: {env_std:.4f}")
    print(f"  Variation (std/mean): {env_variation:.2f}")

    # Detect silent segments (RMS < 0.02)
    silent_windows = np.sum(envelope < 0.02)
    silent_pct = 100.0 * silent_windows / len(envelope) if len(envelope) > 0 else 0
    print(f"  Silent windows (<0.02 RMS): {silent_windows} ({silent_pct:.1f}%)")

    # Zero-crossing rate
    signs = np.sign(samples)
    zero_crossings = np.sum(np.abs(np.diff(signs)) == 2)
    zcr = zero_crossings / len(samples)
    zcr_hz = zcr * sample_rate / 2
    print(f"\nZero-crossing rate: {zcr:.4f} (~{zcr_hz:.0f} Hz equivalent)")

    # Print diagnostic
    print("\n=== Diagnosis ===")
    issues = []

    if env_variation < 0.3:
        issues.append(f"Constant envelope (variation={env_variation:.2f}, should be >0.3)")

    if silent_pct < 10:
        issues.append(f"No silent segments ({silent_pct:.1f}%, speech usually has >10%)")

    if zcr_hz < 500 or zcr_hz > 8000:
        issues.append(f"Unusual zero-crossing rate ({zcr_hz:.0f} Hz, speech is usually 500-8000 Hz)")

    if env_mean < 0.01:
        issues.append(f"Very low amplitude (RMS={env_mean:.4f})")

    if not issues:
        print("Audio appears to have speech-like characteristics!")
    else:
        print("PROBLEMS DETECTED:")
        for issue in issues:
            print(f"  - {issue}")

    # Plot if matplotlib available
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(12, 8))

        # Waveform
        time = np.arange(len(samples)) / sample_rate
        axes[0].plot(time, samples, linewidth=0.5)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title('Waveform')
        axes[0].set_ylim(-1.1, 1.1)

        # Envelope
        env_time = np.arange(len(envelope)) * 0.05
        axes[1].plot(env_time, envelope)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('RMS')
        axes[1].set_title(f'Envelope (variation={env_variation:.2f})')

        # First 100ms zoom
        zoom_samples = int(sample_rate * 0.1)
        zoom_time = np.arange(zoom_samples) / sample_rate * 1000
        axes[2].plot(zoom_time, samples[:zoom_samples], linewidth=0.5)
        axes[2].set_xlabel('Time (ms)')
        axes[2].set_ylabel('Amplitude')
        axes[2].set_title('First 100ms (zoomed)')

        plt.tight_layout()
        output_png = wav_path.replace('.wav', '_analysis.png')
        plt.savefig(output_png, dpi=100)
        print(f"\nPlot saved to: {output_png}")
    except ImportError:
        print("\n(matplotlib not available for visualization)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_audio.py <wav_file>")
        sys.exit(1)
    analyze_audio(sys.argv[1])
