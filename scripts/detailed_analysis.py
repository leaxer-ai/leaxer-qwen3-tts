#!/usr/bin/env python3
"""
Detailed audio analysis - dump segment-by-segment envelope and stats
"""

import sys
import wave
import numpy as np

def load_wav(filepath):
    with wave.open(filepath, 'rb') as wf:
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)
        if sample_width == 2:
            samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            samples = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")
        return samples, sample_rate

def analyze_segments(samples, sample_rate, segment_ms=100):
    """Analyze audio in segments"""
    segment_samples = int(segment_ms * sample_rate / 1000)
    n_segments = len(samples) // segment_samples

    print(f"\nSegment analysis ({segment_ms}ms segments, {n_segments} total):")
    print("-" * 80)
    print(f"{'Seg':>4} {'Time':>8} {'RMS':>8} {'Peak':>8} {'ZCR':>6} {'Pattern':>40}")
    print("-" * 80)

    rms_values = []
    for i in range(min(n_segments, 100)):  # First 100 segments (10 seconds)
        start = i * segment_samples
        end = start + segment_samples
        segment = samples[start:end]

        rms = np.sqrt(np.mean(segment ** 2))
        peak = np.max(np.abs(segment))
        zcr = np.sum(np.abs(np.diff(np.sign(segment))) > 0) / len(segment) * sample_rate

        rms_values.append(rms)

        # Visual pattern (scaled RMS bar)
        bar_len = int(rms * 200)
        bar = '#' * min(bar_len, 40)

        time_sec = i * segment_ms / 1000
        print(f"{i:4d} {time_sec:7.2f}s {rms:8.4f} {peak:8.4f} {zcr:6.0f} |{bar}")

    print("-" * 80)
    rms_values = np.array(rms_values)
    print(f"\nEnvelope statistics:")
    print(f"  Mean RMS: {np.mean(rms_values):.4f}")
    print(f"  Std RMS:  {np.std(rms_values):.4f}")
    print(f"  Min RMS:  {np.min(rms_values):.4f}")
    print(f"  Max RMS:  {np.max(rms_values):.4f}")
    print(f"  Variation (std/mean): {np.std(rms_values)/np.mean(rms_values):.3f}")

    # Check for silence (pauses)
    silence_threshold = 0.01
    silent_segments = np.sum(rms_values < silence_threshold)
    print(f"  Silent segments (<{silence_threshold}): {silent_segments}/{len(rms_values)}")

    return rms_values

def analyze_frequency_content(samples, sample_rate):
    """Analyze frequency content"""
    # Compute FFT of entire signal
    n = len(samples)
    fft = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(n, 1/sample_rate)

    # Find dominant frequencies
    top_k = 10
    top_indices = np.argsort(fft)[-top_k:][::-1]

    print(f"\nTop {top_k} dominant frequencies:")
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. {freqs[idx]:.1f} Hz (magnitude: {fft[idx]:.2f})")

    # Energy in frequency bands
    bands = [
        (0, 300, "Sub-bass/fundamental"),
        (300, 1000, "Low speech formants"),
        (1000, 3000, "Main speech content"),
        (3000, 6000, "High speech/sibilants"),
        (6000, 12000, "Very high/noise"),
    ]

    print(f"\nEnergy by frequency band:")
    total_energy = np.sum(fft ** 2)
    for low, high, name in bands:
        mask = (freqs >= low) & (freqs < high)
        band_energy = np.sum(fft[mask] ** 2)
        pct = 100 * band_energy / total_energy
        bar = '#' * int(pct / 2)
        print(f"  {low:5d}-{high:5d} Hz: {pct:5.1f}% |{bar}")

def check_periodicity(samples, sample_rate):
    """Check for periodic structure (pitch)"""
    # Use autocorrelation to find pitch
    # Typical pitch range: 80-400 Hz (male) or 150-500 Hz (female)
    min_period = int(sample_rate / 500)  # 500 Hz max pitch
    max_period = int(sample_rate / 80)   # 80 Hz min pitch

    # Take a segment from the middle
    mid = len(samples) // 2
    segment_len = sample_rate // 10  # 100ms
    segment = samples[mid:mid+segment_len]

    # Compute autocorrelation
    autocorr = np.correlate(segment, segment, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Take positive lags only
    autocorr = autocorr / autocorr[0]  # Normalize

    # Find peaks in the pitch range
    peak_idx = min_period + np.argmax(autocorr[min_period:max_period])
    peak_value = autocorr[peak_idx]

    estimated_pitch = sample_rate / peak_idx if peak_value > 0.3 else 0

    print(f"\nPeriodicity analysis (middle 100ms segment):")
    print(f"  Autocorrelation peak at lag {peak_idx} samples")
    print(f"  Peak autocorrelation value: {peak_value:.3f}")
    if estimated_pitch > 0:
        print(f"  Estimated pitch: {estimated_pitch:.1f} Hz")
        print(f"  (This suggests periodic/voiced speech)")
    else:
        print(f"  No clear pitch detected (unvoiced/noise)")

def main():
    if len(sys.argv) < 2:
        print("Usage: python detailed_analysis.py <wav_file>")
        sys.exit(1)

    wav_path = sys.argv[1]
    print(f"Loading: {wav_path}")
    samples, sample_rate = load_wav(wav_path)
    print(f"  Duration: {len(samples)/sample_rate:.2f}s, Sample rate: {sample_rate}Hz")

    # Segment analysis
    rms_values = analyze_segments(samples, sample_rate, segment_ms=100)

    # Frequency analysis
    analyze_frequency_content(samples, sample_rate)

    # Periodicity analysis
    check_periodicity(samples, sample_rate)

    print("\n" + "=" * 80)
    print("DIAGNOSIS:")
    print("=" * 80)

    # Check issues
    issues = []

    envelope_var = np.std(rms_values) / np.mean(rms_values)
    if envelope_var < 0.3:
        issues.append(f"- Constant envelope (variation={envelope_var:.2f}) - no pauses or loudness variation")

    silent_pct = np.sum(rms_values < 0.01) / len(rms_values)
    if silent_pct < 0.1:
        issues.append(f"- No silent segments ({silent_pct*100:.1f}%) - speech should have pauses")

    if issues:
        print("PROBLEMS DETECTED:")
        for issue in issues:
            print(issue)
        print("\nThis does NOT look like speech. The vocoder or code predictor may have bugs.")
    else:
        print("Audio characteristics appear speech-like!")

if __name__ == '__main__':
    main()
