#!/usr/bin/env python3
"""
Audio Visualization Script for leaxer-qwen
Analyzes WAV files to determine if output is speech or noise.

Speech characteristics:
- Waveform: Varying amplitude, clear envelope patterns
- Spectrogram: Formants (horizontal bands), pitch contours, varied over time
- NOT: Constant frequency, white noise, or static patterns
"""

import sys
import wave
import struct
import numpy as np
import os

def load_wav(filepath):
    """Load WAV file and return samples as numpy array"""
    with wave.open(filepath, 'rb') as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()

        raw_data = wf.readframes(n_frames)

        if sample_width == 2:  # 16-bit
            samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:  # 32-bit
            samples = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        if n_channels > 1:
            samples = samples[::n_channels]  # Take first channel

        return samples, sample_rate

def compute_spectrogram(samples, sample_rate, window_size=1024, hop_size=256):
    """Compute spectrogram using STFT"""
    n_windows = (len(samples) - window_size) // hop_size + 1

    # Hann window
    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(window_size) / window_size))

    spectrogram = []
    for i in range(n_windows):
        start = i * hop_size
        frame = samples[start:start + window_size] * window
        spectrum = np.abs(np.fft.rfft(frame))
        spectrogram.append(spectrum)

    spectrogram = np.array(spectrogram).T  # [freq, time]

    # Convert to dB
    spectrogram = 20 * np.log10(spectrogram + 1e-10)

    return spectrogram, sample_rate / 2  # max_freq = Nyquist

def save_visualization_ppm(filepath, waveform, spectrogram, sample_rate, max_freq):
    """Save visualization as PPM image (no external dependencies)"""

    # Image dimensions
    width = 1200
    height = 800
    waveform_height = 200
    spec_height = 500
    margin = 50

    # Create RGB image buffer
    img = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

    # --- Draw waveform ---
    wave_y_center = margin + waveform_height // 2
    wave_x_start = margin
    wave_x_end = width - margin
    wave_width = wave_x_end - wave_x_start

    # Downsample waveform for display
    samples_per_pixel = max(1, len(waveform) // wave_width)

    prev_y = None
    for x in range(wave_width):
        start_idx = x * samples_per_pixel
        end_idx = min(start_idx + samples_per_pixel, len(waveform))
        if start_idx >= len(waveform):
            break

        # Get min/max for this pixel column
        chunk = waveform[start_idx:end_idx]
        min_val = np.min(chunk)
        max_val = np.max(chunk)

        # Map to y coordinates
        y_min = int(wave_y_center - max_val * (waveform_height // 2 - 10))
        y_max = int(wave_y_center - min_val * (waveform_height // 2 - 10))

        y_min = max(margin, min(margin + waveform_height, y_min))
        y_max = max(margin, min(margin + waveform_height, y_max))

        # Draw vertical line (blue)
        for y in range(y_min, y_max + 1):
            img[y, wave_x_start + x] = [0, 100, 200]

    # Draw waveform center line (gray)
    for x in range(wave_x_start, wave_x_end):
        img[wave_y_center, x] = [180, 180, 180]

    # --- Draw spectrogram ---
    spec_y_start = margin + waveform_height + 50
    spec_y_end = spec_y_start + spec_height
    spec_x_start = margin
    spec_x_end = width - margin
    spec_width = spec_x_end - spec_x_start

    # Normalize spectrogram
    spec_min = np.percentile(spectrogram, 5)
    spec_max = np.percentile(spectrogram, 95)
    spec_norm = (spectrogram - spec_min) / (spec_max - spec_min + 1e-10)
    spec_norm = np.clip(spec_norm, 0, 1)

    # Only show frequencies up to 8kHz (most speech content)
    freq_bins = spectrogram.shape[0]
    max_display_freq = 8000
    max_bin = int(freq_bins * max_display_freq / max_freq)
    spec_norm = spec_norm[:max_bin, :]

    # Resize spectrogram to fit display
    spec_resized = np.zeros((spec_height, spec_width))
    for y in range(spec_height):
        src_y = int((spec_height - 1 - y) * spec_norm.shape[0] / spec_height)
        src_y = min(src_y, spec_norm.shape[0] - 1)
        for x in range(spec_width):
            src_x = int(x * spec_norm.shape[1] / spec_width)
            src_x = min(src_x, spec_norm.shape[1] - 1)
            spec_resized[y, x] = spec_norm[src_y, src_x]

    # Apply colormap (viridis-like: dark blue -> green -> yellow)
    for y in range(spec_height):
        for x in range(spec_width):
            v = spec_resized[y, x]
            # Simple viridis-like colormap
            if v < 0.25:
                r = int(68 + v * 4 * (33 - 68))
                g = int(1 + v * 4 * (145 - 1))
                b = int(84 + v * 4 * (140 - 84))
            elif v < 0.5:
                t = (v - 0.25) * 4
                r = int(33 + t * (94 - 33))
                g = int(145 + t * (201 - 145))
                b = int(140 + t * (98 - 140))
            elif v < 0.75:
                t = (v - 0.5) * 4
                r = int(94 + t * (190 - 94))
                g = int(201 + t * (220 - 201))
                b = int(98 + t * (57 - 98))
            else:
                t = (v - 0.75) * 4
                r = int(190 + t * (253 - 190))
                g = int(220 + t * (231 - 220))
                b = int(57 + t * (37 - 57))

            img[spec_y_start + y, spec_x_start + x] = [r, g, b]

    # --- Add labels ---
    # (Simple text rendering not possible without dependencies, so we'll add markers)

    # Draw frequency markers on left (0, 2k, 4k, 6k, 8kHz)
    for freq in [0, 2000, 4000, 6000, 8000]:
        y = spec_y_start + int((1 - freq / max_display_freq) * spec_height)
        if spec_y_start <= y < spec_y_end:
            for x in range(spec_x_start - 5, spec_x_start):
                if 0 <= x < width:
                    img[y, x] = [0, 0, 0]

    # Save as PPM (simple format, no dependencies)
    ppm_path = filepath.replace('.png', '.ppm')
    with open(ppm_path, 'wb') as f:
        f.write(f"P6\n{width} {height}\n255\n".encode())
        f.write(img.tobytes())

    return ppm_path

def analyze_audio(samples, sample_rate, spectrogram):
    """Analyze audio characteristics to determine if it's speech-like"""

    results = {}

    # 1. Amplitude statistics
    results['peak'] = float(np.max(np.abs(samples)))
    results['rms'] = float(np.sqrt(np.mean(samples ** 2)))
    results['crest_factor'] = results['peak'] / (results['rms'] + 1e-10)

    # 2. Zero-crossing rate (speech typically 50-200 per 10ms)
    zero_crossings = np.sum(np.abs(np.diff(np.sign(samples))) > 0)
    duration = len(samples) / sample_rate
    results['zcr_per_second'] = zero_crossings / duration

    # 3. Spectral centroid (where is the "center of mass" of the spectrum)
    freqs = np.linspace(0, sample_rate / 2, spectrogram.shape[0])
    spec_linear = 10 ** (spectrogram / 20)  # Convert back from dB
    centroid = np.sum(freqs[:, np.newaxis] * spec_linear, axis=0) / (np.sum(spec_linear, axis=0) + 1e-10)
    results['mean_spectral_centroid'] = float(np.mean(centroid))
    results['std_spectral_centroid'] = float(np.std(centroid))

    # 4. Spectral flatness (noise vs tonal)
    # Flatness = geometric mean / arithmetic mean (1 = white noise, 0 = pure tone)
    geometric_mean = np.exp(np.mean(np.log(spec_linear + 1e-10), axis=0))
    arithmetic_mean = np.mean(spec_linear, axis=0)
    flatness = geometric_mean / (arithmetic_mean + 1e-10)
    results['mean_spectral_flatness'] = float(np.mean(flatness))

    # 5. Temporal variation (speech has varying envelope)
    window_ms = 20
    window_samples = int(window_ms * sample_rate / 1000)
    n_windows = len(samples) // window_samples
    envelope = []
    for i in range(n_windows):
        chunk = samples[i * window_samples:(i + 1) * window_samples]
        envelope.append(np.sqrt(np.mean(chunk ** 2)))
    envelope = np.array(envelope)
    results['envelope_std'] = float(np.std(envelope))
    results['envelope_mean'] = float(np.mean(envelope))
    results['envelope_variation'] = results['envelope_std'] / (results['envelope_mean'] + 1e-10)

    # 6. Assessment
    issues = []

    # Check for white noise (high flatness)
    if results['mean_spectral_flatness'] > 0.8:
        issues.append("HIGH spectral flatness (%.2f) - sounds like white noise" % results['mean_spectral_flatness'])

    # Check for constant amplitude (low envelope variation)
    if results['envelope_variation'] < 0.3:
        issues.append("LOW envelope variation (%.2f) - constant amplitude like noise" % results['envelope_variation'])

    # Check spectral centroid (speech typically 500-2000 Hz)
    if results['mean_spectral_centroid'] > 4000:
        issues.append("HIGH spectral centroid (%.0f Hz) - energy concentrated in high frequencies" % results['mean_spectral_centroid'])

    # Check for very low variation in spectral centroid (constant tone)
    if results['std_spectral_centroid'] < 200:
        issues.append("LOW spectral centroid variation (%.0f Hz) - sounds like constant tone" % results['std_spectral_centroid'])

    # Check crest factor (speech typically 10-20 dB, noise ~3-4)
    if results['crest_factor'] < 3:
        issues.append("LOW crest factor (%.1f) - clipped or distorted" % results['crest_factor'])

    results['issues'] = issues
    results['is_speech_like'] = len(issues) == 0

    return results

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_audio.py <wav_file> [output_image]")
        sys.exit(1)

    wav_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else wav_path.replace('.wav', '_analysis.ppm')

    print(f"Loading: {wav_path}")
    samples, sample_rate = load_wav(wav_path)
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Duration: {len(samples) / sample_rate:.2f} seconds")
    print(f"  Samples: {len(samples)}")

    print("\nComputing spectrogram...")
    spectrogram, max_freq = compute_spectrogram(samples, sample_rate)
    print(f"  Spectrogram shape: {spectrogram.shape} (freq bins x time frames)")

    print("\nAnalyzing audio characteristics...")
    analysis = analyze_audio(samples, sample_rate, spectrogram)

    print("\n" + "=" * 60)
    print("AUDIO ANALYSIS RESULTS")
    print("=" * 60)
    print(f"  Peak amplitude: {analysis['peak']:.4f}")
    print(f"  RMS amplitude: {analysis['rms']:.4f}")
    print(f"  Crest factor: {analysis['crest_factor']:.2f}")
    print(f"  Zero-crossing rate: {analysis['zcr_per_second']:.0f}/sec")
    print(f"  Mean spectral centroid: {analysis['mean_spectral_centroid']:.0f} Hz")
    print(f"  Spectral centroid std: {analysis['std_spectral_centroid']:.0f} Hz")
    print(f"  Spectral flatness: {analysis['mean_spectral_flatness']:.3f}")
    print(f"  Envelope variation: {analysis['envelope_variation']:.3f}")

    print("\n" + "-" * 60)
    if analysis['is_speech_like']:
        print("ASSESSMENT: Audio characteristics are SPEECH-LIKE")
    else:
        print("ASSESSMENT: Audio does NOT look like speech")
        print("\nIssues detected:")
        for issue in analysis['issues']:
            print(f"  - {issue}")
    print("-" * 60)

    print(f"\nSaving visualization to: {output_path}")
    saved_path = save_visualization_ppm(output_path, samples, spectrogram, sample_rate, max_freq)
    print(f"  Saved: {saved_path}")

    # Also save a simple text report
    report_path = wav_path.replace('.wav', '_report.txt')
    with open(report_path, 'w') as f:
        f.write("Audio Analysis Report\n")
        f.write("=" * 40 + "\n")
        f.write(f"File: {wav_path}\n")
        f.write(f"Sample rate: {sample_rate} Hz\n")
        f.write(f"Duration: {len(samples) / sample_rate:.2f} seconds\n\n")
        f.write("Metrics:\n")
        for key, value in analysis.items():
            if key not in ['issues', 'is_speech_like']:
                f.write(f"  {key}: {value}\n")
        f.write(f"\nSpeech-like: {analysis['is_speech_like']}\n")
        if analysis['issues']:
            f.write("\nIssues:\n")
            for issue in analysis['issues']:
                f.write(f"  - {issue}\n")
    print(f"  Report: {report_path}")

    return 0 if analysis['is_speech_like'] else 1

if __name__ == '__main__':
    sys.exit(main())
