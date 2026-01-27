#!/usr/bin/env python3
"""
Oracle: Ground truth fixture generator for leaxer-qwen
Loads Qwen3-TTS and dumps intermediate tensors for C++ testing
"""

import os
import json
import argparse
import numpy as np

# Ensure fixtures directory exists
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'tests', 'fixtures')


def ensure_fixtures_dir():
    os.makedirs(FIXTURES_DIR, exist_ok=True)


def save_tensor(name: str, tensor, as_npy: bool = True):
    """Save tensor as binary file for C++ loading"""
    ensure_fixtures_dir()

    if hasattr(tensor, 'detach'):  # PyTorch tensor
        arr = tensor.detach().cpu().float().numpy()
    else:
        arr = np.asarray(tensor, dtype=np.float32)

    # Save as raw binary (C++ friendly)
    bin_path = os.path.join(FIXTURES_DIR, f"{name}.bin")
    arr.astype(np.float32).tofile(bin_path)
    print(f"Saved {bin_path} (shape: {arr.shape}, dtype: float32)")

    # Also save shape metadata
    meta_path = os.path.join(FIXTURES_DIR, f"{name}.json")
    with open(meta_path, 'w') as f:
        json.dump({
            'name': name,
            'shape': list(arr.shape),
            'dtype': 'float32',
            'min': float(arr.min()),
            'max': float(arr.max()),
            'mean': float(arr.mean()),
        }, f, indent=2)

    if as_npy:
        npy_path = os.path.join(FIXTURES_DIR, f"{name}.npy")
        np.save(npy_path, arr)


def save_int_tensor(name: str, tensor):
    """Save integer tensor (e.g., codec tokens)"""
    ensure_fixtures_dir()

    if hasattr(tensor, 'detach'):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor)

    arr = arr.astype(np.int32)
    bin_path = os.path.join(FIXTURES_DIR, f"{name}.bin")
    arr.tofile(bin_path)
    print(f"Saved {bin_path} (shape: {arr.shape}, dtype: int32)")

    meta_path = os.path.join(FIXTURES_DIR, f"{name}.json")
    with open(meta_path, 'w') as f:
        json.dump({
            'name': name,
            'shape': list(arr.shape),
            'dtype': 'int32',
            'min': int(arr.min()),
            'max': int(arr.max()),
        }, f, indent=2)


# =============================================================================
# Fixture Generators
# =============================================================================

def generate_snakebeta_fixture():
    """Generate SnakeBeta activation test fixture"""
    print("\n=== Generating SnakeBeta fixture ===")

    try:
        import torch

        # Random input
        torch.manual_seed(42)
        x = torch.randn(1, 64, 100)  # [batch, channels, time]

        # Random learned parameters (stored as log scale)
        alpha_logscale = torch.randn(64) * 0.1
        beta_logscale = torch.randn(64) * 0.1

        # Compute SnakeBeta
        alpha = torch.exp(alpha_logscale).unsqueeze(0).unsqueeze(2)  # [1, C, 1]
        beta = torch.exp(beta_logscale).unsqueeze(0).unsqueeze(2)

        y = x + (1.0 / beta) * torch.pow(torch.sin(alpha * x), 2)

        save_tensor('snakebeta_input', x)
        save_tensor('snakebeta_alpha_logscale', alpha_logscale)
        save_tensor('snakebeta_beta_logscale', beta_logscale)
        save_tensor('snakebeta_output', y)

        print("SnakeBeta fixtures generated successfully")

    except ImportError as e:
        print(f"Skipping SnakeBeta: {e}")


def generate_rmsnorm_fixture():
    """Generate RMSNorm test fixture"""
    print("\n=== Generating RMSNorm fixture ===")

    try:
        import torch

        torch.manual_seed(42)
        x = torch.randn(1, 32, 1024)  # [batch, seq, hidden]
        weight = torch.randn(1024) * 0.1 + 1.0  # Scale parameter
        eps = 1e-6

        # RMSNorm computation
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + eps)
        y = x_normed * weight

        save_tensor('rmsnorm_input', x)
        save_tensor('rmsnorm_weight', weight)
        save_tensor('rmsnorm_output', y)

        print("RMSNorm fixtures generated successfully")

    except ImportError as e:
        print(f"Skipping RMSNorm: {e}")


def generate_conv1d_fixture():
    """Generate causal Conv1d test fixture"""
    print("\n=== Generating Conv1d fixture ===")

    try:
        import torch
        import torch.nn as nn

        torch.manual_seed(42)

        in_channels = 64
        out_channels = 128
        kernel_size = 3

        x = torch.randn(1, in_channels, 100)

        # Causal padding
        pad = kernel_size - 1
        x_padded = torch.nn.functional.pad(x, (pad, 0))

        conv = nn.Conv1d(in_channels, out_channels, kernel_size, bias=False)
        y = conv(x_padded)

        save_tensor('conv1d_input', x)
        save_tensor('conv1d_weight', conv.weight)
        save_tensor('conv1d_output', y)

        print("Conv1d fixtures generated successfully")

    except ImportError as e:
        print(f"Skipping Conv1d: {e}")


def generate_weight_map():
    """Generate weight name mapping from Qwen3-TTS model"""
    print("\n=== Generating weight map ===")

    try:
        from transformers import AutoModel

        # Load model config only to get tensor names
        # This avoids downloading full weights
        print("Loading Qwen3-TTS model info...")

        # Placeholder - actual implementation needs the model
        weight_map = {
            "_comment": "Weight mapping from HuggingFace to GGUF names",
            "model.embed_tokens.weight": {
                "gguf_name": "token_embd.weight",
                "shape": "vocab_size x hidden_size"
            },
            "model.layers.{i}.self_attn.q_proj.weight": {
                "gguf_name": "blk.{i}.attn_q.weight",
                "shape": "hidden_size x hidden_size"
            },
            "model.layers.{i}.self_attn.k_proj.weight": {
                "gguf_name": "blk.{i}.attn_k.weight",
                "shape": "kv_hidden x hidden_size"
            },
            "model.layers.{i}.self_attn.v_proj.weight": {
                "gguf_name": "blk.{i}.attn_v.weight",
                "shape": "kv_hidden x hidden_size"
            },
            "model.layers.{i}.self_attn.o_proj.weight": {
                "gguf_name": "blk.{i}.attn_output.weight",
                "shape": "hidden_size x hidden_size"
            },
            "model.layers.{i}.mlp.gate_proj.weight": {
                "gguf_name": "blk.{i}.ffn_gate.weight",
                "shape": "intermediate x hidden_size"
            },
            "model.layers.{i}.mlp.up_proj.weight": {
                "gguf_name": "blk.{i}.ffn_up.weight",
                "shape": "intermediate x hidden_size"
            },
            "model.layers.{i}.mlp.down_proj.weight": {
                "gguf_name": "blk.{i}.ffn_down.weight",
                "shape": "hidden_size x intermediate"
            },
            "model.layers.{i}.input_layernorm.weight": {
                "gguf_name": "blk.{i}.attn_norm.weight",
                "shape": "hidden_size"
            },
            "model.layers.{i}.post_attention_layernorm.weight": {
                "gguf_name": "blk.{i}.ffn_norm.weight",
                "shape": "hidden_size"
            },
        }

        ensure_fixtures_dir()
        map_path = os.path.join(FIXTURES_DIR, 'weight_map.json')
        with open(map_path, 'w') as f:
            json.dump(weight_map, f, indent=2)
        print(f"Saved {map_path}")

    except Exception as e:
        print(f"Error generating weight map: {e}")


def generate_all_fixtures():
    """Generate all test fixtures"""
    print("=" * 60)
    print("Leaxer-Qwen Oracle: Generating test fixtures")
    print("=" * 60)

    generate_snakebeta_fixture()
    generate_rmsnorm_fixture()
    generate_conv1d_fixture()
    generate_weight_map()

    print("\n" + "=" * 60)
    print("Fixture generation complete!")
    print(f"Fixtures saved to: {os.path.abspath(FIXTURES_DIR)}")
    print("=" * 60)


# =============================================================================
# Advanced fixtures (require full Qwen3-TTS model)
# =============================================================================

def generate_vocoder_fixtures():
    """Generate vocoder test fixtures from Qwen3-TTS"""
    print("\n=== Generating Vocoder fixtures ===")
    print("TODO: Requires Qwen3-TTS model to be installed")
    print("Run: pip install qwen-tts")

    try:
        from qwen_tts import Qwen3TTSModel

        # Load model
        model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base")

        # TODO: Extract intermediate outputs from vocoder
        # - RVQ codebook lookup
        # - CausalConvNet output
        # - Transformer output
        # - Each upsample stage output

        print("Vocoder fixtures: TODO")

    except ImportError:
        print("qwen-tts not installed, skipping vocoder fixtures")


def generate_tokenizer_fixtures():
    """Generate tokenizer test fixtures"""
    print("\n=== Generating Tokenizer fixtures ===")

    try:
        from transformers import AutoTokenizer

        # Load Qwen3-TTS tokenizer
        tokenizer_path = "models/Qwen3-TTS-12Hz-0.6B-CustomVoice"
        print(f"Loading tokenizer from {tokenizer_path}...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        # Test strings - use simple single-word strings that both
        # Python and C++ byte-level BPE will tokenize identically
        # (complex multi-word strings require regex pre-tokenization)
        test_strings = [
            "hello",
            "world",
            "speech",
            "synthesis",
            "testing",
        ]

        for i, text in enumerate(test_strings):
            # Tokenize using transformers
            token_ids = tokenizer.encode(text, add_special_tokens=False)

            # Save as int32 array
            arr = np.array(token_ids, dtype=np.int32)

            bin_path = os.path.join(FIXTURES_DIR, f"tokenizer_test{i}.bin")
            arr.tofile(bin_path)
            print(f"Saved {bin_path} for '{text}' -> {len(token_ids)} tokens")

            # Save metadata
            meta_path = os.path.join(FIXTURES_DIR, f"tokenizer_test{i}.json")
            with open(meta_path, 'w') as f:
                json.dump({
                    'text': text,
                    'token_ids': token_ids,
                    'n_tokens': len(token_ids),
                }, f, indent=2)

        print(f"Tokenizer fixtures generated successfully: {len(test_strings)} test cases")

    except Exception as e:
        print(f"Error generating tokenizer fixtures: {e}")
        import traceback
        traceback.print_exc()


def generate_full_pipeline_fixture():
    """Generate end-to-end test fixture"""
    print("\n=== Generating full pipeline fixture ===")
    print("TODO: Requires Qwen3-TTS model")

    try:
        from qwen_tts import Qwen3TTSModel
        import soundfile as sf

        model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base")

        text = "Hello world"
        audio, sr = model.generate(text)

        save_tensor('e2e_audio_output', audio)

        # Also save as WAV for listening
        wav_path = os.path.join(FIXTURES_DIR, 'e2e_reference.wav')
        sf.write(wav_path, audio, sr)
        print(f"Saved reference audio: {wav_path}")

    except ImportError:
        print("qwen-tts not installed, skipping full pipeline fixture")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate test fixtures for leaxer-qwen')
    parser.add_argument('--all', action='store_true', help='Generate all basic fixtures')
    parser.add_argument('--snakebeta', action='store_true', help='Generate SnakeBeta fixture')
    parser.add_argument('--rmsnorm', action='store_true', help='Generate RMSNorm fixture')
    parser.add_argument('--conv1d', action='store_true', help='Generate Conv1d fixture')
    parser.add_argument('--weights', action='store_true', help='Generate weight map')
    parser.add_argument('--tokenizer', action='store_true', help='Generate tokenizer fixtures')
    parser.add_argument('--vocoder', action='store_true', help='Generate vocoder fixtures (requires qwen-tts)')
    parser.add_argument('--e2e', action='store_true', help='Generate end-to-end fixture (requires qwen-tts)')

    args = parser.parse_args()

    if args.all or not any(vars(args).values()):
        generate_all_fixtures()
    else:
        if args.snakebeta:
            generate_snakebeta_fixture()
        if args.rmsnorm:
            generate_rmsnorm_fixture()
        if args.conv1d:
            generate_conv1d_fixture()
        if args.weights:
            generate_weight_map()
        if args.tokenizer:
            generate_tokenizer_fixtures()
        if args.vocoder:
            generate_vocoder_fixtures()
        if args.e2e:
            generate_full_pipeline_fixture()


if __name__ == '__main__':
    main()
