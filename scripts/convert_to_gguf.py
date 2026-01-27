#!/usr/bin/env python3
"""
Convert Qwen3-TTS PyTorch weights to GGUF format using official gguf library

Usage:
    python scripts/convert_to_gguf.py \
        --model-path ./models/Qwen3-TTS-12Hz-0.6B-Base \
        --output qwen3_tts_0.6b.gguf

    python scripts/convert_to_gguf.py \
        --model-path ./models/Qwen3-TTS-Tokenizer-12Hz \
        --output vocoder.gguf \
        --vocoder

Supports:
    - Qwen3-TTS-12Hz-0.6B-Base (28 talker layers + 5 code predictor layers)
    - Qwen3-TTS-12Hz-1.7B-Base
    - Qwen3-TTS-Tokenizer-12Hz (vocoder/decoder)
"""

import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np

try:
    from safetensors.torch import load_file as safe_load
except ImportError:
    print("Error: safetensors not installed. Run: pip install safetensors", file=sys.stderr)
    sys.exit(1)

try:
    import torch
except ImportError:
    print("Error: PyTorch not installed. Run: pip install torch", file=sys.stderr)
    sys.exit(1)

try:
    import gguf
except ImportError:
    print("Error: gguf not installed. Run: pip install gguf", file=sys.stderr)
    sys.exit(1)


def convert_qwen_tts_to_gguf(
    model_path: str,
    output_path: str,
    use_f16: bool = True
):
    """
    Convert Qwen3-TTS model to GGUF format using official gguf library

    Args:
        model_path: Path to local model directory with config.json and model.safetensors
        output_path: Output GGUF file path
        use_f16: Use FP16 instead of FP32
    """
    print(f"Loading model from: {model_path}")

    # Load config from JSON file
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"Model config loaded: {config.get('model_type', 'unknown')}")

    # Extract talker config (main transformer)
    talker_config = config.get("talker_config", config)
    code_predictor_config = talker_config.get("code_predictor_config", {})

    # Initialize GGUF writer with official library
    writer = gguf.GGUFWriter(output_path, arch="qwen3-tts")

    # Write metadata from config
    writer.add_name(Path(model_path).name)
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16 if use_f16 else gguf.LlamaFileType.ALL_F32)

    # Talker model hyperparameters
    writer.add_context_length(talker_config.get("max_position_embeddings", 32768))
    writer.add_embedding_length(talker_config.get("hidden_size", 1024))
    writer.add_block_count(talker_config.get("num_hidden_layers", 28))
    writer.add_feed_forward_length(talker_config.get("intermediate_size", 3072))
    writer.add_head_count(talker_config.get("num_attention_heads", 16))
    writer.add_head_count_kv(talker_config.get("num_key_value_heads", 8))
    writer.add_layer_norm_rms_eps(talker_config.get("rms_norm_eps", 1e-06))
    writer.add_rope_freq_base(float(talker_config.get("rope_theta", 1000000)))

    # Vocab size
    writer.add_vocab_size(talker_config.get("text_vocab_size", 151936))

    # TTS-specific metadata (custom keys)
    writer.add_uint32("qwen3.tts.audio_sample_rate", 24000)
    writer.add_uint32("qwen3.tts.audio_token_rate", 12)
    writer.add_uint32("qwen3.tts.n_codebooks", talker_config.get("num_code_groups", 16))
    writer.add_uint32("qwen3.tts.codebook_size", 2048)
    writer.add_uint32("qwen3.code_predictor.block_count", code_predictor_config.get("num_hidden_layers", 5))
    writer.add_uint32("qwen3.code_predictor.hidden_size", code_predictor_config.get("hidden_size", 1024))

    # Special token IDs
    writer.add_bos_token_id(config.get("tts_bos_token_id", 151672))
    writer.add_eos_token_id(config.get("tts_eos_token_id", 151673))
    writer.add_pad_token_id(config.get("tts_pad_token_id", 151671))

    writer.add_uint32("qwen3.tts.codec_bos_id", talker_config.get("codec_bos_id", 2149))
    writer.add_uint32("qwen3.tts.codec_eos_id", talker_config.get("codec_eos_token_id", 2150))
    writer.add_uint32("qwen3.tts.codec_pad_id", talker_config.get("codec_pad_id", 2148))

    # Load model weights from safetensors
    safetensors_path = Path(model_path) / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"model.safetensors not found in {model_path}")

    print("Loading model weights from safetensors...")

    state_dict = safe_load(safetensors_path)
    print(f"Found {len(state_dict)} tensors in safetensors file")

    # Convert and add all tensors
    n_converted = 0
    for name, tensor in state_dict.items():
        # Convert bfloat16 to float16 (numpy doesn't support bfloat16)
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float16)
        elif tensor.dtype == torch.float32 and use_f16:
            tensor = tensor.to(torch.float16)

        # Convert to numpy
        np_tensor = tensor.cpu().numpy()

        # Convert name: replace dots with underscores and shorten for GGUF compatibility
        # GGML has a 64 character limit for tensor names
        gguf_name = name.replace(".", "_")

        # Shorten common long prefixes to fit GGML 64-char limit
        # Layer naming
        gguf_name = gguf_name.replace("talker_code_predictor_model_layers", "talker_cp_l")
        gguf_name = gguf_name.replace("code_predictor_model_layers", "cp_l")
        gguf_name = gguf_name.replace("talker_model_layers", "tk_l")

        # Attention and norm naming
        gguf_name = gguf_name.replace("post_attention_layernorm", "post_ln")
        gguf_name = gguf_name.replace("input_layernorm", "in_ln")
        gguf_name = gguf_name.replace("self_attn", "attn")

        # Embedding naming (handle both conventions)
        gguf_name = gguf_name.replace("text_embedding", "emb")
        gguf_name = gguf_name.replace("embed_tokens", "emb")

        # FFN/MLP naming - standardize to ffn
        gguf_name = gguf_name.replace("_mlp_gate_proj", "_ffn_gate_proj")
        gguf_name = gguf_name.replace("_mlp_up_proj", "_ffn_up_proj")
        gguf_name = gguf_name.replace("_mlp_down_proj", "_ffn_down_proj")

        # Code predictor output heads
        gguf_name = gguf_name.replace("talker_code_predictor_lm_head", "talker_code_predictor_output_heads")

        # Truncate if still too long
        if len(gguf_name) > 63:
            gguf_name = gguf_name[:63]

        # Determine tensor type
        if np_tensor.dtype == np.float16:
            tensor_type = gguf.GGMLQuantizationType.F16
        elif np_tensor.dtype == np.float32:
            tensor_type = gguf.GGMLQuantizationType.F32
        else:
            # Convert unknown types to float32
            np_tensor = np_tensor.astype(np.float32)
            tensor_type = gguf.GGMLQuantizationType.F32

        writer.add_tensor(gguf_name, np_tensor, raw_dtype=tensor_type)
        n_converted += 1

        if n_converted % 50 == 0:
            print(f"  Converted {n_converted} tensors...")

    print(f"Converted {n_converted} tensors total")

    # Write GGUF file
    print(f"Writing GGUF file to: {output_path}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    # Verify file was created
    file_size = os.path.getsize(output_path)
    print(f"GGUF file written successfully ({file_size / (1024**2):.2f} MB)")


def convert_vocoder_to_gguf(
    model_path: str,
    output_path: str,
    use_f16: bool = True
):
    """
    Convert Qwen3-TTS-Tokenizer-12Hz decoder (vocoder) to GGUF format

    Args:
        model_path: Path to Tokenizer-12Hz model directory
        output_path: Output GGUF file path
        use_f16: Use FP16 instead of FP32
    """
    print(f"Loading vocoder from: {model_path}")

    # Load config from JSON file
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"Vocoder config loaded: {config.get('model_type', 'unknown')}")

    # Extract decoder config
    decoder_config = config.get("decoder_config", {})

    # Initialize GGUF writer
    writer = gguf.GGUFWriter(output_path, arch="qwen3-tts-vocoder")

    # Write metadata
    writer.add_name(Path(model_path).name)
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16 if use_f16 else gguf.LlamaFileType.ALL_F32)

    # Vocoder-specific metadata
    writer.add_uint32("qwen3.vocoder.n_codebooks", decoder_config.get("n_codebooks", 16))
    writer.add_uint32("qwen3.vocoder.codebook_size", decoder_config.get("codebook_size", 2048))
    writer.add_uint32("qwen3.vocoder.sample_rate", decoder_config.get("sample_rate", 24000))
    writer.add_uint32("qwen3.vocoder.hidden_dim", decoder_config.get("hidden_dim", 512))

    # Load model weights from safetensors
    safetensors_path = Path(model_path) / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"model.safetensors not found in {model_path}")

    print("Loading vocoder weights from safetensors...")

    state_dict = safe_load(safetensors_path)
    print(f"Found {len(state_dict)} tensors in safetensors file")

    # Filter only decoder tensors (ignore encoder)
    decoder_tensors = {k: v for k, v in state_dict.items() if k.startswith("decoder.")}
    print(f"Found {len(decoder_tensors)} decoder tensors")

    # Convert and add decoder tensors
    n_converted = 0
    for name, tensor in decoder_tensors.items():
        # Convert bfloat16 to float16
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float16)
        elif tensor.dtype == torch.float32 and use_f16:
            tensor = tensor.to(torch.float16)

        # Convert to numpy
        np_tensor = tensor.cpu().numpy()

        # Convert name: replace dots with underscores
        gguf_name = name.replace(".", "_")

        # Truncate if too long
        if len(gguf_name) > 63:
            gguf_name = gguf_name[:63]

        # Determine tensor type
        if np_tensor.dtype == np.float16:
            tensor_type = gguf.GGMLQuantizationType.F16
        elif np_tensor.dtype == np.float32:
            tensor_type = gguf.GGMLQuantizationType.F32
        else:
            np_tensor = np_tensor.astype(np.float32)
            tensor_type = gguf.GGMLQuantizationType.F32

        writer.add_tensor(gguf_name, np_tensor, raw_dtype=tensor_type)
        n_converted += 1

        if n_converted % 50 == 0:
            print(f"  Converted {n_converted} tensors...")

    print(f"Converted {n_converted} tensors total")

    # Write GGUF file
    print(f"Writing GGUF file to: {output_path}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    # Verify file was created
    file_size = os.path.getsize(output_path)
    print(f"GGUF file written successfully ({file_size / (1024**2):.2f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Qwen3-TTS PyTorch weights to GGUF format"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model directory with config.json and model.safetensors"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output GGUF file path"
    )
    parser.add_argument(
        "--use-f16",
        action="store_true",
        default=True,
        help="Use FP16 instead of FP32 (default: True)"
    )
    parser.add_argument(
        "--use-f32",
        action="store_true",
        help="Use FP32 instead of FP16"
    )
    parser.add_argument(
        "--vocoder",
        action="store_true",
        help="Convert vocoder (decoder) from Tokenizer-12Hz model instead of talker"
    )

    args = parser.parse_args()

    # Handle FP16/FP32 flag
    use_f16 = args.use_f16 and not args.use_f32

    try:
        if args.vocoder:
            convert_vocoder_to_gguf(
                model_path=args.model_path,
                output_path=args.output,
                use_f16=use_f16
            )
        else:
            convert_qwen_tts_to_gguf(
                model_path=args.model_path,
                output_path=args.output,
                use_f16=use_f16
            )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
