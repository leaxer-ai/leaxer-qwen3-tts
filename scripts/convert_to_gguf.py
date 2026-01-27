#!/usr/bin/env python3
"""
Convert Qwen3-TTS PyTorch weights to GGUF format

Usage:
    python scripts/convert_to_gguf.py \
        --model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
        --output qwen3_tts_0.6b.gguf

Supports:
    - Qwen3-TTS-12Hz-0.6B-Base (12 layers)
    - Qwen3-TTS-12Hz-1.7B-Base (20 layers)
"""

import os
import sys
import argparse
import struct
import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

# Try importing required libraries with helpful error messages
try:
    import torch
except ImportError:
    print("Error: PyTorch not installed. Run: pip install torch", file=sys.stderr)
    sys.exit(1)

try:
    from transformers import AutoModel, AutoConfig
except ImportError:
    print("Error: transformers not installed. Run: pip install transformers", file=sys.stderr)
    sys.exit(1)

# GGUF constants (from gguf specification)
GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_VERSION = 3

# GGML type enumeration
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8

# GGUF metadata value types
GGUF_METADATA_VALUE_TYPE_UINT8 = 0
GGUF_METADATA_VALUE_TYPE_INT8 = 1
GGUF_METADATA_VALUE_TYPE_UINT16 = 2
GGUF_METADATA_VALUE_TYPE_INT16 = 3
GGUF_METADATA_VALUE_TYPE_UINT32 = 4
GGUF_METADATA_VALUE_TYPE_INT32 = 5
GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6
GGUF_METADATA_VALUE_TYPE_BOOL = 7
GGUF_METADATA_VALUE_TYPE_STRING = 8
GGUF_METADATA_VALUE_TYPE_ARRAY = 9
GGUF_METADATA_VALUE_TYPE_UINT64 = 10
GGUF_METADATA_VALUE_TYPE_INT64 = 11
GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12


class GGUFWriter:
    """Minimal GGUF writer for Qwen3-TTS model conversion"""

    def __init__(self, output_path: str, use_f16: bool = True):
        self.output_path = output_path
        self.use_f16 = use_f16
        self.metadata: Dict[str, Any] = {}
        self.tensors: Dict[str, np.ndarray] = {}
        self.tensor_type: Dict[str, int] = {}

    def add_metadata(self, key: str, value: Any):
        """Add metadata key-value pair"""
        self.metadata[key] = value

    def add_tensor(self, name: str, tensor: np.ndarray, dtype: Optional[int] = None):
        """Add a tensor to be written to GGUF file"""
        if dtype is None:
            dtype = GGML_TYPE_F16 if self.use_f16 else GGML_TYPE_F32

        self.tensors[name] = tensor
        self.tensor_type[name] = dtype

    def write(self):
        """Write GGUF file"""
        with open(self.output_path, 'wb') as f:
            # Write header
            f.write(struct.pack('<I', GGUF_MAGIC))
            f.write(struct.pack('<I', GGUF_VERSION))
            f.write(struct.pack('<Q', len(self.tensors)))  # n_tensors
            f.write(struct.pack('<Q', len(self.metadata)))  # n_kv

            # Write metadata
            for key, value in self.metadata.items():
                self._write_string(f, key)
                self._write_metadata_value(f, value)

            # Write tensor info
            for name, tensor in self.tensors.items():
                self._write_string(f, name)
                n_dims = len(tensor.shape)
                f.write(struct.pack('<I', n_dims))

                # Write dimensions in GGUF format (reversed from numpy)
                for dim in reversed(tensor.shape):
                    f.write(struct.pack('<Q', dim))

                # Write tensor type
                f.write(struct.pack('<I', self.tensor_type[name]))

                # Write offset (placeholder, will compute after all info is written)
                offset_pos = f.tell()
                f.write(struct.pack('<Q', 0))

            # Align to 32 bytes
            data_start = f.tell()
            padding = (32 - (data_start % 32)) % 32
            f.write(b'\0' * padding)
            data_start += padding

            # Write tensor data and record offsets
            tensor_offsets = {}
            for name, tensor in self.tensors.items():
                tensor_offsets[name] = f.tell() - data_start

                # Convert and write tensor data
                data = self._convert_tensor(tensor, self.tensor_type[name])
                f.write(data.tobytes())

                # Align to 32 bytes
                current_pos = f.tell()
                padding = (32 - (current_pos % 32)) % 32
                f.write(b'\0' * padding)

            # Go back and write correct offsets
            current_pos = f.tell()
            metadata_size = 8 + 8 + 8 + 8  # header
            for key, value in self.metadata.items():
                metadata_size += self._get_metadata_size(key, value)

            offset_write_pos = metadata_size
            for name in self.tensors.keys():
                # Seek to offset position for this tensor
                n_dims = len(self.tensors[name].shape)
                offset_write_pos += self._get_string_size(name)
                offset_write_pos += 4  # n_dims
                offset_write_pos += 8 * n_dims  # dimensions
                offset_write_pos += 4  # type

                f.seek(offset_write_pos)
                f.write(struct.pack('<Q', tensor_offsets[name]))
                offset_write_pos += 8

            # Seek back to end
            f.seek(current_pos)

    def _write_string(self, f, s: str):
        """Write a string in GGUF format"""
        encoded = s.encode('utf-8')
        f.write(struct.pack('<Q', len(encoded)))
        f.write(encoded)

    def _get_string_size(self, s: str) -> int:
        """Get size of string in bytes"""
        return 8 + len(s.encode('utf-8'))

    def _write_metadata_value(self, f, value: Any):
        """Write metadata value with type tag"""
        if isinstance(value, bool):
            f.write(struct.pack('<I', GGUF_METADATA_VALUE_TYPE_BOOL))
            f.write(struct.pack('<?', value))
        elif isinstance(value, int):
            if value >= 0 and value < 2**32:
                f.write(struct.pack('<I', GGUF_METADATA_VALUE_TYPE_UINT32))
                f.write(struct.pack('<I', value))
            else:
                f.write(struct.pack('<I', GGUF_METADATA_VALUE_TYPE_INT64))
                f.write(struct.pack('<q', value))
        elif isinstance(value, float):
            f.write(struct.pack('<I', GGUF_METADATA_VALUE_TYPE_FLOAT32))
            f.write(struct.pack('<f', value))
        elif isinstance(value, str):
            f.write(struct.pack('<I', GGUF_METADATA_VALUE_TYPE_STRING))
            self._write_string(f, value)
        elif isinstance(value, list):
            f.write(struct.pack('<I', GGUF_METADATA_VALUE_TYPE_ARRAY))
            # Simplified: assume all elements are same type
            if len(value) > 0:
                elem_type = GGUF_METADATA_VALUE_TYPE_INT32 if isinstance(value[0], int) else GGUF_METADATA_VALUE_TYPE_FLOAT32
                f.write(struct.pack('<I', elem_type))
                f.write(struct.pack('<Q', len(value)))
                for v in value:
                    if elem_type == GGUF_METADATA_VALUE_TYPE_INT32:
                        f.write(struct.pack('<i', v))
                    else:
                        f.write(struct.pack('<f', v))
        else:
            raise ValueError(f"Unsupported metadata type: {type(value)}")

    def _get_metadata_size(self, key: str, value: Any) -> int:
        """Calculate size of metadata entry"""
        size = self._get_string_size(key) + 4  # key + type tag
        if isinstance(value, bool):
            size += 1
        elif isinstance(value, int):
            size += 4 if (value >= 0 and value < 2**32) else 8
        elif isinstance(value, float):
            size += 4
        elif isinstance(value, str):
            size += self._get_string_size(value)
        elif isinstance(value, list):
            size += 4 + 8  # elem_type + count
            if len(value) > 0:
                size += len(value) * (4 if isinstance(value[0], int) else 4)
        return size

    def _convert_tensor(self, tensor: np.ndarray, dtype: int) -> np.ndarray:
        """Convert tensor to specified GGML type"""
        if dtype == GGML_TYPE_F32:
            return tensor.astype(np.float32)
        elif dtype == GGML_TYPE_F16:
            return tensor.astype(np.float16)
        else:
            # For quantized types, just use F32 for now (quantization is complex)
            print(f"Warning: Quantization type {dtype} not implemented, using F32")
            return tensor.astype(np.float32)


def convert_qwen_tts_to_gguf(
    model_path: str,
    output_path: str,
    use_f16: bool = True,
    vocab_only: bool = False
):
    """
    Convert Qwen3-TTS model to GGUF format

    Args:
        model_path: Path or HuggingFace model ID (e.g., "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
        output_path: Output GGUF file path
        use_f16: Use FP16 instead of FP32
        vocab_only: Only convert vocabulary/config (skip weights)
    """
    print(f"Loading model from: {model_path}")

    # Load config
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print(f"Model config loaded: {config.model_type}")

    # Initialize GGUF writer
    writer = GGUFWriter(output_path, use_f16=use_f16)

    # Write metadata from config
    writer.add_metadata("general.architecture", "qwen3-tts")
    writer.add_metadata("general.name", model_path.split("/")[-1])
    writer.add_metadata("general.file_type", 1 if use_f16 else 0)

    # Model hyperparameters
    writer.add_metadata("qwen3.context_length", getattr(config, "max_position_embeddings", 32768))
    writer.add_metadata("qwen3.embedding_length", config.hidden_size)
    writer.add_metadata("qwen3.block_count", config.num_hidden_layers)
    writer.add_metadata("qwen3.feed_forward_length", config.intermediate_size)
    writer.add_metadata("qwen3.attention.head_count", config.num_attention_heads)
    writer.add_metadata("qwen3.attention.head_count_kv", config.num_key_value_heads)
    writer.add_metadata("qwen3.attention.layer_norm_rms_epsilon", config.rms_norm_eps)
    writer.add_metadata("qwen3.rope.freq_base", getattr(config, "rope_theta", 10000.0))

    # Vocabulary
    writer.add_metadata("tokenizer.ggml.model", "gpt2")
    writer.add_metadata("tokenizer.ggml.vocab_size", config.vocab_size)
    writer.add_metadata("tokenizer.ggml.bos_token_id", config.bos_token_id if hasattr(config, "bos_token_id") else 151643)
    writer.add_metadata("tokenizer.ggml.eos_token_id", config.eos_token_id if hasattr(config, "eos_token_id") else 151643)
    writer.add_metadata("tokenizer.ggml.padding_token_id", getattr(config, "pad_token_id", 151643))

    # TTS-specific metadata
    writer.add_metadata("qwen3.tts.audio_sample_rate", 24000)
    writer.add_metadata("qwen3.tts.audio_token_rate", 12)  # 12 Hz
    writer.add_metadata("qwen3.tts.n_codebooks", 16)
    writer.add_metadata("qwen3.tts.codebook_size", 2048)

    if vocab_only:
        print("Vocabulary-only mode: skipping weight conversion")
        writer.write()
        print(f"GGUF file written to: {output_path}")
        return

    # Load model weights
    print("Loading model weights...")
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if use_f16 else torch.float32,
        trust_remote_code=True
    )
    model.eval()

    print("Converting weights to GGUF format...")

    # Convert state dict to GGUF tensors
    state_dict = model.state_dict()

    # Weight name mapping: HuggingFace -> GGUF
    # This follows llama.cpp naming conventions
    tensor_map = {
        "model.embed_tokens.weight": "token_embd.weight",
        "model.norm.weight": "output_norm.weight",
    }

    # Layer-specific mappings
    for i in range(config.num_hidden_layers):
        tensor_map.update({
            f"model.layers.{i}.self_attn.q_proj.weight": f"blk.{i}.attn_q.weight",
            f"model.layers.{i}.self_attn.k_proj.weight": f"blk.{i}.attn_k.weight",
            f"model.layers.{i}.self_attn.v_proj.weight": f"blk.{i}.attn_v.weight",
            f"model.layers.{i}.self_attn.o_proj.weight": f"blk.{i}.attn_output.weight",
            f"model.layers.{i}.mlp.gate_proj.weight": f"blk.{i}.ffn_gate.weight",
            f"model.layers.{i}.mlp.up_proj.weight": f"blk.{i}.ffn_up.weight",
            f"model.layers.{i}.mlp.down_proj.weight": f"blk.{i}.ffn_down.weight",
            f"model.layers.{i}.input_layernorm.weight": f"blk.{i}.attn_norm.weight",
            f"model.layers.{i}.post_attention_layernorm.weight": f"blk.{i}.ffn_norm.weight",
        })

    # Convert and add tensors
    n_converted = 0
    for hf_name, gguf_name in tensor_map.items():
        if hf_name in state_dict:
            tensor = state_dict[hf_name]
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.cpu().numpy()
            writer.add_tensor(gguf_name, tensor)
            n_converted += 1

    # Also add any vocoder/code predictor weights if present
    for name, tensor in state_dict.items():
        # Skip already mapped tensors
        if name in tensor_map:
            continue

        # Convert name to GGUF format (replace dots with underscores, etc.)
        gguf_name = name.replace(".", "_")

        if isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu().numpy()

        writer.add_tensor(gguf_name, tensor)
        n_converted += 1

    print(f"Converted {n_converted} tensors")

    # Write GGUF file
    print(f"Writing GGUF file to: {output_path}")
    writer.write()

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
        help="Path to model or HuggingFace model ID (e.g., Qwen/Qwen3-TTS-12Hz-0.6B-Base)"
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
        "--vocab-only",
        action="store_true",
        help="Only convert vocabulary and config (skip weights)"
    )

    args = parser.parse_args()

    # Handle FP16/FP32 flag
    use_f16 = args.use_f16 and not args.use_f32

    try:
        convert_qwen_tts_to_gguf(
            model_path=args.model_path,
            output_path=args.output,
            use_f16=use_f16,
            vocab_only=args.vocab_only
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
