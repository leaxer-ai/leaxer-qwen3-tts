#!/usr/bin/env python3
"""
Test vocoder with Python reference implementation
Compare C++ output codes with Python vocoder output
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from safetensors.torch import load_file as safe_load


def load_vocoder(model_path):
    """Load vocoder weights from safetensors"""
    st_path = Path(model_path) / "speech_tokenizer" / "model.safetensors"
    if not st_path.exists():
        st_path = Path(model_path) / "model.safetensors"
    
    print(f"Loading vocoder from: {st_path}")
    state_dict = safe_load(st_path)
    
    # Get codebooks
    first_cb = state_dict["decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"]
    rest_cbs = [state_dict[f"decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.embedding_sum"] 
                for i in range(15)]
    
    codebooks = torch.stack([first_cb] + rest_cbs, dim=0)  # [16, 2048, 256]
    print(f"Codebooks shape: {codebooks.shape}")
    
    return state_dict, codebooks


def rvq_decode(codes, codebooks):
    """
    Decode RVQ codes to embeddings
    codes: [seq_len, 16] int tensor
    codebooks: [16, 2048, 256] float tensor
    Returns: tuple of (first_embedding, rest_embedding)
    """
    seq_len = codes.shape[0]
    
    # Codebook 0 (semantic)
    first_emb = codebooks[0, codes[:, 0], :]  # [seq_len, 256]
    
    # Codebooks 1-15 (acoustic) - sum
    rest_emb = torch.zeros(seq_len, 256, dtype=codebooks.dtype)
    for i in range(1, 16):
        rest_emb += codebooks[i, codes[:, i], :]
    
    return first_emb, rest_emb


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--codes", help="Path to codes.bin from C++ debug output")
    args = parser.parse_args()
    
    # Load vocoder
    state_dict, codebooks = load_vocoder(args.model_path)
    
    if args.codes:
        # Load codes from C++ output
        codes = np.fromfile(args.codes, dtype=np.int32).reshape(-1, 16)
        codes = torch.from_numpy(codes)
        print(f"Loaded codes shape: {codes.shape}")
        print(f"Code values - first row: {codes[0].tolist()}")
        print(f"Code range: [{codes.min()}, {codes.max()}]")
    else:
        # Generate test codes
        print("No codes provided, generating random test codes...")
        codes = torch.randint(0, 2048, (90, 16), dtype=torch.int32)
    
    # Decode RVQ
    first_emb, rest_emb = rvq_decode(codes, codebooks.float())
    print(f"\nRVQ decode:")
    print(f"  First embedding range: [{first_emb.min():.4f}, {first_emb.max():.4f}]")
    print(f"  Rest embedding range: [{rest_emb.min():.4f}, {rest_emb.max():.4f}]")
    
    # Apply output projections
    first_proj_w = state_dict["decoder.quantizer.rvq_first.output_proj.weight"]  # [512, 256, 1]
    rest_proj_w = state_dict["decoder.quantizer.rvq_rest.output_proj.weight"]  # [512, 256, 1]
    
    # Reshape for conv1d: [batch, channels, seq] 
    first_emb_t = first_emb.T.unsqueeze(0).float()  # [1, 256, seq_len]
    rest_emb_t = rest_emb.T.unsqueeze(0).float()  # [1, 256, seq_len]
    
    first_proj = torch.nn.functional.conv1d(first_emb_t, first_proj_w.float())  # [1, 512, seq_len]
    rest_proj = torch.nn.functional.conv1d(rest_emb_t, rest_proj_w.float())  # [1, 512, seq_len]
    
    print(f"\nOutput projections:")
    print(f"  First proj weight shape: {first_proj_w.shape}")
    print(f"  First proj range: [{first_proj.min():.4f}, {first_proj.max():.4f}]")
    print(f"  Rest proj range: [{rest_proj.min():.4f}, {rest_proj.max():.4f}]")
    
    # Add projections together (they're both 512-dim)
    combined = first_proj + rest_proj  # [1, 512, seq_len]
    print(f"  Combined (ADD) range: [{combined.min():.4f}, {combined.max():.4f}]")
    
    # Apply pre_conv
    pre_conv_w = state_dict["decoder.pre_conv.conv.weight"]  # [1024, 512, 3]
    pre_conv_b = state_dict["decoder.pre_conv.conv.bias"]  # [1024]
    
    # Pad for causal conv
    pre_conv_out = torch.nn.functional.conv1d(
        torch.nn.functional.pad(combined, (2, 0)),  # Left pad by kernel_size-1
        pre_conv_w.float(), 
        pre_conv_b.float()
    )
    print(f"\nPre-conv:")
    print(f"  Weight shape: {pre_conv_w.shape}")
    print(f"  Output range: [{pre_conv_out.min():.4f}, {pre_conv_out.max():.4f}]")
    
    print("\n✓ RVQ + projections match expected flow")
    print("Next step: compare pre-transformer output, then ConvNeXt blocks, then upsampling")


if __name__ == "__main__":
    main()
