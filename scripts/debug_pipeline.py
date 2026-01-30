#!/usr/bin/env python3
"""
Debug Pipeline: Compare Python (reference) vs C++ (leaxer) at each stage
Dumps intermediate tensors to files for comparison

Usage:
    python scripts/debug_pipeline.py \
        --model-path ./models/Qwen3-TTS-12Hz-0.6B-Base \
        --vocoder-path ./models/Qwen3-TTS-Tokenizer-12Hz \
        --text "Hello world" \
        --output-dir ./debug_output
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_tensor(path, tensor, name="tensor"):
    """Save tensor as both .npy and .bin with metadata"""
    if hasattr(tensor, 'detach'):
        arr = tensor.detach().cpu().float().numpy()
    else:
        arr = np.asarray(tensor, dtype=np.float32)
    
    np.save(f"{path}.npy", arr)
    arr.astype(np.float32).tofile(f"{path}.bin")
    
    with open(f"{path}.json", 'w') as f:
        json.dump({
            'name': name,
            'shape': list(arr.shape),
            'dtype': 'float32',
            'min': float(arr.min()),
            'max': float(arr.max()),
            'mean': float(arr.mean()),
            'std': float(arr.std()),
        }, f, indent=2)
    
    print(f"  Saved {path} shape={arr.shape} range=[{arr.min():.4f}, {arr.max():.4f}]")

def save_int_tensor(path, tensor, name="tensor"):
    """Save integer tensor"""
    if hasattr(tensor, 'detach'):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor)
    
    arr = arr.astype(np.int32)
    np.save(f"{path}.npy", arr)
    arr.tofile(f"{path}.bin")
    
    with open(f"{path}.json", 'w') as f:
        json.dump({
            'name': name,
            'shape': list(arr.shape),
            'dtype': 'int32',
            'min': int(arr.min()),
            'max': int(arr.max()),
        }, f, indent=2)
    
    print(f"  Saved {path} shape={arr.shape} range=[{arr.min()}, {arr.max()}]")


def debug_tokenizer(model_path, text, output_dir):
    """Test tokenizer and dump token IDs"""
    print("\n=== Stage 1: Tokenizer ===")
    
    try:
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        print(f"  Text: '{text}'")
        print(f"  Tokens: {tokens}")
        print(f"  Decoded: '{tokenizer.decode(tokens)}'")
        
        save_int_tensor(f"{output_dir}/01_tokens", tokens, "input_tokens")
        
        # Also save vocab info
        vocab_size = len(tokenizer)
        print(f"  Vocab size: {vocab_size}")
        
        # Special tokens
        special = {
            'bos_token_id': tokenizer.bos_token_id,
            'eos_token_id': tokenizer.eos_token_id,
            'pad_token_id': tokenizer.pad_token_id,
        }
        print(f"  Special tokens: {special}")
        
        with open(f"{output_dir}/01_tokenizer_info.json", 'w') as f:
            json.dump({
                'text': text,
                'tokens': tokens,
                'vocab_size': vocab_size,
                'special_tokens': special,
            }, f, indent=2)
        
        return tokens
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def debug_embedding(model, tokens, output_dir):
    """Test embedding layer"""
    print("\n=== Stage 2: Embedding ===")
    
    try:
        import torch
        
        token_tensor = torch.tensor([tokens], dtype=torch.long)
        
        # Get embedding weight
        if hasattr(model, 'talker'):
            embed_weight = model.talker.embed_tokens.weight
        else:
            embed_weight = model.model.embed_tokens.weight
        
        print(f"  Embed weight shape: {embed_weight.shape}")
        
        # Compute embedding
        embedded = torch.nn.functional.embedding(token_tensor, embed_weight)
        print(f"  Embedded shape: {embedded.shape}")
        
        save_tensor(f"{output_dir}/02_embedded", embedded[0], "embedded")
        
        return embedded
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def debug_text_projection(model, embedded, output_dir):
    """Test text projection (embedding_dim -> hidden_dim)"""
    print("\n=== Stage 3: Text Projection ===")
    
    try:
        import torch
        
        if hasattr(model, 'talker'):
            talker = model.talker
        else:
            talker = model.model
        
        # Check if text_proj exists
        if hasattr(talker, 'text_proj'):
            text_proj = talker.text_proj
            print(f"  text_proj found")
            
            # Get intermediate output
            x = embedded
            for i, layer in enumerate(text_proj):
                x = layer(x)
                print(f"    Layer {i}: {type(layer).__name__} -> shape {x.shape}")
            
            projected = x
        else:
            print(f"  No text_proj layer found, using embedding directly")
            projected = embedded
        
        save_tensor(f"{output_dir}/03_projected", projected[0], "projected")
        
        return projected
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def debug_transformer_layer(model, hidden, layer_idx, output_dir):
    """Test a single transformer layer"""
    print(f"\n=== Stage 4.{layer_idx}: Transformer Layer {layer_idx} ===")
    
    try:
        if hasattr(model, 'talker'):
            layers = model.talker.layers
        else:
            layers = model.model.layers
        
        layer = layers[layer_idx]
        
        # Forward through layer
        output = layer(hidden)[0]  # Returns tuple (hidden_states, ...)
        
        print(f"  Input shape: {hidden.shape}")
        print(f"  Output shape: {output.shape}")
        
        save_tensor(f"{output_dir}/04_layer{layer_idx:02d}_output", output[0], f"layer_{layer_idx}")
        
        return output
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def debug_lm_head(model, hidden, output_dir):
    """Test LM head (final projection to vocab)"""
    print("\n=== Stage 5: LM Head ===")
    
    try:
        import torch
        
        if hasattr(model, 'talker'):
            # Apply final norm
            normed = model.talker.norm(hidden)
            logits = model.talker.lm_head(normed)
        else:
            normed = model.model.norm(hidden)
            logits = model.lm_head(normed)
        
        print(f"  Normed shape: {normed.shape}")
        print(f"  Logits shape: {logits.shape}")
        
        save_tensor(f"{output_dir}/05_normed", normed[0], "normed")
        save_tensor(f"{output_dir}/05_logits", logits[0], "logits")
        
        # Get top predictions
        probs = torch.softmax(logits[0, -1], dim=-1)
        top_k = torch.topk(probs, 10)
        print(f"  Top 10 predictions at last position:")
        for i, (idx, prob) in enumerate(zip(top_k.indices.tolist(), top_k.values.tolist())):
            print(f"    {i+1}. token {idx}: {prob:.4f}")
        
        return logits
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def debug_weight_shapes(model, output_dir):
    """Dump all weight shapes for comparison with GGUF"""
    print("\n=== Weight Shapes ===")
    
    shapes = {}
    for name, param in model.named_parameters():
        shapes[name] = list(param.shape)
        if 'layer' not in name or '.0.' in name:  # Print first layer only
            print(f"  {name}: {param.shape}")
    
    with open(f"{output_dir}/weight_shapes.json", 'w') as f:
        json.dump(shapes, f, indent=2)
    
    print(f"  Saved {len(shapes)} weight shapes")


def main():
    parser = argparse.ArgumentParser(description="Debug TTS pipeline stages")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to Qwen3-TTS model")
    parser.add_argument("--vocoder-path", type=str, default=None,
                        help="Path to vocoder model (optional)")
    parser.add_argument("--text", type=str, default="Hello world",
                        help="Text to synthesize")
    parser.add_argument("--output-dir", type=str, default="./debug_output",
                        help="Output directory for debug tensors")
    parser.add_argument("--layers", type=int, default=2,
                        help="Number of transformer layers to debug")
    args = parser.parse_args()
    
    ensure_dir(args.output_dir)
    print(f"Debug output will be saved to: {args.output_dir}")
    
    # Stage 1: Tokenizer
    tokens = debug_tokenizer(args.model_path, args.text, args.output_dir)
    if tokens is None:
        print("Tokenizer failed, cannot continue")
        return
    
    # Load model
    print("\n=== Loading Model ===")
    try:
        import torch
        from transformers import AutoModelForCausalLM
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,  # Use FP32 for comparison
        )
        model.eval()
        print(f"  Model loaded: {type(model).__name__}")
        
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Debug weight shapes
    debug_weight_shapes(model, args.output_dir)
    
    # Stage 2: Embedding
    embedded = debug_embedding(model, tokens, args.output_dir)
    if embedded is None:
        return
    
    # Stage 3: Text Projection
    projected = debug_text_projection(model, embedded, args.output_dir)
    if projected is None:
        projected = embedded  # Fall back to embedding
    
    # Stage 4: Transformer layers (first N)
    hidden = projected
    for i in range(min(args.layers, 28)):
        hidden = debug_transformer_layer(model, hidden, i, args.output_dir)
        if hidden is None:
            break
    
    # Stage 5: LM Head (final)
    if hidden is not None:
        # Process through remaining layers
        print(f"\n=== Processing remaining layers ===")
        try:
            if hasattr(model, 'talker'):
                layers = model.talker.layers
            else:
                layers = model.model.layers
            
            for i in range(args.layers, len(layers)):
                hidden = layers[i](hidden)[0]
            print(f"  Processed layers {args.layers} to {len(layers)-1}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
        
        debug_lm_head(model, hidden, args.output_dir)
    
    print("\n=== Debug Complete ===")
    print(f"Output saved to: {args.output_dir}/")
    print("Compare these tensors with C++ output to find discrepancies")


if __name__ == "__main__":
    main()
