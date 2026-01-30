#!/usr/bin/env python3
"""
Run ONNX inference for Qwen3-TTS to get ground truth codec tokens.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

ONNX_DIR = Path(__file__).parent.parent / "hf_onnx_bundle" / "onnx_kv_06b"
MODEL_PATH = Path(__file__).parent.parent / "models" / "Qwen3-TTS-12Hz-0.6B-Base"

def main():
    import onnxruntime as ort
    
    print("=" * 60)
    print("ONNX Inference - Qwen3-TTS")
    print("=" * 60)
    
    # Load config
    config_path = MODEL_PATH / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    talker_cfg = config["talker_config"]
    
    # Special tokens
    tts_bos_token_id = config["tts_bos_token_id"]  # 151672
    im_start_token_id = config["im_start_token_id"]  # 151644
    im_end_token_id = config["im_end_token_id"]  # 151645
    codec_bos_id = talker_cfg["codec_bos_id"]  # 2149
    codec_eos_id = talker_cfg["codec_eos_token_id"]  # 2150
    codec_lang_english = talker_cfg["codec_language_id"]["english"]  # 2050
    codec_nothink_id = talker_cfg["codec_nothink_id"]  # 2155
    
    # Text token for "Hi"
    text_token = 13048
    
    # Build prefill sequence: <|im_start|>Hi<|im_end|><tts_text_bos>
    prefill_tokens = [im_start_token_id, text_token, im_end_token_id, tts_bos_token_id]
    print(f"\n[1] Prefill tokens: {prefill_tokens}")
    
    # Check available ONNX models
    print(f"\n[2] Available ONNX models in {ONNX_DIR}:")
    for f in sorted(ONNX_DIR.glob("*.onnx")):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name} ({size_mb:.1f} MB)")
    
    # Load text_project.onnx to see input/output
    print(f"\n[3] Inspecting text_project.onnx...")
    text_proj_path = ONNX_DIR / "text_project.onnx"
    if text_proj_path.exists():
        sess = ort.InferenceSession(str(text_proj_path), providers=['CPUExecutionProvider'])
        
        print("  Inputs:")
        for inp in sess.get_inputs():
            print(f"    {inp.name}: {inp.shape} ({inp.type})")
        
        print("  Outputs:")
        for out in sess.get_outputs():
            print(f"    {out.name}: {out.shape} ({out.type})")
        
        # Try running it
        # Assuming input is token embeddings from text embedding
        # text_hidden_size = 2048
        hidden_size = talker_cfg["text_hidden_size"]  # 2048
        target_hidden = talker_cfg["hidden_size"]  # 1024
        
        # Create dummy input
        seq_len = len(prefill_tokens)
        dummy_input = np.random.randn(1, seq_len, hidden_size).astype(np.float32)
        
        try:
            result = sess.run(None, {sess.get_inputs()[0].name: dummy_input})
            print(f"\n  Output shape: {result[0].shape}")
            print(f"  Output range: [{result[0].min():.4f}, {result[0].max():.4f}]")
        except Exception as e:
            print(f"  Error: {e}")
    
    # Inspect codec_embed.onnx
    print(f"\n[4] Inspecting codec_embed.onnx...")
    codec_embed_path = ONNX_DIR / "codec_embed.onnx"
    if codec_embed_path.exists():
        sess = ort.InferenceSession(str(codec_embed_path), providers=['CPUExecutionProvider'])
        
        print("  Inputs:")
        for inp in sess.get_inputs():
            print(f"    {inp.name}: {inp.shape} ({inp.type})")
        
        print("  Outputs:")
        for out in sess.get_outputs():
            print(f"    {out.name}: {out.shape} ({out.type})")
        
        # Try running it with a codec token
        codec_token = np.array([[codec_lang_english]], dtype=np.int64)
        try:
            result = sess.run(None, {sess.get_inputs()[0].name: codec_token})
            print(f"\n  Embedding for codec {codec_lang_english} (English):")
            print(f"    Shape: {result[0].shape}")
            print(f"    First 5 values: {result[0][0, 0, :5].tolist()}")
        except Exception as e:
            print(f"  Error: {e}")
    
    # Inspect talker_prefill.onnx
    print(f"\n[5] Inspecting talker_prefill.onnx...")
    talker_prefill_path = ONNX_DIR / "talker_prefill.onnx"
    if talker_prefill_path.exists():
        sess = ort.InferenceSession(str(talker_prefill_path), providers=['CPUExecutionProvider'])
        
        print("  Inputs:")
        for inp in sess.get_inputs():
            print(f"    {inp.name}: {inp.shape} ({inp.type})")
        
        print("  Outputs:")
        for out in sess.get_outputs():
            print(f"    {out.name}: {out.shape} ({out.type})")
    
    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
