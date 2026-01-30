#!/usr/bin/env python3
"""
Dump ground truth values from Python Qwen3-TTS for C++ comparison.
Focuses on tokenizer + prefill sequence (no model inference needed).
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from safetensors.torch import load_file as safe_load

MODEL_PATH = Path(__file__).parent.parent / "models" / "Qwen3-TTS-12Hz-0.6B-Base"

def main():
    print("=" * 60)
    print("Qwen3-TTS Python Reference - Debug Values")
    print("=" * 60)
    
    # Load tokenizer
    from transformers import AutoTokenizer
    
    print(f"\n[1] Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    # === Test text ===
    text = "Hi"
    
    print(f"\n[2] Tokenizing text: '{text}'")
    
    # Basic token encoding
    text_tokens = tokenizer.encode(text, add_special_tokens=False)
    print(f"  Text tokens (no special): {text_tokens}")
    print(f"  Decoded: '{tokenizer.decode(text_tokens)}'")
    
    # Get special token IDs from config
    config_path = MODEL_PATH / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    tts_bos_token_id = config["tts_bos_token_id"]
    tts_eos_token_id = config["tts_eos_token_id"]
    im_start_token_id = config["im_start_token_id"]
    im_end_token_id = config["im_end_token_id"]
    assistant_token_id = config["assistant_token_id"]
    
    talker_cfg = config["talker_config"]
    codec_bos_id = talker_cfg["codec_bos_id"]
    codec_eos_id = talker_cfg["codec_eos_token_id"]
    codec_pad_id = talker_cfg["codec_pad_id"]
    codec_lang_english = talker_cfg["codec_language_id"]["english"]
    codec_nothink_id = talker_cfg["codec_nothink_id"]
    
    print(f"\n[3] Special Tokens (Text Domain):")
    print(f"  tts_bos_token_id: {tts_bos_token_id} -> {repr(tokenizer.decode([tts_bos_token_id]))}")
    print(f"  tts_eos_token_id: {tts_eos_token_id} -> {repr(tokenizer.decode([tts_eos_token_id]))}")
    print(f"  im_start_token_id: {im_start_token_id} -> {repr(tokenizer.decode([im_start_token_id]))}")
    print(f"  im_end_token_id: {im_end_token_id} -> {repr(tokenizer.decode([im_end_token_id]))}")
    print(f"  assistant_token_id: {assistant_token_id}")
    
    print(f"\n[4] Special Tokens (Codec Domain):")
    print(f"  codec_bos_id: {codec_bos_id}")
    print(f"  codec_eos_id: {codec_eos_id}")
    print(f"  codec_pad_id: {codec_pad_id}")
    print(f"  codec_lang_english: {codec_lang_english}")
    print(f"  codec_nothink_id: {codec_nothink_id}")
    
    # Build prefill sequence matching Qwen3-TTS format
    # Based on typical format: <|im_start|>{text}<|im_end|><tts_text_bos>
    print(f"\n[5] Building prefill sequence...")
    
    prefill_tokens = [im_start_token_id] + text_tokens + [im_end_token_id, tts_bos_token_id]
    
    print(f"\n[6] Prefill Token IDs ({len(prefill_tokens)} tokens):")
    print(f"  {prefill_tokens}")
    
    print(f"\n[7] Token-by-token decode:")
    for i, tok in enumerate(prefill_tokens):
        try:
            decoded = tokenizer.decode([tok])
            print(f"  [{i:3d}] {tok:6d} -> {repr(decoded)}")
        except:
            print(f"  [{i:3d}] {tok:6d} -> <decode error>")
    
    # Load model weights to inspect embedding structure
    print(f"\n[8] Loading model weights for structure inspection...")
    model_path = MODEL_PATH / "model.safetensors"
    state_dict = safe_load(str(model_path))
    
    # Find relevant weight names
    print(f"\n[9] Weight tensor names containing 'embed':")
    for name in sorted(state_dict.keys()):
        if 'embed' in name.lower():
            shape = tuple(state_dict[name].shape)
            print(f"  {name}: {shape}")
    
    # Check talker structure
    print(f"\n[10] Talker config from model:")
    print(f"  hidden_size: {talker_cfg['hidden_size']}")
    print(f"  num_hidden_layers: {talker_cfg['num_hidden_layers']}")
    print(f"  vocab_size (codec): {talker_cfg['vocab_size']}")
    print(f"  text_vocab_size: {talker_cfg['text_vocab_size']}")
    print(f"  text_hidden_size: {talker_cfg['text_hidden_size']}")
    print(f"  num_attention_heads: {talker_cfg['num_attention_heads']}")
    print(f"  head_dim: {talker_cfg['head_dim']}")
    
    # Expected prefill for codec generation:
    # After text is processed, model should output:
    # [codec_lang_id, codec_nothink_id, ...codec_tokens..., codec_eos_id]
    print(f"\n[11] Expected codec sequence start:")
    print(f"  First token after text: codec_lang_english = {codec_lang_english}")
    print(f"  Second token: codec_nothink_id = {codec_nothink_id}")
    print(f"  Then: codec tokens (0-2047 range)")
    print(f"  End: codec_eos_id = {codec_eos_id}")
    
    # Check embedding weights
    print(f"\n[12] Embedding weight inspection:")
    if 'talker.embed_tokens.weight' in state_dict:
        emb = state_dict['talker.embed_tokens.weight']
        print(f"  talker.embed_tokens.weight: {tuple(emb.shape)}")
        # Sample first few embeddings
        print(f"  Token 0 (codec pad?) first 5 values: {emb[0, :5].tolist()}")
        print(f"  Token {codec_bos_id} first 5 values: {emb[codec_bos_id, :5].tolist()}")
        print(f"  Token {codec_lang_english} first 5 values: {emb[codec_lang_english, :5].tolist()}")
    
    # Check if there's a separate text embedding
    if 'talker.text_embed_tokens.weight' in state_dict:
        text_emb = state_dict['talker.text_embed_tokens.weight']
        print(f"\n  talker.text_embed_tokens.weight: {tuple(text_emb.shape)}")
        # Sample embedding for text token
        if text_tokens[0] < text_emb.shape[0]:
            print(f"  Token {text_tokens[0]} ('{text}') first 5 values: {text_emb[text_tokens[0], :5].tolist()}")
    
    # === Save results ===
    output_dir = Path(__file__).parent.parent / ".todo"
    output_dir.mkdir(exist_ok=True)
    
    results = {
        "text": text,
        "text_tokens": text_tokens,
        "prefill_tokens": prefill_tokens,
        "special_tokens": {
            "tts_bos_token_id": tts_bos_token_id,
            "tts_eos_token_id": tts_eos_token_id,
            "im_start_token_id": im_start_token_id,
            "im_end_token_id": im_end_token_id,
            "assistant_token_id": assistant_token_id,
            "codec_bos_id": codec_bos_id,
            "codec_eos_id": codec_eos_id,
            "codec_pad_id": codec_pad_id,
            "codec_lang_english": codec_lang_english,
            "codec_nothink_id": codec_nothink_id,
        },
        "talker_config": {
            "hidden_size": talker_cfg["hidden_size"],
            "text_hidden_size": talker_cfg["text_hidden_size"],
            "num_hidden_layers": talker_cfg["num_hidden_layers"],
            "vocab_size": talker_cfg["vocab_size"],
            "text_vocab_size": talker_cfg["text_vocab_size"],
        },
        "expected_codec_start": [codec_lang_english, codec_nothink_id],
    }
    
    # Write markdown report
    md_path = output_dir / "PYTHON_DEBUG_OUTPUT.md"
    with open(md_path, 'w') as f:
        f.write("# Python Reference Debug Output\n\n")
        f.write(f"**Text:** `{text}`\n\n")
        
        f.write("## Text Tokens\n")
        f.write(f"```\n{text_tokens}\n```\n\n")
        
        f.write("## Special Token IDs (Text Domain)\n")
        f.write("| Token | ID | Decoded |\n|-------|----|---------|\n")
        f.write(f"| tts_bos_token_id | {tts_bos_token_id} | `<tts_text_bos>` |\n")
        f.write(f"| tts_eos_token_id | {tts_eos_token_id} | `<tts_text_eos>` |\n")
        f.write(f"| im_start_token_id | {im_start_token_id} | `<\\|im_start\\|>` |\n")
        f.write(f"| im_end_token_id | {im_end_token_id} | `<\\|im_end\\|>` |\n")
        f.write("\n")
        
        f.write("## Special Token IDs (Codec Domain)\n")
        f.write("| Token | ID |\n|-------|----|n")
        f.write(f"| codec_bos_id | {codec_bos_id} |\n")
        f.write(f"| codec_eos_id | {codec_eos_id} |\n")
        f.write(f"| codec_pad_id | {codec_pad_id} |\n")
        f.write(f"| codec_lang_english | {codec_lang_english} |\n")
        f.write(f"| codec_nothink_id | {codec_nothink_id} |\n")
        f.write("\n")
        
        f.write("## Prefill Sequence\n")
        f.write(f"Length: {len(prefill_tokens)} tokens\n\n")
        f.write("```\n")
        for i, tok in enumerate(prefill_tokens):
            try:
                decoded = tokenizer.decode([tok])
                f.write(f"[{i:3d}] {tok:6d} -> {repr(decoded)}\n")
            except:
                f.write(f"[{i:3d}] {tok:6d} -> <special>\n")
        f.write("```\n\n")
        
        f.write("## Expected Codec Output Start\n")
        f.write("After processing the text prefill, the model should generate:\n")
        f.write(f"1. `{codec_lang_english}` (English language ID)\n")
        f.write(f"2. `{codec_nothink_id}` (no-think mode)\n")
        f.write(f"3. Codec tokens in range [0, 2047]\n")
        f.write(f"4. End with `{codec_eos_id}` (codec EOS)\n\n")
        
        f.write("## Talker Model Config\n")
        f.write(f"- hidden_size: {talker_cfg['hidden_size']}\n")
        f.write(f"- text_hidden_size: {talker_cfg['text_hidden_size']}\n")
        f.write(f"- num_hidden_layers: {talker_cfg['num_hidden_layers']}\n")
        f.write(f"- vocab_size (codec): {talker_cfg['vocab_size']}\n")
        f.write(f"- text_vocab_size: {talker_cfg['text_vocab_size']}\n")
    
    print(f"\n[13] Results saved to: {md_path}")
    
    # Also save JSON
    json_path = output_dir / "python_debug.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  JSON saved to: {json_path}")
    
    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
