#!/usr/bin/env python3
"""Compare tensor shapes between safetensors and GGUF files."""

import os
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import re

# Paths
SAFETENSOR_PATH = "/Users/rizki/leaxer-qwen3-tts/models/Qwen3-TTS-12Hz-0.6B-Base/model.safetensors"
GGUF_PATH = "/Users/rizki/leaxer-qwen3-tts/qwen3_tts_0.6b.gguf"
OUTPUT_PATH = "/Users/rizki/leaxer-qwen3-tts/.todo/TENSOR_COMPARISON.md"

def load_safetensors_info():
    """Load safetensors and extract tensor info."""
    from safetensors import safe_open
    import torch
    
    tensors = {}
    with safe_open(SAFETENSOR_PATH, framework="pt") as f:
        for name in f.keys():
            tensor = f.get_tensor(name)
            # Convert to float32 for statistics
            tensor_f32 = tensor.float()
            tensors[name] = {
                'shape': tuple(tensor.shape),
                'dtype': str(tensor.dtype),
                'min': float(tensor_f32.min().item()),
                'max': float(tensor_f32.max().item()),
                'mean': float(tensor_f32.mean().item()),
                'std': float(tensor_f32.std().item()),
            }
    return tensors

def load_gguf_info():
    """Load GGUF and extract tensor info."""
    from gguf import GGUFReader
    
    reader = GGUFReader(GGUF_PATH)
    tensors = {}
    
    for tensor in reader.tensors:
        name = tensor.name
        shape = tuple(tensor.shape)
        dtype = str(tensor.tensor_type)
        
        # Get actual data for statistics
        data = tensor.data
        if hasattr(data, 'astype'):
            data_float = data.astype(np.float32)
        else:
            data_float = np.array(data, dtype=np.float32)
        
        tensors[name] = {
            'shape': shape,
            'dtype': dtype,
            'min': float(np.min(data_float)) if data_float.size > 0 else 0,
            'max': float(np.max(data_float)) if data_float.size > 0 else 0,
            'mean': float(np.mean(data_float)) if data_float.size > 0 else 0,
            'std': float(np.std(data_float)) if data_float.size > 0 else 0,
        }
    
    return tensors

# Build comprehensive name mapping
NAME_MAP = {
    # Embeddings
    'talker.model.text_embedding.weight': 'talker_model_text_emb_weight',
    'talker.model.codec_embedding.weight': 'talker_model_codec_emb_weight',
    
    # Output heads
    'talker.codec_head.weight': 'talker_codec_head_weight',
    
    # Layer patterns - will be handled specially
}

def build_name_mappings(st_names, gguf_names):
    """Build bidirectional name mappings based on patterns."""
    st_to_gguf = {}
    gguf_to_st = {}
    
    # 1. Try simple underscore conversion first
    gguf_set = set(gguf_names)
    for st_name in st_names:
        simple = st_name.replace('.', '_')
        if simple in gguf_set:
            st_to_gguf[st_name] = simple
            gguf_to_st[simple] = st_name
    
    # 2. Apply known transformations for unmatched
    transforms = [
        # Standard Qwen naming
        (r'self_attn\.q_proj', 'attn_q'),
        (r'self_attn\.k_proj', 'attn_k'),
        (r'self_attn\.v_proj', 'attn_v'),
        (r'self_attn\.o_proj', 'attn_output'),
        (r'self_attn\.q_norm', 'attn_q_norm'),
        (r'self_attn\.k_norm', 'attn_k_norm'),
        (r'mlp\.gate_proj', 'ffn_gate'),
        (r'mlp\.up_proj', 'ffn_up'),
        (r'mlp\.down_proj', 'ffn_down'),
        (r'input_layernorm', 'attn_norm'),
        (r'post_attention_layernorm', 'ffn_norm'),
        # Code predictor
        (r'code_predictor\.lm_head', 'code_predictor_output_heads'),
        (r'code_predictor\.tok_embeddings', 'code_predictor_tok_emb'),
        (r'code_predictor\.transformer\.layers\.(\d+)', r'code_predictor_transformer_l\1'),
        (r'code_predictor\.transformer\.norm', 'code_predictor_transformer_ln'),
        # Embeddings
        (r'text_embedding', 'text_emb'),
        (r'codec_embedding', 'codec_emb'),
        (r'tok_embeddings', 'tok_emb'),
        # Linear
        (r'linear_fc1', 'fc1'),
        (r'linear_fc2', 'fc2'),
        # Talker model layers
        (r'talker\.model\.layers\.(\d+)', r'talker_model_l\1'),
        (r'talker\.model\.norm', 'talker_model_ln'),
    ]
    
    for st_name in st_names:
        if st_name in st_to_gguf:
            continue
            
        transformed = st_name
        for pattern, replacement in transforms:
            transformed = re.sub(pattern, replacement, transformed)
        transformed = transformed.replace('.', '_')
        
        if transformed in gguf_set:
            st_to_gguf[st_name] = transformed
            gguf_to_st[transformed] = st_name
    
    return st_to_gguf, gguf_to_st

def shapes_match(shape1, shape2):
    """Check if shapes match (considering possible transposition)."""
    if shape1 == shape2:
        return "✅ exact"
    if len(shape1) == len(shape2) == 2:
        if shape1 == (shape2[1], shape2[0]):
            return "⚠️ TRANSPOSED"
    if len(shape1) != len(shape2):
        return f"❌ rank ({len(shape1)}D vs {len(shape2)}D)"
    return "❌ MISMATCH"

def generate_report(st_tensors, gguf_tensors):
    """Generate the comparison report."""
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    st_to_gguf, gguf_to_st = build_name_mappings(st_tensors.keys(), gguf_tensors.keys())
    
    lines = []
    lines.append("# Tensor Shape Comparison: safetensors vs GGUF\n\n")
    lines.append(f"**Safetensors:** `{SAFETENSOR_PATH}`\n")
    lines.append(f"**GGUF:** `{GGUF_PATH}`\n\n")
    lines.append(f"| Metric | Safetensors | GGUF |\n")
    lines.append(f"|--------|-------------|------|\n")
    lines.append(f"| Total tensors | {len(st_tensors)} | {len(gguf_tensors)} |\n")
    lines.append(f"| Matched by name | {len(st_to_gguf)} | {len(gguf_to_st)} |\n\n")
    
    # Key tensors of interest
    key_tensors = [
        ('talker.model.text_embedding.weight', 'Text embedding - vocab×dim'),
        ('talker.model.codec_embedding.weight', 'Codec embedding - codebook×dim'),
        ('talker.codec_head.weight', 'Codec output head'),
    ]
    
    # Add code_predictor embeddings
    for name in sorted(st_tensors.keys()):
        if 'code_predictor' in name and ('embed' in name or 'lm_head' in name or 'tok_' in name):
            key_tensors.append((name, 'Code predictor embedding/head'))
    
    lines.append("## 🎯 Key Tensors of Interest\n\n")
    lines.append("| Safetensor Name | ST Shape | ST Dtype | GGUF Name | GGUF Shape | Match |\n")
    lines.append("|-----------------|----------|----------|-----------|------------|-------|\n")
    
    for st_name, desc in key_tensors:
        if st_name not in st_tensors:
            continue
        st_info = st_tensors[st_name]
        
        gguf_name = st_to_gguf.get(st_name, None)
        if gguf_name and gguf_name in gguf_tensors:
            gguf_info = gguf_tensors[gguf_name]
            match = shapes_match(st_info['shape'], gguf_info['shape'])
            lines.append(f"| `{st_name}` | {st_info['shape']} | {st_info['dtype']} | `{gguf_name}` | {gguf_info['shape']} | {match} |\n")
        else:
            # Try to find by simple underscore
            simple_gguf = st_name.replace('.', '_')
            if simple_gguf in gguf_tensors:
                gguf_info = gguf_tensors[simple_gguf]
                match = shapes_match(st_info['shape'], gguf_info['shape'])
                lines.append(f"| `{st_name}` | {st_info['shape']} | {st_info['dtype']} | `{simple_gguf}` | {gguf_info['shape']} | {match} |\n")
            else:
                lines.append(f"| `{st_name}` | {st_info['shape']} | {st_info['dtype']} | ❌ NOT FOUND | - | ❌ MISSING |\n")
    
    # Analysis section
    lines.append("\n## 🔍 Embedding Analysis\n\n")
    
    for name in sorted(st_tensors.keys()):
        if 'embedding' in name.lower() or 'embed' in name.lower() or 'tok_emb' in name.lower():
            info = st_tensors[name]
            lines.append(f"### `{name}`\n")
            lines.append(f"- **Shape:** {info['shape']}\n")
            if len(info['shape']) == 2:
                lines.append(f"- **Vocab size:** {info['shape'][0]}\n")
                lines.append(f"- **Embedding dim:** {info['shape'][1]}\n")
            lines.append(f"- **Dtype:** {info['dtype']}\n")
            lines.append(f"- **Value range:** [{info['min']:.4f}, {info['max']:.4f}]\n")
            lines.append(f"- **Mean:** {info['mean']:.6f}, Std: {info['std']:.4f}\n")
            
            # Compare with GGUF
            gguf_name = st_to_gguf.get(name)
            if not gguf_name:
                gguf_name = name.replace('.', '_')
            if gguf_name in gguf_tensors:
                ginfo = gguf_tensors[gguf_name]
                lines.append(f"- **GGUF name:** `{gguf_name}`\n")
                lines.append(f"- **GGUF shape:** {ginfo['shape']} {'✅' if info['shape'] == ginfo['shape'] else '⚠️ DIFFERENT'}\n")
                lines.append(f"- **GGUF dtype:** {ginfo['dtype']}\n")
            lines.append("\n")
    
    # Shape comparison by category
    lines.append("## 📊 Full Comparison by Category\n\n")
    
    categories = {
        'talker.model.layers': [],
        'talker.code_predictor': [],
        'speaker_encoder': [],
        'other': [],
    }
    
    for st_name in sorted(st_tensors.keys()):
        categorized = False
        for cat in ['talker.model.layers', 'talker.code_predictor', 'speaker_encoder']:
            if st_name.startswith(cat):
                categories[cat].append(st_name)
                categorized = True
                break
        if not categorized:
            categories['other'].append(st_name)
    
    # Track issues
    transposed = []
    mismatched = []
    missing_in_gguf = []
    
    for cat, names in categories.items():
        if not names:
            continue
        lines.append(f"### {cat}\n\n")
        lines.append("| Safetensor | Shape | GGUF | Shape | Status |\n")
        lines.append("|------------|-------|------|-------|--------|\n")
        
        for st_name in names:
            st_info = st_tensors[st_name]
            
            # Find GGUF match
            gguf_name = st_to_gguf.get(st_name)
            if not gguf_name:
                gguf_name = st_name.replace('.', '_')
            
            if gguf_name in gguf_tensors:
                gguf_info = gguf_tensors[gguf_name]
                match = shapes_match(st_info['shape'], gguf_info['shape'])
                
                # Track issues
                if 'TRANSPOSED' in match:
                    transposed.append((st_name, st_info['shape'], gguf_name, gguf_info['shape']))
                elif 'MISMATCH' in match or 'rank' in match:
                    mismatched.append((st_name, st_info['shape'], gguf_name, gguf_info['shape']))
                
                short_st = st_name.replace(cat + '.', '') if st_name.startswith(cat) else st_name
                short_gguf = gguf_name
                lines.append(f"| {short_st} | {st_info['shape']} | {short_gguf} | {gguf_info['shape']} | {match} |\n")
            else:
                missing_in_gguf.append((st_name, st_info['shape']))
                short_st = st_name.replace(cat + '.', '') if st_name.startswith(cat) else st_name
                lines.append(f"| {short_st} | {st_info['shape']} | ❌ | - | MISSING |\n")
        
        lines.append("\n")
    
    # Summary of issues
    lines.append("## ⚠️ Issues Found\n\n")
    
    lines.append(f"### Transposed Tensors ({len(transposed)})\n\n")
    if transposed:
        lines.append("| Safetensor | ST Shape | GGUF | GGUF Shape |\n")
        lines.append("|------------|----------|------|------------|\n")
        for st_name, st_shape, gguf_name, gguf_shape in transposed:
            lines.append(f"| `{st_name}` | {st_shape} | `{gguf_name}` | {gguf_shape} |\n")
    else:
        lines.append("None found.\n")
    lines.append("\n")
    
    lines.append(f"### Shape Mismatches ({len(mismatched)})\n\n")
    if mismatched:
        lines.append("| Safetensor | ST Shape | GGUF | GGUF Shape |\n")
        lines.append("|------------|----------|------|------------|\n")
        for st_name, st_shape, gguf_name, gguf_shape in mismatched:
            lines.append(f"| `{st_name}` | {st_shape} | `{gguf_name}` | {gguf_shape} |\n")
    else:
        lines.append("None found.\n")
    lines.append("\n")
    
    lines.append(f"### Missing from GGUF ({len(missing_in_gguf)})\n\n")
    if missing_in_gguf:
        for st_name, st_shape in missing_in_gguf:
            lines.append(f"- `{st_name}` {st_shape}\n")
    else:
        lines.append("None found.\n")
    lines.append("\n")
    
    # GGUF-only tensors
    gguf_only = set(gguf_tensors.keys()) - set(gguf_to_st.keys()) - set(n.replace('.', '_') for n in st_tensors.keys())
    lines.append(f"### Extra in GGUF ({len(gguf_only)})\n\n")
    if gguf_only:
        for name in sorted(gguf_only):
            lines.append(f"- `{name}` {gguf_tensors[name]['shape']}\n")
    else:
        lines.append("None found.\n")
    
    # Value statistics summary
    lines.append("\n## 📈 Value Statistics\n\n")
    lines.append("| Tensor | Source | Min | Max | Mean | Std |\n")
    lines.append("|--------|--------|-----|-----|------|-----|\n")
    
    important_tensors = [
        'talker.model.text_embedding.weight',
        'talker.model.codec_embedding.weight',
        'talker.codec_head.weight',
    ]
    
    for name in important_tensors:
        if name in st_tensors:
            info = st_tensors[name]
            lines.append(f"| `{name}` | safetensors | {info['min']:.4f} | {info['max']:.4f} | {info['mean']:.6f} | {info['std']:.4f} |\n")
            
            gguf_name = st_to_gguf.get(name) or name.replace('.', '_')
            if gguf_name in gguf_tensors:
                ginfo = gguf_tensors[gguf_name]
                lines.append(f"| `{gguf_name}` | GGUF | {ginfo['min']:.4f} | {ginfo['max']:.4f} | {ginfo['mean']:.6f} | {ginfo['std']:.4f} |\n")
    
    with open(OUTPUT_PATH, 'w') as f:
        f.writelines(lines)
    
    print(f"Report written to: {OUTPUT_PATH}")
    return len(transposed), len(mismatched), len(missing_in_gguf), len(gguf_only)

def main():
    print("Loading safetensors...")
    st_tensors = load_safetensors_info()
    print(f"  Found {len(st_tensors)} tensors")
    
    print("Loading GGUF...")
    gguf_tensors = load_gguf_info()
    print(f"  Found {len(gguf_tensors)} tensors")
    
    print("Generating comparison report...")
    stats = generate_report(st_tensors, gguf_tensors)
    
    print(f"\n=== Issues Found ===")
    print(f"Transposed: {stats[0]}")
    print(f"Mismatched: {stats[1]}")
    print(f"Missing in GGUF: {stats[2]}")
    print(f"Extra in GGUF: {stats[3]}")

if __name__ == "__main__":
    main()
