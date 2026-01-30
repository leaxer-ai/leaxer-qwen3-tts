#!/usr/bin/env python3
"""
Vocoder Isolation Test - Comprehensive stage-by-stage analysis

This script:
1. Loads the vocoder weights from safetensors directly (no ONNX needed)
2. Runs the Python vocoder forward pass with full implementation
3. Saves intermediate tensors at each stage for comparison with C++
4. Generates test audio from debug_codes.bin

Key stages verified:
- RVQ decode (codebook lookup)
- Output projections (256 → 512)
- Pre-conv (512 → 1024)
- Pre-transformer (8 layers with attention)
- ConvNeXt upsample (2x 2-fold)
- Causal conv (1024 → 1536)
- Upsample stages (4 stages with SnakeBeta + ResBlocks)
- Final conv (96 → 1)

Usage:
    python scripts/vocoder_test.py --model-path models/Qwen3-TTS-12Hz-0.6B-Base --codes debug_codes.bin
"""

import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import load_file as safe_load
import soundfile as sf
from dataclasses import dataclass


# ============================================================================
# Helper Functions
# ============================================================================

def snake_beta(x, alpha_logscale, beta_logscale):
    """SnakeBeta activation: x + (1/beta) * sin^2(alpha * x)
    
    x: [batch, channels, seq] or [seq, channels]
    alpha/beta: [channels]
    """
    if x.dim() == 3:
        alpha = torch.exp(alpha_logscale).unsqueeze(0).unsqueeze(-1)  # [1, C, 1]
        beta = torch.exp(beta_logscale).unsqueeze(0).unsqueeze(-1)
    else:  # [seq, channels]
        alpha = torch.exp(alpha_logscale).unsqueeze(0)  # [1, C]
        beta = torch.exp(beta_logscale).unsqueeze(0)
    return x + (1.0 / beta) * torch.sin(alpha * x) ** 2


def causal_conv1d(x, weight, bias=None, dilation=1):
    """Causal conv1d with left padding
    
    x: [batch, in_channels, seq]
    weight: [out_channels, in_channels, kernel_size]
    """
    kernel_size = weight.shape[2]
    padding = (kernel_size - 1) * dilation
    x_padded = F.pad(x, (padding, 0))
    return F.conv1d(x_padded, weight, bias, dilation=dilation)


def rms_norm(x, weight, eps=1e-6):
    """RMS normalization
    
    x: [seq, dim]
    weight: [dim]
    """
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return (x / rms) * weight


def save_tensor_debug(tensor, name, output_dir):
    """Save tensor for debugging"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    np.save(output_dir / f"{name}.npy", tensor.detach().cpu().numpy())


def print_range(name, tensor):
    """Print tensor range for debugging"""
    t = tensor.float()
    print(f"  {name}: [{t.min().item():.6f}, {t.max().item():.6f}], shape={list(t.shape)}")


# ============================================================================
# Vocoder Components
# ============================================================================

class PreTransformerLayer:
    """Single pre-transformer layer with attention and FFN
    
    Architecture from weights:
    - Q/K/V projections: [1024, 512] meaning hidden=512, qkv_dim=1024
    - Using 16 heads with head_dim=64: 16*64=1024 for QKV
    - O projection: [512, 1024] back to hidden_dim
    - FFN: gate/up [1024, 512], down [512, 1024]
    """
    
    def __init__(self, sd, layer_idx, hidden_dim=512, intermediate_dim=1024, num_heads=16):
        prefix = f"decoder.pre_transformer.layers.{layer_idx}"
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        # QKV dimension is 1024, divided among 16 heads = 64 per head
        self.head_dim = 64  # Fixed based on weight shapes
        
        # Load weights
        self.input_ln_weight = sd[f"{prefix}.input_layernorm.weight"].float()
        self.post_ln_weight = sd[f"{prefix}.post_attention_layernorm.weight"].float()
        
        self.q_weight = sd[f"{prefix}.self_attn.q_proj.weight"].float()
        self.k_weight = sd[f"{prefix}.self_attn.k_proj.weight"].float()
        self.v_weight = sd[f"{prefix}.self_attn.v_proj.weight"].float()
        self.o_weight = sd[f"{prefix}.self_attn.o_proj.weight"].float()
        
        self.gate_weight = sd[f"{prefix}.mlp.gate_proj.weight"].float()
        self.up_weight = sd[f"{prefix}.mlp.up_proj.weight"].float()
        self.down_weight = sd[f"{prefix}.mlp.down_proj.weight"].float()
        
        # Layer scales (residual scaling) - note different key names
        self.attn_scale = sd.get(f"{prefix}.self_attn_layer_scale.scale")
        self.ffn_scale = sd.get(f"{prefix}.mlp_layer_scale.scale")
        if self.attn_scale is not None:
            self.attn_scale = self.attn_scale.float()
        if self.ffn_scale is not None:
            self.ffn_scale = self.ffn_scale.float()
    
    def forward(self, x):
        """
        x: [seq_len, hidden_dim] where hidden_dim=512
        Returns: [seq_len, hidden_dim]
        """
        # Pre-attention RMS norm
        h = rms_norm(x, self.input_ln_weight)
        
        # Self-attention
        # Q/K/V: [seq, 512] -> [seq, 1024] (16 heads * 64 head_dim)
        q = F.linear(h, self.q_weight)
        k = F.linear(h, self.k_weight)
        v = F.linear(h, self.v_weight)
        
        seq_len = x.shape[0]
        # Reshape for multi-head attention: [seq, num_heads, head_dim]
        q = q.view(seq_len, self.num_heads, self.head_dim)
        k = k.view(seq_len, self.num_heads, self.head_dim)
        v = v.view(seq_len, self.num_heads, self.head_dim)
        
        # Scaled dot-product attention
        scale = 1.0 / (self.head_dim ** 0.5)
        attn = torch.einsum('qhd,khd->hqk', q, k) * scale  # [heads, seq, seq]
        attn = F.softmax(attn, dim=-1)
        attn_out = torch.einsum('hqk,khd->qhd', attn, v)  # [seq, heads, head_dim]
        attn_out = attn_out.reshape(seq_len, -1)  # [seq, 1024]
        
        # Output projection: [seq, 1024] -> [seq, 512]
        attn_out = F.linear(attn_out, self.o_weight)
        
        # Apply attention layer scale and residual
        if self.attn_scale is not None:
            attn_out = attn_out * self.attn_scale
        x = x + attn_out
        
        # Pre-FFN RMS norm
        h = rms_norm(x, self.post_ln_weight)
        
        # SwiGLU FFN: down(silu(gate) * up)
        gate = F.linear(h, self.gate_weight)
        up = F.linear(h, self.up_weight)
        ffn_out = F.linear(F.silu(gate) * up, self.down_weight)
        
        # Apply FFN layer scale and residual
        if self.ffn_scale is not None:
            ffn_out = ffn_out * self.ffn_scale
        x = x + ffn_out
        
        return x


class ConvNeXtBlock:
    """ConvNeXt block: dwconv -> norm -> pwconv1 -> gelu -> pwconv2 -> gamma -> residual"""
    
    def __init__(self, sd, prefix, channels):
        self.channels = channels
        
        self.dwconv_weight = sd.get(f"{prefix}.dwconv.weight")
        self.dwconv_bias = sd.get(f"{prefix}.dwconv.bias")
        self.norm_weight = sd.get(f"{prefix}.norm.weight")
        self.norm_bias = sd.get(f"{prefix}.norm.bias")
        self.pwconv1_weight = sd.get(f"{prefix}.pwconv1.weight")
        self.pwconv1_bias = sd.get(f"{prefix}.pwconv1.bias")
        self.pwconv2_weight = sd.get(f"{prefix}.pwconv2.weight")
        self.pwconv2_bias = sd.get(f"{prefix}.pwconv2.bias")
        self.gamma = sd.get(f"{prefix}.gamma")
        
        # Convert to float
        for attr in ['dwconv_weight', 'dwconv_bias', 'norm_weight', 'norm_bias',
                     'pwconv1_weight', 'pwconv1_bias', 'pwconv2_weight', 'pwconv2_bias', 'gamma']:
            v = getattr(self, attr)
            if v is not None:
                setattr(self, attr, v.float())
    
    def forward(self, x):
        """
        x: [batch, channels, seq]
        Returns: [batch, channels, seq]
        """
        if self.dwconv_weight is None:
            return x
        
        residual = x
        
        # Depthwise causal conv (groups=channels)
        h = causal_conv1d(x, self.dwconv_weight, self.dwconv_bias)
        
        # Permute to [batch, seq, channels] for LayerNorm
        h = h.transpose(1, 2)
        h = F.layer_norm(h, [self.channels], self.norm_weight, self.norm_bias)
        
        # Pointwise convs with GELU
        h = F.linear(h, self.pwconv1_weight, self.pwconv1_bias)
        h = F.gelu(h)
        h = F.linear(h, self.pwconv2_weight, self.pwconv2_bias)
        
        # Apply gamma (LayerScale)
        if self.gamma is not None:
            h = h * self.gamma
        
        # Permute back to [batch, channels, seq]
        h = h.transpose(1, 2)
        
        return residual + h


class ResBlock:
    """ResBlock: act1 -> conv1 -> act2 -> conv2 -> residual"""
    
    def __init__(self, sd, prefix, channels, kernel_size=7, dilation=1):
        self.channels = channels
        self.dilation = dilation
        
        self.act1_alpha = sd.get(f"{prefix}.act1.alpha")
        self.act1_beta = sd.get(f"{prefix}.act1.beta")
        self.conv1_weight = sd.get(f"{prefix}.conv1.conv.weight")
        self.conv1_bias = sd.get(f"{prefix}.conv1.conv.bias")
        self.act2_alpha = sd.get(f"{prefix}.act2.alpha")
        self.act2_beta = sd.get(f"{prefix}.act2.beta")
        self.conv2_weight = sd.get(f"{prefix}.conv2.conv.weight")
        self.conv2_bias = sd.get(f"{prefix}.conv2.conv.bias")
        
        # Convert to float
        for attr in ['act1_alpha', 'act1_beta', 'conv1_weight', 'conv1_bias',
                     'act2_alpha', 'act2_beta', 'conv2_weight', 'conv2_bias']:
            v = getattr(self, attr)
            if v is not None:
                setattr(self, attr, v.float())
    
    def forward(self, x):
        """
        x: [batch, channels, seq]
        Returns: [batch, channels, seq]
        """
        if self.act1_alpha is None:
            return x
        
        residual = x
        
        # act1 -> conv1
        h = snake_beta(x, self.act1_alpha, self.act1_beta)
        h = causal_conv1d(h, self.conv1_weight, self.conv1_bias, dilation=self.dilation)
        
        # act2 -> conv2 (kernel=1)
        h = snake_beta(h, self.act2_alpha, self.act2_beta)
        h = F.conv1d(h, self.conv2_weight, self.conv2_bias)
        
        return residual + h


# ============================================================================
# Full Vocoder
# ============================================================================

class VocoderFull:
    """Complete Qwen3-TTS Vocoder implementation"""
    
    def __init__(self, model_path):
        st_path = Path(model_path) / "speech_tokenizer" / "model.safetensors"
        print(f"Loading vocoder from: {st_path}")
        self.sd = safe_load(st_path)
        
        # Load codebooks
        self.first_cb = self.sd["decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"].float()
        self.rest_cbs = [self.sd[f"decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.embedding_sum"].float()
                         for i in range(15)]
        
        # Output projections
        self.first_proj_w = self.sd["decoder.quantizer.rvq_first.output_proj.weight"].float()
        self.rest_proj_w = self.sd["decoder.quantizer.rvq_rest.output_proj.weight"].float()
        
        # Pre-conv
        self.pre_conv_w = self.sd["decoder.pre_conv.conv.weight"].float()
        self.pre_conv_b = self.sd["decoder.pre_conv.conv.bias"].float()
        
        # Pre-transformer
        self.pre_trans_in_w = self.sd["decoder.pre_transformer.input_proj.weight"].float()
        self.pre_trans_in_b = self.sd["decoder.pre_transformer.input_proj.bias"].float()
        self.pre_trans_out_w = self.sd["decoder.pre_transformer.output_proj.weight"].float()
        self.pre_trans_out_b = self.sd["decoder.pre_transformer.output_proj.bias"].float()
        
        # Pre-transformer layers
        self.pre_transformer_layers = [PreTransformerLayer(self.sd, i) for i in range(8)]
        
        # ConvNeXt upsample blocks
        self.convnext_transconv = [
            (self.sd["decoder.upsample.0.0.conv.weight"].float(),
             self.sd["decoder.upsample.0.0.conv.bias"].float()),
            (self.sd["decoder.upsample.1.0.conv.weight"].float(),
             self.sd["decoder.upsample.1.0.conv.bias"].float()),
        ]
        self.convnext_blocks = [
            ConvNeXtBlock(self.sd, "decoder.upsample.0.1", 1024),
            ConvNeXtBlock(self.sd, "decoder.upsample.1.1", 1024),
        ]
        
        # Causal conv (decoder.0)
        self.causal_conv_w = self.sd["decoder.decoder.0.conv.weight"].float()
        self.causal_conv_b = self.sd["decoder.decoder.0.conv.bias"].float()
        
        # Upsample stages (decoder.1-4)
        self.upsample_stages = []
        upsample_channels = [1536, 768, 384, 192]  # Input channels
        out_channels = [768, 384, 192, 96]
        for stage in range(4):
            prefix = f"decoder.decoder.{stage + 1}"
            self.upsample_stages.append({
                'alpha': self.sd[f"{prefix}.block.0.alpha"].float(),
                'beta': self.sd[f"{prefix}.block.0.beta"].float(),
                'transconv_w': self.sd[f"{prefix}.block.1.conv.weight"].float(),
                'transconv_b': self.sd[f"{prefix}.block.1.conv.bias"].float(),
                'resblocks': [
                    ResBlock(self.sd, f"{prefix}.block.{2+rb}", out_channels[stage], 7, d)
                    for rb, d in enumerate([1, 3, 9])
                ],
            })
        
        # Final
        self.final_alpha = self.sd["decoder.decoder.5.alpha"].float()
        self.final_beta = self.sd["decoder.decoder.5.beta"].float()
        self.final_conv_w = self.sd["decoder.decoder.6.conv.weight"].float()
        self.final_conv_b = self.sd["decoder.decoder.6.conv.bias"].float()
        
        print("Vocoder weights loaded!")
    
    def forward(self, codes, save_intermediates=False, output_dir=None):
        """
        codes: [seq_len, 16] int tensor
        Returns: audio [samples]
        """
        seq_len = codes.shape[0]
        intermediates = {}
        
        # ============================================================
        # Stage 1: RVQ decode
        # ============================================================
        print("\n=== Stage 1: RVQ Decode ===")
        
        # Codebook 0 (semantic)
        first_emb = self.first_cb[codes[:, 0], :]  # [seq_len, 256]
        
        # Codebooks 1-15 (acoustic) - sum
        rest_emb = torch.zeros(seq_len, 256)
        for i, cb in enumerate(self.rest_cbs):
            rest_emb += cb[codes[:, i + 1], :]
        
        print_range("first_emb (codebook 0)", first_emb)
        print_range("rest_emb (codebooks 1-15 sum)", rest_emb)
        
        if save_intermediates:
            intermediates['rvq_first_emb'] = first_emb.clone()
            intermediates['rvq_rest_emb'] = rest_emb.clone()
        
        # ============================================================
        # Stage 2: Output projections
        # ============================================================
        print("\n=== Stage 2: Output Projections ===")
        
        # [seq_len, 256] -> [1, 256, seq_len]
        first_emb_t = first_emb.T.unsqueeze(0)
        rest_emb_t = rest_emb.T.unsqueeze(0)
        
        first_proj = F.conv1d(first_emb_t, self.first_proj_w)  # [1, 512, seq_len]
        rest_proj = F.conv1d(rest_emb_t, self.rest_proj_w)
        
        combined = first_proj + rest_proj  # ADD, not concat!
        
        print_range("first_proj", first_proj)
        print_range("rest_proj", rest_proj)
        print_range("combined (ADD)", combined)
        
        if save_intermediates:
            intermediates['proj_first'] = first_proj.clone()
            intermediates['proj_rest'] = rest_proj.clone()
            intermediates['proj_combined'] = combined.clone()
        
        # ============================================================
        # Stage 3: Pre-conv (512 → 1024)
        # ============================================================
        print("\n=== Stage 3: Pre-conv ===")
        
        pre_conv_out = causal_conv1d(combined, self.pre_conv_w, self.pre_conv_b)
        
        print_range("pre_conv_out", pre_conv_out)
        
        if save_intermediates:
            intermediates['pre_conv_out'] = pre_conv_out.clone()
        
        # ============================================================
        # Stage 4: Pre-transformer
        # ============================================================
        print("\n=== Stage 4: Pre-transformer ===")
        
        # Input projection: 1024 -> 512
        # [1, 1024, seq] -> [seq, 1024]
        x = pre_conv_out.squeeze(0).T
        h = F.linear(x, self.pre_trans_in_w, self.pre_trans_in_b)  # [seq, 512]
        
        print_range("pre_trans_input_proj", h)
        
        if save_intermediates:
            intermediates['pre_trans_in'] = h.clone()
        
        # 8 transformer layers
        for i, layer in enumerate(self.pre_transformer_layers):
            h = layer.forward(h)
            if i == 0 or i == 7:
                print_range(f"pre_trans_layer_{i}", h)
        
        if save_intermediates:
            intermediates['pre_trans_layers_out'] = h.clone()
        
        # Output projection: 512 -> 1024
        out = F.linear(h, self.pre_trans_out_w, self.pre_trans_out_b)
        out = out.T.unsqueeze(0)  # [1, 1024, seq]
        
        # Residual connection
        post_trans = pre_conv_out + out
        
        print_range("pre_trans_output_proj", out)
        print_range("post_trans (with residual)", post_trans)
        
        if save_intermediates:
            intermediates['pre_trans_out_proj'] = out.clone()
            intermediates['post_trans'] = post_trans.clone()
        
        # ============================================================
        # Stage 5: ConvNeXt Upsample (2 stages, each 2x)
        # ============================================================
        print("\n=== Stage 5: ConvNeXt Upsample ===")
        
        x = post_trans
        for stage, ((tc_w, tc_b), cn_block) in enumerate(zip(self.convnext_transconv, self.convnext_blocks)):
            # Transposed conv for 2x upsample
            # PyTorch ConvTranspose1d with causal trimming
            kernel_size = tc_w.shape[2]
            stride = 2
            
            raw_out = F.conv_transpose1d(x, tc_w, tc_b, stride=stride)
            # Causal trim: remove (kernel_size - stride) from start
            left_pad = kernel_size - stride
            x = raw_out[:, :, left_pad:]
            
            print_range(f"ConvNeXt {stage} transconv", x)
            
            # ConvNeXt block
            x = cn_block.forward(x)
            
            print_range(f"ConvNeXt {stage} complete", x)
        
        if save_intermediates:
            intermediates['convnext_out'] = x.clone()
        
        # ============================================================
        # Stage 6: Causal conv (1024 → 1536)
        # ============================================================
        print("\n=== Stage 6: Causal Conv ===")
        
        x = causal_conv1d(x, self.causal_conv_w, self.causal_conv_b)
        
        print_range("causal_conv_out", x)
        
        if save_intermediates:
            intermediates['causal_out'] = x.clone()
        
        # ============================================================
        # Stage 7: Upsample stages (4 stages)
        # ============================================================
        print("\n=== Stage 7: Upsample Stages ===")
        
        strides = [8, 5, 4, 3]
        for stage, (config, stride) in enumerate(zip(self.upsample_stages, strides)):
            # SnakeBeta activation
            x = snake_beta(x, config['alpha'], config['beta'])
            print_range(f"Upsample {stage} snake_beta", x)
            
            # Transposed conv for upsampling
            tc_w = config['transconv_w']
            tc_b = config['transconv_b']
            kernel_size = tc_w.shape[2]
            
            raw_out = F.conv_transpose1d(x, tc_w, tc_b, stride=stride)
            left_pad = kernel_size - stride
            x = raw_out[:, :, left_pad:]
            
            print_range(f"Upsample {stage} transconv", x)
            
            # 3 ResBlocks with dilations 1, 3, 9
            for rb in config['resblocks']:
                x = rb.forward(x)
            
            print_range(f"Upsample {stage} after ResBlocks", x)
        
        if save_intermediates:
            intermediates['upsample_out'] = x.clone()
        
        # ============================================================
        # Stage 8: Final
        # ============================================================
        print("\n=== Stage 8: Final ===")
        
        x = snake_beta(x, self.final_alpha, self.final_beta)
        print_range("final_snake_beta", x)
        
        x = causal_conv1d(x, self.final_conv_w, self.final_conv_b)
        print_range("final_conv (before clamp)", x)
        
        # Clamp to [-1, 1]
        audio = torch.clamp(x, -1, 1)
        print_range("audio (after clamp)", audio)
        
        if save_intermediates:
            intermediates['audio'] = audio.clone()
        
        if save_intermediates and output_dir:
            print(f"\nSaving intermediates to {output_dir}...")
            for name, tensor in intermediates.items():
                save_tensor_debug(tensor, name, output_dir)
        
        return audio.squeeze()


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Vocoder isolation test")
    parser.add_argument("--model-path", required=True, help="Path to Qwen3-TTS model")
    parser.add_argument("--codes", default="debug_codes.bin", help="Path to codes file")
    parser.add_argument("--output", default="vocoder_test_output.wav", help="Output audio file")
    parser.add_argument("--save-intermediates", action="store_true", help="Save intermediate tensors")
    parser.add_argument("--output-dir", default="vocoder_debug", help="Directory for intermediate tensors")
    args = parser.parse_args()
    
    # Load codes
    codes_path = Path(args.codes)
    if not codes_path.exists():
        print(f"Error: codes file not found: {codes_path}")
        sys.exit(1)
    
    codes = np.fromfile(codes_path, dtype=np.int32).reshape(-1, 16)
    codes = torch.from_numpy(codes)
    print(f"Loaded codes: shape={codes.shape}, range=[{codes.min()}, {codes.max()}]")
    print(f"First row: {codes[0].tolist()}")
    
    # Create vocoder
    vocoder = VocoderFull(args.model_path)
    
    # Run inference
    print("\n" + "="*60)
    print("Running vocoder forward pass...")
    print("="*60)
    
    with torch.no_grad():
        audio = vocoder.forward(
            codes, 
            save_intermediates=args.save_intermediates,
            output_dir=args.output_dir
        )
    
    # Save audio
    audio_np = audio.squeeze().numpy()
    sf.write(args.output, audio_np, 24000)
    print(f"\nSaved audio to {args.output}")
    print(f"Audio: {len(audio_np)} samples, {len(audio_np)/24000:.2f}s @ 24kHz")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Input codes: {codes.shape[0]} frames @ 12Hz = {codes.shape[0]/12:.2f}s")
    print(f"Output audio: {len(audio_np)} samples @ 24kHz = {len(audio_np)/24000:.2f}s")
    print(f"Upsample factor: {len(audio_np) / codes.shape[0]:.1f}x")


if __name__ == "__main__":
    main()
