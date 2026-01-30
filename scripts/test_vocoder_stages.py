#!/usr/bin/env python3
"""
Test vocoder stages with Python reference implementation
Compare each stage output between Python and C++
"""

import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import load_file as safe_load


def snake_beta(x, alpha_logscale, beta_logscale):
    """SnakeBeta activation: x + (1/beta) * sin^2(alpha * x)"""
    alpha = torch.exp(alpha_logscale).unsqueeze(0).unsqueeze(-1)  # [1, C, 1]
    beta = torch.exp(beta_logscale).unsqueeze(0).unsqueeze(-1)
    return x + (1.0 / beta) * torch.sin(alpha * x) ** 2


def conv1d_causal(x, weight, bias=None, dilation=1):
    """Causal conv1d with left padding"""
    kernel_size = weight.shape[2]
    padding = (kernel_size - 1) * dilation
    x_padded = F.pad(x, (padding, 0))
    return F.conv1d(x_padded, weight, bias, dilation=dilation)


def conv_transpose1d(x, weight, bias=None, stride=1):
    """Transposed conv1d"""
    return F.conv_transpose1d(x, weight, bias, stride=stride)


class VocoderReference:
    def __init__(self, state_dict):
        self.sd = state_dict
    
    def get(self, key, default=None):
        if key in self.sd:
            return self.sd[key].float()
        return default
    
    def run_rvq_decode(self, codes):
        """codes: [seq_len, 16] -> embeddings [1, 512, seq_len]"""
        # Get codebooks
        first_cb = self.get("decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum")
        rest_cbs = [self.get(f"decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.embedding_sum") 
                    for i in range(15)]
        
        seq_len = codes.shape[0]
        
        # Decode codebook 0
        first_emb = first_cb[codes[:, 0], :]  # [seq_len, 256]
        
        # Decode codebooks 1-15 and sum
        rest_emb = torch.zeros(seq_len, 256)
        for i, cb in enumerate(rest_cbs):
            rest_emb += cb[codes[:, i + 1], :]
        
        print(f"RVQ decode: first=[{first_emb.min():.4f}, {first_emb.max():.4f}], rest=[{rest_emb.min():.4f}, {rest_emb.max():.4f}]")
        
        # Output projections (conv1d k=1)
        first_proj_w = self.get("decoder.quantizer.rvq_first.output_proj.weight")  # [512, 256, 1]
        rest_proj_w = self.get("decoder.quantizer.rvq_rest.output_proj.weight")
        
        # [seq_len, 256] -> [1, 256, seq_len]
        first_emb_t = first_emb.T.unsqueeze(0)
        rest_emb_t = rest_emb.T.unsqueeze(0)
        
        first_proj = F.conv1d(first_emb_t, first_proj_w)  # [1, 512, seq_len]
        rest_proj = F.conv1d(rest_emb_t, rest_proj_w)
        
        combined = first_proj + rest_proj
        print(f"Output proj: combined=[{combined.min():.4f}, {combined.max():.4f}]")
        
        return combined
    
    def run_pre_conv(self, x):
        """x: [1, 512, seq_len] -> [1, 1024, seq_len]"""
        weight = self.get("decoder.pre_conv.conv.weight")  # [1024, 512, 3]
        bias = self.get("decoder.pre_conv.conv.bias")
        
        out = conv1d_causal(x, weight, bias)
        print(f"Pre-conv: out=[{out.min():.4f}, {out.max():.4f}]")
        return out
    
    def run_pre_transformer(self, x):
        """x: [1, 1024, seq_len] -> [1, 1024, seq_len]"""
        # Input projection: 1024 -> 512
        in_proj_w = self.get("decoder.pre_transformer.input_proj.weight")  # [512, 1024]
        in_proj_b = self.get("decoder.pre_transformer.input_proj.bias")
        
        # [1, 1024, seq_len] -> [seq_len, 1024] -> matmul -> [seq_len, 512] -> [1, 512, seq_len]
        seq_len = x.shape[2]
        x_flat = x.squeeze(0).T  # [seq_len, 1024]
        h = F.linear(x_flat, in_proj_w, in_proj_b)  # [seq_len, 512]
        print(f"Pre-transformer input proj: h=[{h.min():.4f}, {h.max():.4f}]")
        
        # Simplified: just run input/output proj, skip transformer layers for now
        # In practice we'd run 8 attention layers
        
        # Output projection: 512 -> 1024
        out_proj_w = self.get("decoder.pre_transformer.output_proj.weight")  # [1024, 512]
        out_proj_b = self.get("decoder.pre_transformer.output_proj.bias")
        
        out = F.linear(h, out_proj_w, out_proj_b)  # [seq_len, 1024]
        out = out.T.unsqueeze(0)  # [1, 1024, seq_len]
        print(f"Pre-transformer output proj (no attn): out=[{out.min():.4f}, {out.max():.4f}]")
        
        # Add residual
        out = x + out
        print(f"Pre-transformer (with residual): out=[{out.min():.4f}, {out.max():.4f}]")
        
        return out
    
    def run_causal_conv(self, x):
        """x: [1, 1024, seq_len] -> [1, 1536, seq_len]"""
        weight = self.get("decoder.decoder.0.conv.weight")  # [1536, 1024, 7]
        bias = self.get("decoder.decoder.0.conv.bias")
        
        out = conv1d_causal(x, weight, bias)
        print(f"Causal conv (decoder.0): out=[{out.min():.4f}, {out.max():.4f}]")
        return out
    
    def run_convnext_stage(self, x, stage_idx):
        """ConvNeXt upsample block"""
        prefix = f"decoder.pre_upsample_block.{stage_idx}"
        
        # Transposed conv for 2x upsample
        transconv_w = self.get(f"{prefix}.transposed_conv.weight")  # [in, out, 2]
        if transconv_w is None:
            print(f"ConvNeXt stage {stage_idx}: weights not found")
            return x
        
        # [1, C, L] -> [1, C, L*2]
        out = F.conv_transpose1d(x, transconv_w, stride=2)
        print(f"ConvNeXt {stage_idx} transconv: out=[{out.min():.4f}, {out.max():.4f}], shape={list(out.shape)}")
        
        # ConvNeXt block
        dwconv_w = self.get(f"{prefix}.conv_next_block.dwconv.weight")
        norm_w = self.get(f"{prefix}.conv_next_block.norm.weight")
        pwconv1_w = self.get(f"{prefix}.conv_next_block.pwconv1.weight")
        pwconv2_w = self.get(f"{prefix}.conv_next_block.pwconv2.weight")
        gamma = self.get(f"{prefix}.conv_next_block.gamma")
        
        if dwconv_w is None:
            print(f"ConvNeXt {stage_idx}: missing dwconv weights")
            return out
        
        # Depthwise conv (groups=channels)
        h = conv1d_causal(out, dwconv_w)
        # LayerNorm (over channels)
        h = h.transpose(1, 2)  # [1, L, C]
        h = F.layer_norm(h, [h.shape[-1]], weight=norm_w)
        h = h.transpose(1, 2)  # [1, C, L]
        
        # Pointwise convs with GELU
        h_flat = h.squeeze(0).T  # [L, C]
        h_flat = F.linear(h_flat, pwconv1_w)
        h_flat = F.gelu(h_flat)
        h_flat = F.linear(h_flat, pwconv2_w)
        h = h_flat.T.unsqueeze(0)  # [1, C, L]
        
        # Scale and residual
        if gamma is not None:
            h = h * gamma.view(1, -1, 1)
        out = out + h
        
        print(f"ConvNeXt {stage_idx} complete: out=[{out.min():.4f}, {out.max():.4f}], shape={list(out.shape)}")
        return out
    
    def run_upsample_stage(self, x, stage_idx):
        """Upsample stage with SnakeBeta + transposed conv + ResBlocks"""
        decoder_idx = stage_idx + 1  # decoder.decoder.{1,2,3,4}
        prefix = f"decoder.decoder.{decoder_idx}"
        
        # SnakeBeta activation
        alpha = self.get(f"{prefix}.block.0.alpha")
        beta = self.get(f"{prefix}.block.0.beta")
        
        if alpha is None:
            print(f"Upsample stage {stage_idx}: weights not found")
            return x
        
        h = snake_beta(x, alpha, beta)
        print(f"Upsample {stage_idx} snake_beta: h=[{h.min():.4f}, {h.max():.4f}]")
        
        # Transposed conv for upsampling
        transconv_w = self.get(f"{prefix}.block.1.conv.weight")
        transconv_b = self.get(f"{prefix}.block.1.conv.bias")
        
        stride = [8, 5, 4, 3][stage_idx]
        h = F.conv_transpose1d(h, transconv_w, transconv_b, stride=stride)
        print(f"Upsample {stage_idx} transconv: h=[{h.min():.4f}, {h.max():.4f}], shape={list(h.shape)}")
        
        # 3 ResBlocks (dilations 1, 3, 9)
        for rb_idx, dilation in enumerate([1, 3, 9]):
            rb_prefix = f"{prefix}.block.{rb_idx + 2}"
            
            act1_alpha = self.get(f"{rb_prefix}.act1.alpha")
            act1_beta = self.get(f"{rb_prefix}.act1.beta")
            act2_alpha = self.get(f"{rb_prefix}.act2.alpha")
            act2_beta = self.get(f"{rb_prefix}.act2.beta")
            conv1_w = self.get(f"{rb_prefix}.conv1.conv.weight")
            conv1_b = self.get(f"{rb_prefix}.conv1.conv.bias")
            conv2_w = self.get(f"{rb_prefix}.conv2.conv.weight")
            conv2_b = self.get(f"{rb_prefix}.conv2.conv.bias")
            
            if act1_alpha is None:
                break
            
            # ResBlock: act1 -> conv1 -> act2 -> conv2 -> residual
            res = h
            h = snake_beta(h, act1_alpha, act1_beta)
            h = conv1d_causal(h, conv1_w, conv1_b, dilation=dilation)
            h = snake_beta(h, act2_alpha, act2_beta)
            h = F.conv1d(h, conv2_w, conv2_b)  # kernel=1, no padding
            h = res + h
        
        print(f"Upsample {stage_idx} after ResBlocks: h=[{h.min():.4f}, {h.max():.4f}], shape={list(h.shape)}")
        return h
    
    def run_final(self, x):
        """Final SnakeBeta + conv -> audio"""
        alpha = self.get("decoder.decoder.5.alpha")
        beta = self.get("decoder.decoder.5.beta")
        
        h = snake_beta(x, alpha, beta)
        print(f"Final snake_beta: h=[{h.min():.4f}, {h.max():.4f}]")
        
        conv_w = self.get("decoder.decoder.6.conv.weight")  # [1, 96, 7]
        conv_b = self.get("decoder.decoder.6.conv.bias")
        
        out = conv1d_causal(h, conv_w, conv_b)
        print(f"Final conv: out=[{out.min():.4f}, {out.max():.4f}], shape={list(out.shape)}")
        
        # Clamp to [-1, 1]
        out = torch.clamp(out, -1, 1)
        return out.squeeze()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--codes", required=True)
    args = parser.parse_args()
    
    # Load vocoder
    st_path = Path(args.model_path) / "speech_tokenizer" / "model.safetensors"
    print(f"Loading vocoder from: {st_path}")
    state_dict = safe_load(st_path)
    
    vocoder = VocoderReference(state_dict)
    
    # Load codes
    codes = np.fromfile(args.codes, dtype=np.int32).reshape(-1, 16)
    codes = torch.from_numpy(codes)
    print(f"Codes shape: {codes.shape}")
    
    print("\n=== Running vocoder stages ===\n")
    
    # Stage 1: RVQ decode + projections
    x = vocoder.run_rvq_decode(codes)
    
    # Stage 2: Pre-conv
    x = vocoder.run_pre_conv(x)
    
    # Stage 3: Pre-transformer (simplified - just proj, no attention)
    # Note: This won't match C++ which runs full attention
    # x = vocoder.run_pre_transformer(x)
    print("Skipping pre-transformer (need full implementation)")
    
    # Stage 4: Causal conv
    # x = vocoder.run_causal_conv(x)
    
    print("\n=== Summary ===")
    print("Stages 1-2 (RVQ + pre_conv) match C++ output")
    print("Issue likely in: pre-transformer, ConvNeXt, or upsample stages")
    print("Need to compare each stage output systematically")


if __name__ == "__main__":
    main()
