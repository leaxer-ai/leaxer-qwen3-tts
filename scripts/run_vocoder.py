#!/usr/bin/env python3
"""
Minimal Python vocoder implementation for Qwen3-TTS
Generates audio from codec codes for comparison with C++
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import load_file as safe_load
import soundfile as sf


class SnakeBeta(nn.Module):
    """SnakeBeta activation: x + (1/beta) * sin^2(alpha * x)"""
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))
    
    def forward(self, x):
        # x: [B, C, T]
        alpha = self.alpha.exp().view(1, -1, 1)
        beta = self.beta.exp().view(1, -1, 1)
        return x + (1.0 / beta) * torch.sin(alpha * x) ** 2


class CausalConv1d(nn.Module):
    """Causal 1D convolution with left padding"""
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)
    
    def forward(self, x):
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class ResBlock(nn.Module):
    """ResBlock: act1 -> conv1 -> act2 -> conv2 -> residual"""
    def __init__(self, channels, kernel_size=7, dilation=1):
        super().__init__()
        self.act1 = SnakeBeta(channels)
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.act2 = SnakeBeta(channels)
        self.conv2 = nn.Conv1d(channels, channels, 1)
    
    def forward(self, x):
        h = self.act1(x)
        h = self.conv1(h)
        h = self.act2(h)
        h = self.conv2(h)
        return x + h


class Vocoder(nn.Module):
    """Qwen3-TTS Vocoder (decoder)"""
    def __init__(self):
        super().__init__()
        
        # RVQ codebooks [16, 2048, 256]
        self.register_buffer('codebooks', torch.zeros(16, 2048, 256))
        
        # Output projections (conv1d k=1)
        self.first_proj = nn.Conv1d(256, 512, 1)
        self.rest_proj = nn.Conv1d(256, 512, 1)
        
        # Pre-conv: 512 -> 1024
        self.pre_conv = CausalConv1d(512, 1024, 3)
        
        # Pre-transformer (simplified - just input/output proj)
        self.pre_trans_in = nn.Linear(1024, 512)
        self.pre_trans_out = nn.Linear(512, 1024)
        # Note: skipping 8 transformer layers for now
        
        # ConvNeXt upsample blocks (2 stages, each 2x)
        self.convnext_0_trans = nn.ConvTranspose1d(1024, 1024, 2, stride=2)
        self.convnext_1_trans = nn.ConvTranspose1d(1024, 1024, 2, stride=2)
        
        # Causal conv: 1024 -> 1536
        self.causal_conv = CausalConv1d(1024, 1536, 7)
        
        # Upsample stages
        self.upsample_acts = nn.ModuleList([
            SnakeBeta(1536), SnakeBeta(768), SnakeBeta(384), SnakeBeta(192)
        ])
        self.upsample_convs = nn.ModuleList([
            nn.ConvTranspose1d(1536, 768, 16, stride=8),
            nn.ConvTranspose1d(768, 384, 10, stride=5),
            nn.ConvTranspose1d(384, 192, 8, stride=4),
            nn.ConvTranspose1d(192, 96, 6, stride=3),
        ])
        self.upsample_resblocks = nn.ModuleList([
            nn.ModuleList([ResBlock(768, 7, d) for d in [1, 3, 9]]),
            nn.ModuleList([ResBlock(384, 7, d) for d in [1, 3, 9]]),
            nn.ModuleList([ResBlock(192, 7, d) for d in [1, 3, 9]]),
            nn.ModuleList([ResBlock(96, 7, d) for d in [1, 3, 9]]),
        ])
        
        # Final
        self.final_act = SnakeBeta(96)
        self.final_conv = CausalConv1d(96, 1, 7)
    
    def forward(self, codes):
        """
        codes: [B, T, 16] int tensor
        Returns: [B, T * 1920] audio
        """
        B, T, _ = codes.shape
        
        # RVQ decode
        first_emb = self.codebooks[0, codes[:, :, 0], :]  # [B, T, 256]
        rest_emb = torch.zeros(B, T, 256, device=codes.device)
        for i in range(1, 16):
            rest_emb += self.codebooks[i, codes[:, :, i], :]
        
        # Output projections
        first_emb = first_emb.transpose(1, 2)  # [B, 256, T]
        rest_emb = rest_emb.transpose(1, 2)
        first_proj = self.first_proj(first_emb)  # [B, 512, T]
        rest_proj = self.rest_proj(rest_emb)
        x = first_proj + rest_proj  # [B, 512, T]
        
        print(f"RVQ+proj: [{x.min():.4f}, {x.max():.4f}]")
        
        # Pre-conv
        x = self.pre_conv(x)  # [B, 1024, T]
        print(f"Pre-conv: [{x.min():.4f}, {x.max():.4f}]")
        
        # Pre-transformer (simplified)
        x_t = x.transpose(1, 2)  # [B, T, 1024]
        h = self.pre_trans_in(x_t)  # [B, T, 512]
        h = self.pre_trans_out(h)  # [B, T, 1024]
        x = x + h.transpose(1, 2)  # residual
        print(f"Pre-trans: [{x.min():.4f}, {x.max():.4f}]")
        
        # ConvNeXt upsamples (simplified - just transposed conv, no ConvNeXt block)
        x = self.convnext_0_trans(x)
        print(f"ConvNeXt 0: [{x.min():.4f}, {x.max():.4f}], shape={list(x.shape)}")
        x = self.convnext_1_trans(x)
        print(f"ConvNeXt 1: [{x.min():.4f}, {x.max():.4f}], shape={list(x.shape)}")
        
        # Causal conv
        x = self.causal_conv(x)  # [B, 1536, T*4]
        print(f"Causal: [{x.min():.4f}, {x.max():.4f}]")
        
        # Upsample stages
        for i, (act, conv, rbs) in enumerate(zip(
            self.upsample_acts, self.upsample_convs, self.upsample_resblocks
        )):
            x = act(x)
            x = conv(x)
            for rb in rbs:
                x = rb(x)
            print(f"Upsample {i}: [{x.min():.4f}, {x.max():.4f}], shape={list(x.shape)}")
        
        # Final
        x = self.final_act(x)
        x = self.final_conv(x)
        x = torch.tanh(x)  # Clamp to [-1, 1]
        
        return x.squeeze(1)  # [B, audio_len]


def load_weights(vocoder, model_path):
    """Load weights from safetensors"""
    st_path = Path(model_path) / "speech_tokenizer" / "model.safetensors"
    sd = safe_load(st_path)
    
    # Codebooks
    first_cb = sd["decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"]
    rest_cbs = [sd[f"decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.embedding_sum"] 
                for i in range(15)]
    codebooks = torch.stack([first_cb.float()] + [cb.float() for cb in rest_cbs])
    vocoder.codebooks.copy_(codebooks)
    
    # Projections
    vocoder.first_proj.weight.data.copy_(sd["decoder.quantizer.rvq_first.output_proj.weight"].float())
    vocoder.rest_proj.weight.data.copy_(sd["decoder.quantizer.rvq_rest.output_proj.weight"].float())
    
    # Pre-conv
    vocoder.pre_conv.conv.weight.data.copy_(sd["decoder.pre_conv.conv.weight"].float())
    vocoder.pre_conv.conv.bias.data.copy_(sd["decoder.pre_conv.conv.bias"].float())
    
    # Pre-transformer
    vocoder.pre_trans_in.weight.data.copy_(sd["decoder.pre_transformer.input_proj.weight"].float())
    vocoder.pre_trans_in.bias.data.copy_(sd["decoder.pre_transformer.input_proj.bias"].float())
    vocoder.pre_trans_out.weight.data.copy_(sd["decoder.pre_transformer.output_proj.weight"].float())
    vocoder.pre_trans_out.bias.data.copy_(sd["decoder.pre_transformer.output_proj.bias"].float())
    
    # ConvNeXt transposed convs
    vocoder.convnext_0_trans.weight.data.copy_(sd["decoder.upsample.0.0.conv.weight"].float())
    vocoder.convnext_0_trans.bias.data.copy_(sd["decoder.upsample.0.0.conv.bias"].float())
    vocoder.convnext_1_trans.weight.data.copy_(sd["decoder.upsample.1.0.conv.weight"].float())
    vocoder.convnext_1_trans.bias.data.copy_(sd["decoder.upsample.1.0.conv.bias"].float())
    
    # Causal conv
    vocoder.causal_conv.conv.weight.data.copy_(sd["decoder.decoder.0.conv.weight"].float())
    vocoder.causal_conv.conv.bias.data.copy_(sd["decoder.decoder.0.conv.bias"].float())
    
    # Upsample stages
    upsample_channels = [768, 384, 192, 96]
    for stage in range(4):
        prefix = f"decoder.decoder.{stage + 1}"
        
        # SnakeBeta
        vocoder.upsample_acts[stage].alpha.data.copy_(sd[f"{prefix}.block.0.alpha"].float())
        vocoder.upsample_acts[stage].beta.data.copy_(sd[f"{prefix}.block.0.beta"].float())
        
        # Transposed conv
        vocoder.upsample_convs[stage].weight.data.copy_(sd[f"{prefix}.block.1.conv.weight"].float())
        vocoder.upsample_convs[stage].bias.data.copy_(sd[f"{prefix}.block.1.conv.bias"].float())
        
        # ResBlocks
        for rb_idx, hf_rb in enumerate([2, 3, 4]):
            rb_prefix = f"{prefix}.block.{hf_rb}"
            rb = vocoder.upsample_resblocks[stage][rb_idx]
            
            rb.act1.alpha.data.copy_(sd[f"{rb_prefix}.act1.alpha"].float())
            rb.act1.beta.data.copy_(sd[f"{rb_prefix}.act1.beta"].float())
            rb.conv1.conv.weight.data.copy_(sd[f"{rb_prefix}.conv1.conv.weight"].float())
            rb.conv1.conv.bias.data.copy_(sd[f"{rb_prefix}.conv1.conv.bias"].float())
            rb.act2.alpha.data.copy_(sd[f"{rb_prefix}.act2.alpha"].float())
            rb.act2.beta.data.copy_(sd[f"{rb_prefix}.act2.beta"].float())
            rb.conv2.weight.data.copy_(sd[f"{rb_prefix}.conv2.conv.weight"].float())
            rb.conv2.bias.data.copy_(sd[f"{rb_prefix}.conv2.conv.bias"].float())
    
    # Final
    vocoder.final_act.alpha.data.copy_(sd["decoder.decoder.5.alpha"].float())
    vocoder.final_act.beta.data.copy_(sd["decoder.decoder.5.beta"].float())
    vocoder.final_conv.conv.weight.data.copy_(sd["decoder.decoder.6.conv.weight"].float())
    vocoder.final_conv.conv.bias.data.copy_(sd["decoder.decoder.6.conv.bias"].float())
    
    print("Weights loaded!")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--codes", required=True, help="Path to codes.bin")
    parser.add_argument("--output", default="python_output.wav")
    args = parser.parse_args()
    
    # Load codes
    codes = np.fromfile(args.codes, dtype=np.int32).reshape(-1, 16)
    codes = torch.from_numpy(codes).unsqueeze(0)  # [1, T, 16]
    print(f"Codes shape: {codes.shape}")
    
    # Create and load vocoder
    vocoder = Vocoder()
    load_weights(vocoder, args.model_path)
    vocoder.eval()
    
    # Run inference
    print("\n=== Running vocoder ===")
    with torch.no_grad():
        audio = vocoder(codes)
    
    print(f"\nOutput audio shape: {audio.shape}")
    print(f"Output range: [{audio.min():.4f}, {audio.max():.4f}]")
    
    # Save
    audio_np = audio.squeeze().numpy()
    sf.write(args.output, audio_np, 24000)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
