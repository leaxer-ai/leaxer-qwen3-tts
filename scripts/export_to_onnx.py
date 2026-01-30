#!/usr/bin/env python3
"""
Export Qwen3-TTS components to ONNX format.

Usage:
    python scripts/export_to_onnx.py --model Qwen/Qwen3-TTS-0.6B-Base --output models/onnx

Components to export (in order of difficulty):
1. speaker_encoder.onnx - ECAPA-TDNN for voice cloning (easy)
2. vocoder.onnx - DiT + BigVGAN decoder (medium)
3. code_predictor.onnx - Sub-LLM for codebooks 1-15 (medium)
4. talker_prefill.onnx - Main LLM prefill phase (hard)
5. talker_decode.onnx - Main LLM decode step (hard)

Note: The Talker export is complex due to:
- M-RoPE (3D position embeddings)
- KV cache management
- Interleaved CodePredictor calls
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ONNX opset version - 17 recommended for modern transformers
OPSET_VERSION = 17


class VocoderWrapper(nn.Module):
    """Wrapper to make vocoder export-friendly."""
    
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
    
    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            codes: [batch, seq_len, 16] int64 codec tokens
        
        Returns:
            audio: [batch, audio_len] float32 waveform
        """
        # The decoder.decode() method handles the conversion
        return self.decoder.decode(codes)


class SpeakerEncoderWrapper(nn.Module):
    """Wrapper for ECAPA-TDNN speaker encoder."""
    
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    
    def forward(self, mels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mels: [batch, time, 128] mel spectrogram
        
        Returns:
            embedding: [batch, 1024] speaker embedding
        """
        return self.encoder(mels)


class CodePredictorWrapper(nn.Module):
    """Wrapper for CodePredictor single-step inference."""
    
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor
    
    def forward(
        self,
        talker_hidden: torch.Tensor,
        codec_0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict codebooks 1-15 given codebook 0 and talker hidden state.
        
        Args:
            talker_hidden: [batch, 1, 2048] from Talker
            codec_0: [batch, 1] int64 codebook 0 token
        
        Returns:
            codebooks: [batch, 15] int64 codebooks 1-15
        """
        # Project talker hidden to predictor dimension
        projected = self.predictor.small_to_mtp_projection(talker_hidden)
        
        # Get embedding for codec_0
        codec_embed = self.predictor.model.get_input_embeddings()[0](codec_0)
        
        # Concatenate as input
        inputs_embeds = torch.cat([projected, codec_embed], dim=1)
        
        # Generate remaining codebooks
        result = self.predictor.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=15,  # codebooks 1-15
            do_sample=False,  # For ONNX, use greedy
        )
        
        return result.sequences


def export_vocoder(
    speech_tokenizer,
    output_path: Path,
    device: str = "cpu",
) -> bool:
    """
    Export the vocoder (decoder) to ONNX.
    
    Args:
        speech_tokenizer: The speech tokenizer containing the decoder
        output_path: Path to save the ONNX model
        device: Device to use for export
    
    Returns:
        True if export successful, False otherwise
    """
    logger.info(f"Exporting vocoder to {output_path}")
    
    try:
        # Get the decoder model
        decoder = speech_tokenizer.model
        decoder = decoder.to(device)
        decoder.eval()
        
        # Wrapper for clean export
        wrapper = VocoderWrapper(decoder)
        wrapper.eval()
        
        # Dummy inputs - typical TTS output length
        batch_size = 1
        seq_len = 100  # ~8 seconds at 12Hz
        num_codebooks = 16
        
        dummy_codes = torch.randint(
            0, 2048,
            (batch_size, seq_len, num_codebooks),
            dtype=torch.long,
            device=device,
        )
        
        # Export to ONNX
        torch.onnx.export(
            wrapper,
            (dummy_codes,),
            str(output_path),
            input_names=["codes"],
            output_names=["audio"],
            dynamic_axes={
                "codes": {0: "batch", 1: "seq_len"},
                "audio": {0: "batch", 1: "audio_len"},
            },
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
        )
        
        logger.info(f"Vocoder exported successfully to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to export vocoder: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_speaker_encoder(
    model,
    output_path: Path,
    device: str = "cpu",
) -> bool:
    """
    Export speaker encoder (ECAPA-TDNN) to ONNX.
    
    Args:
        model: The full Qwen3TTS model
        output_path: Path to save the ONNX model
        device: Device to use for export
    
    Returns:
        True if export successful, False otherwise
    """
    logger.info(f"Exporting speaker encoder to {output_path}")
    
    if model.speaker_encoder is None:
        logger.warning("No speaker encoder found (not a Base model)")
        return False
    
    try:
        encoder = model.speaker_encoder
        encoder = encoder.to(device)
        encoder.eval()
        
        # Wrapper for clean export
        wrapper = SpeakerEncoderWrapper(encoder)
        wrapper.eval()
        
        # Dummy mel spectrogram
        batch_size = 1
        time_steps = 300  # ~3 seconds at 100 fps
        mel_dim = 128
        
        dummy_mels = torch.randn(
            batch_size, time_steps, mel_dim,
            dtype=torch.float32,
            device=device,
        )
        
        # Export to ONNX
        torch.onnx.export(
            wrapper,
            (dummy_mels,),
            str(output_path),
            input_names=["mels"],
            output_names=["speaker_embedding"],
            dynamic_axes={
                "mels": {0: "batch", 1: "time"},
                "speaker_embedding": {0: "batch"},
            },
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
        )
        
        logger.info(f"Speaker encoder exported successfully to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to export speaker encoder: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_code_predictor(
    model,
    output_path: Path,
    device: str = "cpu",
) -> bool:
    """
    Export CodePredictor to ONNX.
    
    This is complex because CodePredictor uses autoregressive generation.
    We export a single forward pass and implement the generation loop externally.
    
    Args:
        model: The full Qwen3TTS model
        output_path: Path to save the ONNX model
        device: Device to use for export
    
    Returns:
        True if export successful, False otherwise
    """
    logger.info(f"Exporting code predictor to {output_path}")
    
    # TODO: Implement proper export with KV cache handling
    # This requires:
    # 1. Export forward pass with explicit KV cache inputs/outputs
    # 2. Handle the 15 separate lm_head outputs
    # 3. Manage dynamic sequence lengths
    
    logger.warning("CodePredictor export not yet implemented - requires KV cache handling")
    return False


def export_talker(
    model,
    output_dir: Path,
    device: str = "cpu",
) -> Tuple[bool, bool]:
    """
    Export Talker to ONNX (both prefill and decode phases).
    
    Args:
        model: The full Qwen3TTS model
        output_dir: Directory to save ONNX models
        device: Device to use for export
    
    Returns:
        Tuple of (prefill_success, decode_success)
    """
    logger.info("Exporting Talker (prefill and decode phases)")
    
    # TODO: Implement proper export
    # This is the most complex part due to:
    # 1. M-RoPE (3D position embeddings)
    # 2. Mixed text + codec embeddings
    # 3. KV cache management
    # 4. Interleaved CodePredictor calls
    
    logger.warning("Talker export not yet implemented - requires M-RoPE and KV cache handling")
    return False, False


def verify_onnx_model(onnx_path: Path, test_input: dict) -> bool:
    """
    Verify an exported ONNX model by running inference.
    
    Args:
        onnx_path: Path to ONNX model
        test_input: Dictionary of numpy arrays for model inputs
    
    Returns:
        True if verification passed, False otherwise
    """
    try:
        import onnx
        import onnxruntime as ort
        
        # Load and check model
        logger.info(f"Verifying {onnx_path}")
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        
        # Run inference
        sess = ort.InferenceSession(str(onnx_path))
        outputs = sess.run(None, test_input)
        
        logger.info(f"Verification passed! Output shapes: {[o.shape for o in outputs]}")
        return True
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3-TTS to ONNX")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-TTS-0.6B-Base",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/onnx"),
        help="Output directory for ONNX models",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for export (cpu recommended for compatibility)",
    )
    parser.add_argument(
        "--components",
        type=str,
        nargs="+",
        default=["vocoder", "speaker_encoder"],
        choices=["vocoder", "speaker_encoder", "code_predictor", "talker", "all"],
        help="Components to export",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify exported models with test inference",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model: {args.model}")
    
    try:
        from qwen_tts import Qwen3TTSModel
        
        model = Qwen3TTSModel.from_pretrained(
            args.model,
            torch_dtype=torch.float32,
            device_map=args.device,
        )
        logger.info("Model loaded successfully")
        
    except ImportError:
        logger.error("Failed to import qwen_tts. Please install it first:")
        logger.error("  pip install qwen-tts")
        return 1
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1
    
    # Expand "all"
    if "all" in args.components:
        args.components = ["vocoder", "speaker_encoder", "code_predictor", "talker"]
    
    # Track results
    results = {}
    
    # Export each component
    if "vocoder" in args.components:
        success = export_vocoder(
            model.model.speech_tokenizer,
            args.output / "vocoder.onnx",
            args.device,
        )
        results["vocoder"] = success
        
        if success and args.verify:
            test_codes = np.random.randint(0, 2048, (1, 50, 16)).astype(np.int64)
            verify_onnx_model(args.output / "vocoder.onnx", {"codes": test_codes})
    
    if "speaker_encoder" in args.components:
        success = export_speaker_encoder(
            model.model,
            args.output / "speaker_encoder.onnx",
            args.device,
        )
        results["speaker_encoder"] = success
        
        if success and args.verify:
            test_mels = np.random.randn(1, 200, 128).astype(np.float32)
            verify_onnx_model(args.output / "speaker_encoder.onnx", {"mels": test_mels})
    
    if "code_predictor" in args.components:
        success = export_code_predictor(
            model.model,
            args.output / "code_predictor.onnx",
            args.device,
        )
        results["code_predictor"] = success
    
    if "talker" in args.components:
        prefill_ok, decode_ok = export_talker(
            model.model,
            args.output,
            args.device,
        )
        results["talker_prefill"] = prefill_ok
        results["talker_decode"] = decode_ok
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("Export Summary:")
    for component, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"  {component}: {status}")
    logger.info("=" * 50)
    
    # Return code based on any failures
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    exit(main())
