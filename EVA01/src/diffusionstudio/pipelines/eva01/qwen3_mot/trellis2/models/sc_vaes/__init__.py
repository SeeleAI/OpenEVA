"""Inference-only sparse convolutional VAE decoders from TRELLIS.2."""

from .fdg_vae import FlexiDualGridVaeDecoder
from .sparse_unet_vae import SparseUnetVaeDecoder

__all__ = ["FlexiDualGridVaeDecoder", "SparseUnetVaeDecoder"]
