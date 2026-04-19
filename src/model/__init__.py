"""
src/model — model architecture package.

Public API:
    from src.model import ConvAutoencoder
"""

from .encoder     import Encoder, EncoderBlock
from .decoder     import Decoder, DecoderBlock
from .autoencoder import ConvAutoencoder

__all__ = [
    "EncoderBlock",
    "Encoder",
    "DecoderBlock",
    "Decoder",
    "ConvAutoencoder",
]