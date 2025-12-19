"""
Models subpackage for LNN-S4 neural network components.

This package contains the Liquid-S4 model architecture for expression control:
- LNNS4Config: Model configuration dataclass
- S4Layer: Structured State Space layer with HiPPO initialization
- S4Block: S4 layer with residual connections and FFN
- LiquidS4Model: Complete model combining S4 and LTC layers
"""

from .config import LNNS4Config
from .s4 import S4Layer, S4Block, make_hippo_matrix
from .liquid_s4 import LiquidS4Model, create_model

__all__ = [
    'LNNS4Config',
    'S4Layer',
    'S4Block',
    'make_hippo_matrix',
    'LiquidS4Model',
    'create_model',
]
