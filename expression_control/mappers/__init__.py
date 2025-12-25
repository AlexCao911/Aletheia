"""
Feature-to-angle mappers for expression control.

Two implementations available:
- RuleMapper: Hand-crafted rules, no training required
- LiquidS4Mapper: Trained neural network (ONNX format)
"""

from expression_control.mappers.base import FeatureToAngleMapper
from expression_control.mappers.rule_mapper import RuleMapper
from expression_control.mappers.liquid_s4_mapper import LiquidS4Mapper

__all__ = [
    "FeatureToAngleMapper",
    "RuleMapper",
    "LiquidS4Mapper",
]
