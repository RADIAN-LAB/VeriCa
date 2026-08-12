"""Adapters that add intent-aware reuse without modifying CoCa."""

from .intent_distributed import IntentAwareDistributedCacheManager

__all__ = ["IntentAwareDistributedCacheManager"]
