"""Stable public contracts for the C3 orchestration framework.

This module provides a compatibility facade over the protocol definitions
already used by the frozen research implementation. Existing import paths
remain supported.
"""

from ..backbones import Backbone, GenerationResult
from ..errors import (
    C3Error,
    ConfigurationError,
    GenerationError,
    ObservabilityError,
    ResourceCleanupError,
    RetrievalError,
)
from ..lifecycle import LifecycleStore
from ..retrievers.base import MemoryStore
from ..tracing import (
    BestEffortTraceSink,
    NullTraceSink,
    TraceSink,
)

# Semantic alias for callers that view retrieval as a capability rather than
# as a persistence implementation. The legacy MemoryStore name remains the
# canonical protocol for backward compatibility.
MemoryRetriever = MemoryStore

__all__ = [
    "Backbone",
    "C3Error",
    "ConfigurationError",
    "GenerationError",
    "ObservabilityError",
    "ResourceCleanupError",
    "RetrievalError",
    "BestEffortTraceSink",
    "GenerationResult",
    "LifecycleStore",
    "MemoryRetriever",
    "MemoryStore",
    "NullTraceSink",
    "TraceSink",
]
