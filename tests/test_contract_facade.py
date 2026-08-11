from src.backbones import Backbone as LegacyBackbone
from src.backbones import GenerationResult as LegacyGenerationResult
from src.contracts import (
    Backbone,
    GenerationResult,
    LifecycleStore,
    MemoryRetriever,
    MemoryStore,
)
from src.lifecycle import LifecycleStore as LegacyLifecycleStore
from src.retrievers.base import MemoryStore as LegacyMemoryStore


def test_public_contract_facade_preserves_existing_protocols() -> None:
    assert Backbone is LegacyBackbone
    assert GenerationResult is LegacyGenerationResult
    assert LifecycleStore is LegacyLifecycleStore
    assert MemoryStore is LegacyMemoryStore


def test_memory_retriever_alias_preserves_memory_store_contract() -> None:
    assert MemoryRetriever is MemoryStore


def test_public_trace_contract_facade_preserves_trace_protocols() -> None:
    from src.contracts import NullTraceSink as PublicNullTraceSink
    from src.contracts import TraceSink as PublicTraceSink
    from src.tracing import NullTraceSink, TraceSink

    assert PublicTraceSink is TraceSink
    assert PublicNullTraceSink is NullTraceSink
