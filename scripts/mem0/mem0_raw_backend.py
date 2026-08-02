from __future__ import annotations

from pathlib import Path
from typing import Any


VECTOR_ROOT = Path(
    "/data/alyjw80/c3_lite/"
    "external/mem0/raw_backend"
)

MODEL_CACHE = Path(
    "/data/alyjw80/c3_lite/cache/"
    "huggingface/sentence_transformers"
)


def build_raw_backend_config(
    *,
    collection_name: str,
    database_name: str,
) -> dict[str, Any]:
    """Build a provenance-preserving Mem0 configuration.

    The LLM configuration is present because Mem0 constructs an LLM
    provider during initialisation. It is not called when add() uses
    infer=False.
    """
    vector_path = (
        VECTOR_ROOT
        / database_name
        / "qdrant"
    )

    history_path = (
        VECTOR_ROOT
        / database_name
        / "history.db"
    )

    vector_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return {
        "llm": {
            "provider": "openai",
            "config": {
                "model": "unused-infer-false",
                "api_key": "unused-local-placeholder",
                "temperature": 0.0,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": (
                    "sentence-transformers/"
                    "multi-qa-MiniLM-L6-cos-v1"
                ),
                "embedding_dims": 384,
                "model_kwargs": {
                    "device": "cpu",
                    "cache_folder": str(
                        MODEL_CACHE
                    ),
                },
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": collection_name,
                "embedding_model_dims": 384,
                "path": str(vector_path),
                "on_disk": True,
            },
        },
        "history_db_path": str(history_path),
    }
