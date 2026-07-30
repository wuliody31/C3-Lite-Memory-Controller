from __future__ import annotations

import argparse
import os
from pathlib import Path

from evaluation.run_benchmark import run_benchmark
from src.backbones import MockBackbone, OllamaBackbone
from src.baselines import BaselineRunner
from src.config import frozen_config, load_config
from src.pipeline import C3Pipeline
from src.retrievers import (
    InMemoryMemoryStore,
    Neo4jMemoryStore,
    ProceduralJsonStore,
)
from src.transformers_backbone import TransformersBackbone


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the C3-Lite benchmark."
    )

    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "no_memory",
            "episodic_only",
            "semantic_only",
            "procedural_only",
            "simple_retrieval",
            "all_memory",
            "c3",
        ],
    )
    parser.add_argument(
        "--config",
        default=str(
            ROOT / "configs" / "c3_lite_v2_2_final.yaml"
        ),
    )
    parser.add_argument("--memory-json")
    parser.add_argument("--procedure-json")
    parser.add_argument("--neo4j", action="store_true")
    parser.add_argument(
        "--backbone",
        choices=["mock", "ollama", "transformers"],
        default="mock",
    )
    parser.add_argument("--model")
    parser.add_argument("--adapter-path")
    parser.add_argument("--output", required=True)

    return parser.parse_args()


def build_memory_store(
    args: argparse.Namespace,
    config: dict,
):
    stopwords = set(
        config["query_analysis"].get("stopwords", [])
    )

    if args.neo4j:
        required = [
            "NEO4J_URI",
            "NEO4J_USER",
            "NEO4J_PASSWORD",
        ]
        missing = [
            name for name in required
            if not os.getenv(name)
        ]

        if missing:
            raise RuntimeError(
                f"Missing environment variables: {missing}"
            )

        return Neo4jMemoryStore(
            uri=os.environ["NEO4J_URI"],
            user=os.environ["NEO4J_USER"],
            password=os.environ["NEO4J_PASSWORD"],
            config=config,
        )

    if args.memory_json:
        return InMemoryMemoryStore.from_json_file(
            args.memory_json,
            stopwords,
        )

    raise RuntimeError(
        "Provide --memory-json or --neo4j."
    )


def build_procedure_store(
    args: argparse.Namespace,
    config: dict,
):
    if not args.procedure_json:
        return None

    stopwords = set(
        config["query_analysis"].get("stopwords", [])
    )

    return ProceduralJsonStore.from_file(
        args.procedure_json,
        stopwords,
    )


def build_backbone(
    args: argparse.Namespace,
    config: dict,
):
    generation = config["generation"]

    if args.backbone == "mock":
        return MockBackbone()

    if args.backbone == "ollama":
        return OllamaBackbone(
            model=(
                args.model
                or generation["default_model"]
            ),
            base_url=generation["ollama_base_url"],
        )

    configured_adapter = generation.get(
        "transformers_adapter_path"
    )

    return TransformersBackbone(
        model=(
            args.model
            or generation["transformers_model"]
        ),
        load_in_4bit=bool(
            generation.get(
                "transformers_load_in_4bit",
                True,
            )
        ),
        compute_dtype=str(
            generation.get(
                "transformers_compute_dtype",
                "bfloat16",
            )
        ),
        enable_thinking=bool(
            generation.get(
                "transformers_enable_thinking",
                False,
            )
        ),
        adapter_path=(
            args.adapter_path
            or configured_adapter
        ),
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    memory_store = build_memory_store(
        args,
        config,
    )
    procedure_store = build_procedure_store(
        args,
        config,
    )
    backbone = build_backbone(
        args,
        config,
    )

    prompt_template = (
        ROOT / "prompts" / "answer_prompt.txt"
    )

    c3_pipeline = C3Pipeline(
        config=config,
        memory_store=memory_store,
        procedure_store=procedure_store,
        backbone=backbone,
        prompt_template=prompt_template,
    )

    baseline_runner = BaselineRunner(
        config=config,
        memory_store=memory_store,
        procedure_store=procedure_store,
        backbone=backbone,
        prompt_template=prompt_template,
    )

    runtime_snapshot = {
        "dataset": args.dataset,
        "methods": args.methods,
        "backbone": args.backbone,
        "model": (
            args.model
            or (
                config["generation"][
                    "transformers_model"
                ]
                if args.backbone == "transformers"
                else config["generation"][
                    "default_model"
                ]
            )
        ),
        "adapter_path": args.adapter_path,
        "memory_source": (
            "neo4j"
            if args.neo4j
            else args.memory_json
        ),
        "procedure_source": args.procedure_json,
    }

    try:
        run_benchmark(
            dataset_path=args.dataset,
            methods=args.methods,
            c3_pipeline=c3_pipeline,
            baseline_runner=baseline_runner,
            output_dir=args.output,
            config_snapshot={
                **frozen_config(config),
                "_runtime": runtime_snapshot,
            },
        )
    finally:
        c3_pipeline.close()


if __name__ == "__main__":
    main()
