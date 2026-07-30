from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_config
from src.pipeline import C3Pipeline
from src.retrievers import (
    InMemoryMemoryStore,
    ProceduralJsonStore,
)
from src.schemas import QueryState
from src.transformers_backbone import (
    TransformersBackbone,
)


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one C3-Lite query using a "
            "Hugging Face GPU backbone."
        )
    )

    parser.add_argument(
        "--query",
        required=True,
    )
    parser.add_argument(
        "--user-id",
        required=True,
    )
    parser.add_argument(
        "--config",
        default=str(
            ROOT
            / "configs"
            / "c3_lite_v2_2_final.yaml"
        ),
    )
    parser.add_argument(
        "--memory-json",
        required=True,
    )
    parser.add_argument(
        "--procedure-json",
    )
    parser.add_argument(
        "--model",
    )
    parser.add_argument(
        "--adapter-path",
    )
    parser.add_argument(
        "--show-trace",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    stopwords = set(
        config["query_analysis"].get(
            "stopwords",
            [],
        )
    )

    memory_store = (
        InMemoryMemoryStore.from_json_file(
            args.memory_json,
            stopwords,
        )
    )

    procedure_store = None

    if args.procedure_json:
        procedure_store = (
            ProceduralJsonStore.from_file(
                args.procedure_json,
                stopwords,
            )
        )

    generation_config = config["generation"]

    configured_adapter = generation_config.get(
        "transformers_adapter_path"
    )

    adapter_path = (
        args.adapter_path
        if args.adapter_path
        else configured_adapter
    )

    backbone = TransformersBackbone(
        model=(
            args.model
            or generation_config[
                "transformers_model"
            ]
        ),
        load_in_4bit=bool(
            generation_config.get(
                "transformers_load_in_4bit",
                True,
            )
        ),
        compute_dtype=str(
            generation_config.get(
                "transformers_compute_dtype",
                "bfloat16",
            )
        ),
        enable_thinking=bool(
            generation_config.get(
                "transformers_enable_thinking",
                False,
            )
        ),
        adapter_path=adapter_path,
    )

    pipeline = C3Pipeline(
        config=config,
        memory_store=memory_store,
        procedure_store=procedure_store,
        backbone=backbone,
        prompt_template=(
            ROOT
            / "prompts"
            / "answer_prompt.txt"
        ),
    )

    try:
        result = pipeline.answer(
            QueryState(
                args.query,
                args.user_id,
            )
        )

        print("\nANSWER")
        print("------")
        print(result.answer)

        controller_summary = {
            "decision": result.decision.value,
            "query_mode": result.query_mode.value,
            "route": result.selected_memory_types,
            "route_scores": result.route_scores,
            "selected_ids": result.selected_ids,
            "coverage": result.coverage,
            "adequacy": result.adequacy,
            "agreement": result.agreement,
            "latency_ms": result.latency_ms,
        }

        print("\nCONTROLLER")
        print(
            json.dumps(
                controller_summary,
                ensure_ascii=False,
                indent=2,
            )
        )

        if args.show_trace:
            print("\nFULL TRACE")
            print("----------")
            print(
                json.dumps(
                    result.to_dict(),
                    ensure_ascii=False,
                    indent=2,
                )
            )
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
