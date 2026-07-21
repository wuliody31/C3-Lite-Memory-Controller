from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from src.backbones import MockBackbone, OllamaBackbone
from src.config import load_config
from src.pipeline import C3Pipeline
from src.retrievers import InMemoryMemoryStore, Neo4jMemoryStore, ProceduralJsonStore
from src.schemas import QueryState

ROOT = Path(__file__).resolve().parent


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one query through C3-Lite.")
    p.add_argument("--query", required=True); p.add_argument("--user-id", required=True)
    p.add_argument("--config", default=str(ROOT / "configs/c3_lite_v2_2_final.yaml")); p.add_argument("--memory-json"); p.add_argument("--procedure-json"); p.add_argument("--neo4j", action="store_true")
    p.add_argument("--backbone", choices=["mock", "ollama"], default="mock"); p.add_argument("--model"); p.add_argument("--show-trace", action="store_true")
    return p.parse_args()


def build(a: argparse.Namespace) -> C3Pipeline:
    config = load_config(a.config); stopwords = set(config["query_analysis"].get("stopwords", []))
    if a.neo4j:
        missing = [x for x in ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"] if not os.getenv(x)]
        if missing: raise RuntimeError(f"Missing environment variables: {missing}")
        memory = Neo4jMemoryStore(uri=os.environ["NEO4J_URI"], user=os.environ["NEO4J_USER"], password=os.environ["NEO4J_PASSWORD"], config=config)
    elif a.memory_json: memory = InMemoryMemoryStore.from_json_file(a.memory_json, stopwords)
    else: raise RuntimeError("Provide --memory-json or --neo4j.")
    procedures = ProceduralJsonStore.from_file(a.procedure_json, stopwords) if a.procedure_json else None
    backbone = OllamaBackbone(model=a.model or config["generation"]["default_model"], base_url=config["generation"]["ollama_base_url"]) if a.backbone == "ollama" else MockBackbone()
    return C3Pipeline(config=config, memory_store=memory, procedure_store=procedures, backbone=backbone, prompt_template=ROOT / "prompts/answer_prompt.txt")


def main() -> None:
    a = args(); pipeline = build(a)
    try:
        result = pipeline.answer(QueryState(a.query, a.user_id))
        print("\nANSWER\n------\n" + result.answer)
        print("\nCONTROLLER\n" + json.dumps({"decision": result.decision.value, "query_mode": result.query_mode.value, "route": result.selected_memory_types, "route_scores": result.route_scores, "selected_ids": result.selected_ids, "coverage": result.coverage, "adequacy": result.adequacy, "agreement": result.agreement, "latency_ms": result.latency_ms}, ensure_ascii=False, indent=2))
        if a.show_trace: print("\nFULL TRACE\n----------\n" + json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    finally: pipeline.close()

if __name__ == "__main__": main()
