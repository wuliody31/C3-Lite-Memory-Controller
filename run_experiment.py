from __future__ import annotations

import argparse
import os
from pathlib import Path
from evaluation.run_benchmark import run_benchmark
from src.backbones import MockBackbone, OllamaBackbone
from src.baselines import BaselineRunner
from src.config import frozen_config, load_config
from src.pipeline import C3Pipeline
from src.retrievers import InMemoryMemoryStore, Neo4jMemoryStore, ProceduralJsonStore

ROOT = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--dataset", required=True); p.add_argument("--methods", nargs="+", default=["no_memory", "episodic_only", "semantic_only", "procedural_only", "simple_retrieval", "all_memory", "c3"]); p.add_argument("--config", default=str(ROOT / "configs/c3_lite_v2_2_final.yaml")); p.add_argument("--memory-json"); p.add_argument("--procedure-json"); p.add_argument("--neo4j", action="store_true"); p.add_argument("--backbone", choices=["mock", "ollama"], default="mock"); p.add_argument("--model"); p.add_argument("--output", required=True); a = p.parse_args()
    config = load_config(a.config); stopwords = set(config["query_analysis"].get("stopwords", []))
    if a.neo4j: memory = Neo4jMemoryStore(uri=os.environ["NEO4J_URI"], user=os.environ["NEO4J_USER"], password=os.environ["NEO4J_PASSWORD"], config=config)
    elif a.memory_json: memory = InMemoryMemoryStore.from_json_file(a.memory_json, stopwords)
    else: raise RuntimeError("Provide --memory-json or --neo4j.")
    procedures = ProceduralJsonStore.from_file(a.procedure_json, stopwords) if a.procedure_json else None
    backbone = OllamaBackbone(model=a.model or config["generation"]["default_model"], base_url=config["generation"]["ollama_base_url"]) if a.backbone == "ollama" else MockBackbone()
    prompt = ROOT / "prompts/answer_prompt.txt"
    c3 = C3Pipeline(config=config, memory_store=memory, procedure_store=procedures, backbone=backbone, prompt_template=prompt)
    baseline = BaselineRunner(config=config, memory_store=memory, procedure_store=procedures, backbone=backbone, prompt_template=prompt)
    try:
        run_benchmark(dataset_path=a.dataset, methods=a.methods, c3_pipeline=c3, baseline_runner=baseline, output_dir=a.output, config_snapshot={**frozen_config(config), "_runtime": {"dataset": a.dataset, "methods": a.methods, "backbone": a.backbone, "model": a.model or config["generation"]["default_model"], "memory_source": "neo4j" if a.neo4j else a.memory_json, "procedure_source": a.procedure_json}})
    finally: c3.close()

if __name__ == "__main__": main()
