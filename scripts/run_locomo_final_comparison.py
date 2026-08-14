from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from src.config import load_config

VARIANTS = (
    "simple_retrieval",
    "mem0_top20",
    "mem0_c3_matched",
    "c3_v3",
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise TypeError(f"{path}:{n} is not a JSON object")
        rows.append(row)
    return rows


def append_jsonl(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "t"}


def unique_method_rows(rows: list[dict[str, Any]], methods: set[str], label: str):
    out = {}
    for row in rows:
        if str(row.get("method", "")).lower() not in methods:
            continue
        qid = str(row["question_id"])
        if qid in out:
            raise AssertionError(f"Duplicate {label} question_id={qid}")
        out[qid] = row
    return out


def mean(rows: list[dict[str, Any]], field: str) -> float:
    if not rows:
        return 0.0
    return statistics.fmean(float(row.get(field, 0.0) or 0.0) for row in rows)


def evidence_payload(ids, gold_source_ids, source_map, runner):
    source_ids = runner.project_ids(ids, source_map)
    metrics = runner.evidence_metrics(source_ids, gold_source_ids)
    return {
        "selected_ids": ids,
        "selected_source_ids": source_ids,
        "selected_count": len(ids),
        "evidence_precision": float(metrics["precision"]),
        "evidence_recall": float(metrics["recall"]),
        "evidence_f1": float(metrics["f1"]),
        "evidence_hit": bool(metrics["hit"]),
        "evidence_overlap_source_ids": list(metrics["overlap_source_ids"]),
    }


def neutral_prompt(query: str, ids: list[str], catalog: dict[str, Any]) -> str:
    lines = [
        "SYSTEM", "",
        "Answer the user question using only the memory evidence below.",
        "Do not invent unsupported facts.",
        "If the evidence is insufficient to determine the answer, say so clearly.",
        "Do not present historical information as current unless the evidence supports it.",
        "", "USER QUERY", "", query.strip(), "", "MEMORY EVIDENCE", "",
    ]
    if not ids:
        lines.append("- None")
    else:
        for mid in ids:
            c = catalog[mid]
            ts = getattr(c, "timestamp", None)
            ts_text = ts.isoformat() if ts is not None else "unknown"
            lines.extend([f"[{mid}] (time={ts_text})", str(getattr(c, "text", "")).strip(), ""])
    lines.extend(["OUTPUT REQUIREMENTS", "", "Answer directly and concisely using only the evidence above."])
    return "\n".join(lines).strip()


def prepare(args):
    runner = load_module(args.runner_module, "locomo_final_runner")
    questions = read_csv(args.questions)
    if len(questions) != args.expected_questions:
        raise AssertionError(f"Expected {args.expected_questions}, found {len(questions)}")
    qmap = {str(q["question_id"]): q for q in questions}
    if len(qmap) != len(questions):
        raise AssertionError("Duplicate canonical question IDs")

    records = runner.load_memory_records(args.memories)
    catalog, source_map = runner.build_catalog(records)
    c3 = unique_method_rows(read_jsonl(args.c3_predictions), {"c3", "c3_lite_controller"}, "C3")
    simple = unique_method_rows(read_jsonl(args.simple_predictions), {"simple_retrieval"}, "Simple")
    mem0_rows = read_jsonl(args.mem0_top20)
    mem0 = {str(r["question_id"]): r for r in mem0_rows}
    expected = set(qmap)
    for label, mapping in (("C3", c3), ("Simple", simple), ("Mem0", mem0)):
        if set(mapping) != expected:
            raise AssertionError(f"{label} question-ID mismatch")

    output = []
    per_variant = {v: [] for v in VARIANTS}
    missing_catalog = set()

    for qid in sorted(expected):
        q = qmap[qid]
        gold_memory_ids = runner.split_ids(q.get("supporting_memory_ids", ""))
        gold_source_ids = runner.project_ids(gold_memory_ids, source_map)
        c3_ids = runner.split_ids(c3[qid].get("selected_ids", []))
        simple_ids = runner.split_ids(simple[qid].get("selected_ids", []))
        mem0_ids = runner.split_ids(mem0[qid].get("retrieved_memory_ids", []))
        variants = {
            "simple_retrieval": simple_ids,
            "mem0_top20": mem0_ids,
            "mem0_c3_matched": mem0_ids[:len(c3_ids)],
            "c3_v3": c3_ids,
        }
        for ids in variants.values():
            missing_catalog.update(mid for mid in ids if mid not in catalog)
        payloads = {}
        for variant, ids in variants.items():
            payload = evidence_payload(ids, gold_source_ids, source_map, runner)
            payloads[variant] = payload
            per_variant[variant].append(payload)
        output.append({
            "question_id": qid,
            "user_id": str(q["user_id"]),
            "question_type": str(q.get("question_type", "unknown")),
            "query": str(q.get("question", c3[qid].get("query", ""))),
            "gold_answer": str(q.get("gold_answer", c3[qid].get("gold_answer", ""))),
            "should_abstain": parse_bool(q.get("should_abstain", False)),
            "gold_memory_ids": gold_memory_ids,
            "gold_source_ids": gold_source_ids,
            "variants": payloads,
        })

    if missing_catalog:
        raise AssertionError(f"IDs absent from catalog: {sorted(missing_catalog)[:20]}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    inputs = args.output_root / "comparison_inputs.jsonl"
    with inputs.open("w", encoding="utf-8") as handle:
        for row in output:
            append_jsonl(handle, row)
    users = sorted({r["user_id"] for r in output})
    (args.output_root / "users.txt").write_text("\n".join(users) + "\n", encoding="utf-8")

    summary = {}
    for variant, rows in per_variant.items():
        summary[variant] = {
            "questions": len(rows),
            "mean_selected_count": mean(rows, "selected_count"),
            "evidence_precision": mean(rows, "evidence_precision"),
            "evidence_recall": mean(rows, "evidence_recall"),
            "evidence_f1": mean(rows, "evidence_f1"),
            "evidence_hit_rate": mean(rows, "evidence_hit"),
        }
    write_json(args.output_root / "retrieval_summary.json", summary)
    write_json(args.output_root / "prepare_manifest.json", {
        "experiment": "locomo_heldout1786_final_external_comparison",
        "dataset_role": "post-development-heldout-from-pilot",
        "questions": len(output),
        "users": users,
        "variants": list(VARIANTS),
        "mem0_matched_definition": "top len(C3-v3 selected_ids) per question",
        "prompt_policy": "same neutral evidence-only prompt for all variants",
        "sha256": {
            "questions": sha256_file(args.questions),
            "memories": sha256_file(args.memories),
            "c3_predictions": sha256_file(args.c3_predictions),
            "simple_predictions": sha256_file(args.simple_predictions),
            "mem0_top20": sha256_file(args.mem0_top20),
            "runner_module": sha256_file(args.runner_module),
            "comparison_inputs": sha256_file(inputs),
        },
    })
    print("LOCOMO FINAL PREPARATION: PASS")
    print("questions:", len(output))
    print("users:", len(users), users)
    print(json.dumps(summary, indent=2))


def generate(args):
    qwen = load_module(args.qwen_helper, "locomo_final_qwen")
    runner = load_module(args.runner_module, "locomo_final_runner")
    config = load_config(args.config)
    generation = config["generation"]
    records = runner.load_memory_records(args.memories)
    catalog, _ = runner.build_catalog(records)
    rows = read_jsonl(args.inputs)
    if args.user_id:
        rows = [r for r in rows if str(r["user_id"]) == args.user_id]
    if args.max_questions > 0:
        rows = rows[:args.max_questions]
    if not rows:
        raise AssertionError("No questions selected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = args.output_dir / "predictions.jsonl"
    existing = read_jsonl(predictions)
    completed = {(str(r["question_id"]), str(r["variant"])) for r in existing}
    expected = {(str(r["question_id"]), v) for r in rows for v in VARIANTS}
    if completed - expected:
        raise AssertionError("Existing output contains unexpected rows")

    print("Loading Qwen3 backbone...", flush=True)
    t0 = time.perf_counter()
    backbone, manifest = qwen.build_backbone(generation)
    manifest.update({
        "construction_seconds": time.perf_counter() - t0,
        "variants": list(VARIANTS),
        "prompt_policy": "neutral_evidence_only_v01",
        "user_id": args.user_id,
        "questions_in_task": len(rows),
    })
    if not args.skip_backbone_smoke:
        s0 = time.perf_counter()
        smoke = backbone.generate("Reply with exactly: BACKBONE_READY", temperature=0.0, max_new_tokens=16)
        manifest["smoke"] = {
            "text": smoke.text,
            "input_tokens": smoke.input_tokens,
            "output_tokens": smoke.output_tokens,
            "latency_seconds": time.perf_counter() - s0,
        }
        print("Backbone smoke:", smoke.text, flush=True)
    write_json(args.output_dir / "backbone_manifest.json", manifest)

    with predictions.open("a", encoding="utf-8") as handle:
        for qi, row in enumerate(rows, 1):
            qid = str(row["question_id"])
            refs = qwen.answer_references(row.get("gold_answer", ""))
            for variant in VARIANTS:
                key = (qid, variant)
                if key in completed:
                    continue
                ids = list(row["variants"][variant]["selected_ids"])
                prompt = neutral_prompt(str(row["query"]), ids, catalog)
                print(f"[{len(completed)+1}/{len(expected)}] q={qi}/{len(rows)} id={qid} variant={variant}", flush=True)
                g0 = time.perf_counter()
                result = backbone.generate(
                    prompt,
                    temperature=float(generation.get("temperature", 0.0)),
                    max_new_tokens=int(generation.get("max_new_tokens", 256)),
                )
                latency = (time.perf_counter() - g0) * 1000.0
                scores = qwen.score_answer(result.text, refs)
                out = {
                    "question_id": qid,
                    "user_id": row["user_id"],
                    "question_type": row["question_type"],
                    "variant": variant,
                    "query": row["query"],
                    "gold_answer": refs,
                    "should_abstain": bool(row["should_abstain"]),
                    "answer": result.text,
                    **scores,
                    "gold_memory_ids": row["gold_memory_ids"],
                    "gold_source_ids": row["gold_source_ids"],
                    **dict(row["variants"][variant]),
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "input_tokens_numeric": result.input_tokens or 0,
                    "output_tokens_numeric": result.output_tokens or 0,
                    "generation_latency_ms": latency,
                    "final_prompt": prompt,
                }
                append_jsonl(handle, out)
                completed.add(key)

    final = read_jsonl(predictions)
    keys = {(str(r["question_id"]), str(r["variant"])) for r in final}
    if keys != expected:
        raise AssertionError(f"Incomplete generation: {sorted(expected - keys)[:20]}")
    print("LOCOMO FINAL GENERATION TASK: PASS")
    print("questions:", len(rows), "generations:", len(expected))


def analyze(args):
    qwen = load_module(args.qwen_helper, "locomo_final_qwen")
    inputs = read_jsonl(args.inputs)
    qids = sorted(str(r["question_id"]) for r in inputs)
    rows = []
    for path in sorted(args.runs_root.glob("task_*/predictions.jsonl")):
        rows.extend(read_jsonl(path))
    by_key = {}
    for row in rows:
        key = (str(row["question_id"]), str(row["variant"]))
        if key in by_key:
            raise AssertionError(f"Duplicate row: {key}")
        by_key[key] = row
    expected = {(qid, v) for qid in qids for v in VARIANTS}
    if set(by_key) != expected:
        raise AssertionError(f"Generation mismatch: missing={sorted(expected-set(by_key))[:20]}")
    ordered = [by_key[(qid, v)] for qid in qids for v in VARIANTS]
    by_variant = {v: {str(r["question_id"]): r for r in ordered if r["variant"] == v} for v in VARIANTS}

    summaries = {}
    for variant in VARIANTS:
        selected = list(by_variant[variant].values())
        answerable = [r for r in selected if not bool(r["should_abstain"])]
        summaries[variant] = {
            "questions": len(selected),
            "answer_f1_all": mean(selected, "answer_f1"),
            "answer_f1_answerable": mean(answerable, "answer_f1"),
            "answer_exact_match": mean(selected, "answer_exact_match"),
            "evidence_precision": mean(selected, "evidence_precision"),
            "evidence_recall": mean(selected, "evidence_recall"),
            "evidence_f1": mean(selected, "evidence_f1"),
            "evidence_hit_rate": mean(selected, "evidence_hit"),
            "mean_selected_count": mean(selected, "selected_count"),
            "mean_input_tokens": mean(selected, "input_tokens_numeric"),
            "mean_output_tokens": mean(selected, "output_tokens_numeric"),
            "mean_generation_latency_ms": mean(selected, "generation_latency_ms"),
        }

    comparisons = {
        "c3_minus_simple": ("simple_retrieval", "c3_v3"),
        "c3_minus_mem0_top20": ("mem0_top20", "c3_v3"),
        "c3_minus_mem0_c3_matched": ("mem0_c3_matched", "c3_v3"),
    }
    paired = {}
    for i, (name, (left, right)) in enumerate(comparisons.items()):
        paired[name] = {}
        for j, field in enumerate(("answer_f1", "evidence_f1")):
            deltas = [float(by_variant[right][qid][field]) - float(by_variant[left][qid][field]) for qid in qids]
            stat = qwen.bootstrap_mean_delta(
                deltas,
                repetitions=args.bootstrap_repetitions,
                seed=args.bootstrap_seed + i * 10 + j,
            )
            stat.update({
                "left": left,
                "right": right,
                "right_better": sum(x > 0 for x in deltas),
                "right_worse": sum(x < 0 for x in deltas),
                "tied": sum(x == 0 for x in deltas),
            })
            paired[name][field] = stat

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "predictions_all.jsonl").open("w", encoding="utf-8") as handle:
        for row in ordered:
            append_jsonl(handle, row)
    write_json(args.output_dir / "summary.json", {
        "experiment": "locomo_heldout1786_final_external_comparison",
        "questions": len(qids),
        "generations": len(ordered),
        "variants": summaries,
        "paired_statistics": paired,
    })
    print("LOCOMO FINAL ANALYSIS: PASS")
    print(json.dumps(summaries, indent=2))
    print(json.dumps(paired, indent=2))


def parser():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="command", required=True)
    x = sp.add_parser("prepare")
    x.add_argument("--questions", type=Path, required=True)
    x.add_argument("--memories", type=Path, required=True)
    x.add_argument("--c3-predictions", type=Path, required=True)
    x.add_argument("--simple-predictions", type=Path, required=True)
    x.add_argument("--mem0-top20", type=Path, required=True)
    x.add_argument("--runner-module", type=Path, required=True)
    x.add_argument("--output-root", type=Path, required=True)
    x.add_argument("--expected-questions", type=int, default=1786)
    x = sp.add_parser("generate")
    x.add_argument("--inputs", type=Path, required=True)
    x.add_argument("--memories", type=Path, required=True)
    x.add_argument("--runner-module", type=Path, required=True)
    x.add_argument("--qwen-helper", type=Path, required=True)
    x.add_argument("--config", type=Path, required=True)
    x.add_argument("--output-dir", type=Path, required=True)
    x.add_argument("--user-id", default="")
    x.add_argument("--max-questions", type=int, default=0)
    x.add_argument("--skip-backbone-smoke", action="store_true")
    x = sp.add_parser("analyze")
    x.add_argument("--inputs", type=Path, required=True)
    x.add_argument("--runs-root", type=Path, required=True)
    x.add_argument("--qwen-helper", type=Path, required=True)
    x.add_argument("--output-dir", type=Path, required=True)
    x.add_argument("--bootstrap-repetitions", type=int, default=50000)
    x.add_argument("--bootstrap-seed", type=int, default=20260814)
    return p


def main():
    args = parser().parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "generate":
        generate(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
