from __future__ import annotations
import time
from pathlib import Path
from typing import Any
from tqdm import tqdm
from src.config import get_config
from src.io_utils import read_jsonl, write_jsonl, write_json, find_eval_questions_path, find_procedural_rules_path
from src.neo4j_adapter import Neo4jMemoryAdapter
from src.procedural_matcher import ProceduralRuleMatcher
from src.controller import C3LiteController
from src.baselines import no_memory_baseline, simple_retrieval_baseline, all_memory_baseline
from src.evaluation_utils import score_prediction

METHODS = ['no_memory', 'simple_retrieval', 'all_memory', 'c3_lite_controller']


def select_questions(eval_questions: list[dict[str, Any]], limit: int | None = None, balanced: bool = False) -> list[dict[str, Any]]:
    if limit is None or limit >= len(eval_questions):
        return eval_questions
    if not balanced:
        return eval_questions[:limit]
    groups = {}
    for q in eval_questions:
        groups.setdefault(q['question_type'], []).append(q)
    selected = []
    while len(selected) < limit:
        added = False
        for qtype in sorted(groups):
            if groups[qtype] and len(selected) < limit:
                selected.append(groups[qtype].pop(0)); added = True
        if not added:
            break
    return selected


def run_experiment(output_dir: Path, limit: int | None = None, balanced: bool = False, dataset_path_override: str | None = None, methods: list[str] | None = None) -> dict[str, Any]:
    config = get_config(dataset_path_override)
    eval_path = find_eval_questions_path(config.dataset_path)
    rules_path = find_procedural_rules_path(config.dataset_path)
    eval_questions = read_jsonl(eval_path)
    selected_questions = select_questions(eval_questions, limit=limit, balanced=balanced)
    run_methods = methods or METHODS
    output_dir.mkdir(parents=True, exist_ok=True)
    neo4j = Neo4jMemoryAdapter(config.neo4j_uri, config.neo4j_user, config.neo4j_password)
    procedural = ProceduralRuleMatcher(rules_path)
    controller = C3LiteController(neo4j, procedural, config.answer_mode, config.openai_model, config.openai_api_key)
    predictions, scores = [], []
    try:
        counts = neo4j.smoke_counts()
        for q in tqdm(selected_questions, desc='Running questions'):
            user_id, question = q['user_id'], q['question']
            for method in run_methods:
                start = time.time()
                if method == 'no_memory':
                    pred = no_memory_baseline(question)
                elif method == 'simple_retrieval':
                    pred = simple_retrieval_baseline(neo4j, user_id, question, config.answer_mode, config.openai_model, config.openai_api_key)
                elif method == 'all_memory':
                    pred = all_memory_baseline(neo4j, procedural, user_id, question, config.answer_mode, config.openai_model, config.openai_api_key)
                elif method == 'c3_lite_controller':
                    pred = controller.answer(user_id, question)
                else:
                    raise ValueError(method)
                latency = time.time() - start
                record = {'question_id': q['question_id'], 'user_id': user_id, 'method': method, 'question_type': q['question_type'], 'question': question, 'gold_answer': q.get('gold_answer'), 'expected_route': q.get('expected_route', []), 'required_memory': q.get('required_memory', []), 'supporting_memory_ids': q.get('supporting_memory_ids', []), 'answer_should_include': q.get('answer_should_include', []), 'answer_should_not_include': q.get('answer_should_not_include', []), 'explanation_should_mention': q.get('explanation_should_mention', []), 'rubric_score': q.get('rubric_score'), 'predicted_answer': pred['answer'], 'predicted_query_type': pred['query_type'], 'predicted_route': pred['predicted_route'], 'keywords': pred.get('keywords', []), 'retrieved_memory_ids': pred['retrieved_memory_ids'], 'used_memory_ids': pred['used_memory_ids'], 'outdated_memory_ids': pred['outdated_memory_ids'], 'newer_memory_ids': pred.get('newer_memory_ids', []), 'conflict_notes': pred['conflict_notes'], 'latency_seconds': round(latency, 4)}
                predictions.append(record)
                scores.append(score_prediction(record))
        write_jsonl(output_dir/'predictions.jsonl', predictions)
        write_jsonl(output_dir/'scores.jsonl', scores)
        import pandas as pd
        score_df = pd.DataFrame(scores)
        score_df.to_csv(output_dir/'scores.csv', index=False)
        summary = score_df.groupby('method').agg({'route_exact_match':'mean','route_precision':'mean','route_recall':'mean','route_f1':'mean','evidence_recall':'mean','evidence_precision':'mean','include_score':'mean','not_include_violations':'mean','abstention_heuristic':'mean','latency_seconds':'mean','num_retrieved':'mean','num_used':'mean','num_outdated':'mean'}).reset_index()
        summary.to_csv(output_dir/'summary.csv', index=False)
        run_info = {'dataset_path': str(config.dataset_path), 'eval_questions_path': str(eval_path), 'procedural_rules_path': str(rules_path), 'answer_mode': config.answer_mode, 'methods': run_methods, 'num_questions': len(selected_questions), 'num_predictions': len(predictions), 'neo4j_counts': counts, 'output_dir': str(output_dir)}
        write_json(output_dir/'run_info.json', run_info)
        return run_info
    finally:
        neo4j.close()
