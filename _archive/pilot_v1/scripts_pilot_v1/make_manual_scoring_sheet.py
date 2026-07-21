from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', default='outputs/pilot/predictions.jsonl')
    parser.add_argument('--out', default='outputs/pilot/manual_scoring_sheet.csv')
    args = parser.parse_args()
    rows = read_jsonl(Path(args.predictions))
    sheet_rows = []
    for r in rows:
        sheet_rows.append({'question_id': r['question_id'], 'method': r['method'], 'question_type': r['question_type'], 'question': r['question'], 'gold_answer': r['gold_answer'], 'predicted_answer': r['predicted_answer'], 'manual_score_0_10': '', 'comments': ''})
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sheet_rows).to_csv(out, index=False)
    print(f'Saved manual scoring sheet to {out}')

if __name__ == '__main__':
    main()
