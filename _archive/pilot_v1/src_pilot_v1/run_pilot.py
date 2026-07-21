from __future__ import annotations
import argparse
from pathlib import Path
from src.experiment_runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description='Run pilot experiment on a small subset.')
    parser.add_argument('--dataset-path', default=None)
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--balanced', action='store_true')
    parser.add_argument('--output-dir', default='outputs/pilot')
    args = parser.parse_args()
    info = run_experiment(Path(args.output_dir), args.limit, args.balanced, args.dataset_path)
    print('Pilot completed.')
    print(info)

if __name__ == '__main__':
    main()
