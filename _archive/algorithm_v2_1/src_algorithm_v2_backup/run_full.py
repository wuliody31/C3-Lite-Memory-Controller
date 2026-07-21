from __future__ import annotations
import argparse
from pathlib import Path
from src.experiment_runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description='Run full experiment on all eval questions.')
    parser.add_argument('--dataset-path', default=None)
    parser.add_argument('--output-dir', default='outputs/full')
    args = parser.parse_args()
    info = run_experiment(Path(args.output_dir), None, False, args.dataset_path)
    print('Full experiment completed.')
    print(info)

if __name__ == '__main__':
    main()
