from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

@dataclass
class AppConfig:
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str | None
    dataset_path: Path
    answer_mode: str
    openai_api_key: str | None
    openai_model: str


def _candidate_dataset_paths() -> list[Path]:
    here = Path(__file__).resolve().parents[1]
    parent = here.parent
    return [
        parent / 'Dataset_A_v0_1',
        parent / 'Dataset_A_v0_1_Neo4j_Episodic_Engineer',
        parent / 'Dataset_A_v0_2',
        parent / 'Dataset_A_v0_2_Neo4j_Expanded',
        parent / 'dataset_A_v0_1',
        parent / 'dataset_A_v0_2_neo4j_episodic',
    ]


def resolve_dataset_path(raw: str | None = None) -> Path:
    if raw:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[1] / p
        if p.exists():
            return p.resolve()
    for p in _candidate_dataset_paths():
        if p.exists():
            return p.resolve()
    raise FileNotFoundError('Cannot find dataset folder. Set DATASET_PATH in .env.')


def get_config(dataset_path_override: str | None = None) -> AppConfig:
    return AppConfig(
        neo4j_uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        neo4j_user=os.getenv('NEO4J_USER', 'neo4j'),
        neo4j_password=os.getenv('NEO4J_PASSWORD'),
        dataset_path=resolve_dataset_path(dataset_path_override or os.getenv('DATASET_PATH')),
        answer_mode=os.getenv('ANSWER_MODE', 'mock').lower(),
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
    )
