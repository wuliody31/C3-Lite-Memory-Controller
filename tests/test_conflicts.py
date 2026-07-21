from datetime import datetime
from pathlib import Path
from src.config import load_config
from src.conflict_detector import ConflictDetector
from src.conflict_resolver import ConflictResolver
from src.query_analyzer import QueryAnalyzer
from src.schemas import MemoryCandidate, MemoryType

ROOT = Path(__file__).resolve().parents[1]

def test_current_fact_supersedes_old_fact():
    config = load_config(ROOT / "configs/c3_lite_v2_2_final.yaml")
    old = MemoryCandidate("s_old", MemoryType.SEMANTIC, "Project uses NetworkX.", "user01", datetime(2026,6,1), "project", "uses", "NetworkX", "superseded", 0.9)
    new = MemoryCandidate("s_new", MemoryType.SEMANTIC, "Project uses Neo4j.", "user01", datetime(2026,6,20), "project", "uses", "Neo4j", "current", 1.0, authority="supervisor_confirmed")
    conflicts = ConflictDetector(config).detect([old, new])
    assert len(conflicts) == 1
    features = QueryAnalyzer(config).analyse("What database does the project currently use?")
    _, resolved = ConflictResolver(config).resolve(candidates=[old, new], conflicts=conflicts, features=features)
    assert resolved[0].preferred_ids == ["s_new"]
    assert "s_old" in resolved[0].historical_ids
