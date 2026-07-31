from __future__ import annotations

from pathlib import Path

from src.config import load_config
from src.query_analyzer import QueryAnalyzer
from src.route_planner import RoutePlanner
from src.schemas import MemoryType


ROOT = Path(__file__).resolve().parents[1]


def config() -> dict:
    return load_config(
        ROOT / "configs" / "c3_lite_v2_2_final.yaml"
    )


def test_selected_baselines_query_is_set_query_not_timeline() -> None:
    cfg = config()

    features = QueryAnalyzer(cfg).analyse(
        "What baselines have I selected "
        "for the project evaluation?"
    )
    route = RoutePlanner(cfg).plan(features)

    assert set(route.selected_types) == {
        MemoryType.EPISODIC,
        MemoryType.SEMANTIC,
    }

    assert "earlier state" not in features.information_needs
    assert "current state" not in features.information_needs

    assert any(
        "baseline" in need.lower()
        for need in features.information_needs
    )
