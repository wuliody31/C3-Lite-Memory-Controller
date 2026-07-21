from pathlib import Path
from src.backbones import MockBackbone
from src.config import load_config
from src.pipeline import C3Pipeline
from src.retrievers import InMemoryMemoryStore, ProceduralJsonStore
from src.schemas import QueryState

ROOT = Path(__file__).resolve().parents[1]

def pipeline():
    config = load_config(ROOT / "configs/c3_lite_v2_2_final.yaml")
    stopwords = set(config["query_analysis"]["stopwords"])
    return C3Pipeline(config=config, memory_store=InMemoryMemoryStore.from_json_file(str(ROOT / "examples/demo_memories.json"), stopwords), procedure_store=ProceduralJsonStore.from_file(str(ROOT / "examples/demo_procedures.json"), stopwords), backbone=MockBackbone(), prompt_template=ROOT / "prompts/answer_prompt.txt")

def test_timeline_routes_two_memories():
    result = pipeline().answer(QueryState("How did my project scope change over time?", "user01"))
    assert "episodic" in result.selected_memory_types
    assert "semantic" in result.selected_memory_types
    assert len(result.selected_ids) >= 2

def test_procedural_routing():
    result = pipeline().answer(QueryState("When discussing my dissertation with my supervisor, how should the answer style be?", "user01"))
    assert "procedural" in result.selected_memory_types
    assert any(x.startswith("p_") for x in result.selected_ids)
