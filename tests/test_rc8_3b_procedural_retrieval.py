from __future__ import annotations

from types import SimpleNamespace

from src.retrievers.procedural_json import ProceduralJsonStore
from src.schemas import MemoryType


RULES = [
    {
        "rule_id": "p_project_conflict",
        "user_id": "user01",
        "name": "project_conflict_policy",
        "condition": "When earlier and later project decisions conflict",
        "action": (
            "Prefer the later supervisor-confirmed decision while "
            "describing the older memory as historical context."
        ),
        "instruction": (
            "Prefer the later supervisor-confirmed decision while "
            "describing the older memory as historical context."
        ),
        "scope": "project",
        "priority": 5,
    },
    {
        "rule_id": "p_academic_style",
        "user_id": "user01",
        "name": "academic_supervisor_style",
        "condition": (
            "When the user asks about dissertation or project planning"
        ),
        "action": (
            "Use structured academic language suitable for supervisor "
            "discussion."
        ),
        "instruction": (
            "Use structured academic language suitable for supervisor "
            "discussion."
        ),
        "scope": "project",
        "priority": 5,
    },
    {
        "rule_id": "p_cv_style",
        "user_id": "user02",
        "name": "cv_hr_style",
        "condition": "When writing CV or application wording",
        "action": (
            "Use concise HR-facing wording supported by concrete "
            "technical details."
        ),
        "instruction": (
            "Use concise HR-facing wording supported by concrete "
            "technical details."
        ),
        "scope": "project",
        "priority": 5,
    },
    {
        "rule_id": "p_no_claims",
        "user_id": "user02",
        "name": "no_unsupported_claims",
        "condition": (
            "When CV claims are not supported by memory evidence"
        ),
        "action": (
            "Avoid strong claims such as production deployment unless "
            "evidence exists."
        ),
        "instruction": (
            "Avoid strong claims such as production deployment unless "
            "evidence exists."
        ),
        "scope": "project",
        "priority": 5,
    },
    {
        "rule_id": "p_travel_steps",
        "user_id": "user03",
        "name": "travel_step_by_step",
        "condition": (
            "When explaining buses, trains, tickets or booking interfaces"
        ),
        "action": (
            "Give concrete step-by-step instructions and exact wording "
            "where useful."
        ),
        "instruction": (
            "Give concrete step-by-step instructions and exact wording "
            "where useful."
        ),
        "scope": "project",
        "priority": 5,
    },
    {
        "rule_id": "p_location",
        "user_id": "user03",
        "name": "current_location_rule",
        "condition": (
            "When giving restaurant or route recommendations"
        ),
        "action": (
            "Use the latest hotel location rather than an old hotel "
            "location."
        ),
        "instruction": (
            "Use the latest hotel location rather than an old hotel "
            "location."
        ),
        "scope": "project",
        "priority": 5,
    },
    {
        "rule_id": "p_unrelated_high_priority",
        "user_id": "user03",
        "name": "unrelated_code_policy",
        "condition": "When reviewing Java unit tests",
        "action": "Use strict type checking.",
        "instruction": "Use strict type checking.",
        "scope": "task",
        "priority": 10,
    },
]


def features(
    query: str,
    *,
    task_type: str | None = None,
    asks_conflict: bool = False,
    asks_explanation: bool = False,
    asks_procedure: bool = True,
):
    return SimpleNamespace(
        normalised_query=query.lower(),
        task_type=task_type,
        asks_conflict=asks_conflict,
        asks_explanation=asks_explanation,
        asks_procedure=asks_procedure,
    )


def retrieve_ids(
    store: ProceduralJsonStore,
    *,
    user_id: str,
    query: str,
    task_type: str | None = None,
    asks_conflict: bool = False,
    asks_explanation: bool = False,
) -> list[str]:
    output = store.retrieve(
        memory_type=MemoryType.PROCEDURAL,
        state=SimpleNamespace(user_id=user_id),
        features=features(
            query,
            task_type=task_type,
            asks_conflict=asks_conflict,
            asks_explanation=asks_explanation,
        ),
        top_k=15,
    )
    return [item.memory_id for item in output]


def test_conflict_intent_rescues_project_conflict_policy() -> None:
    store = ProceduralJsonStore(RULES)

    ids = retrieve_ids(
        store,
        user_id="user01",
        query=(
            "Is my project currently a large multi-agent system "
            "or a smaller controller prototype?"
        ),
        asks_conflict=True,
    )

    assert "p_project_conflict" in ids


def test_academic_concept_expansion_recovers_style_rule() -> None:
    store = ProceduralJsonStore(RULES)

    ids = retrieve_ids(
        store,
        user_id="user01",
        query=(
            "Please explain why my dataset should not be centred "
            "on Neo4j."
        ),
        asks_explanation=True,
    )

    assert "p_academic_style" in ids


def test_cv_hr_wording_recovers_cv_style_rule() -> None:
    store = ProceduralJsonStore(RULES)

    ids = retrieve_ids(
        store,
        user_id="user02",
        query=(
            "Answer in HR-friendly language: what is my strongest "
            "project direction?"
        ),
        task_type="cv_writing",
    )

    assert "p_cv_style" in ids


def test_condition_and_action_fields_recover_no_claims_rule() -> None:
    store = ProceduralJsonStore(RULES)

    ids = retrieve_ids(
        store,
        user_id="user02",
        query=(
            "If a CV claim is not stored in memory, how should "
            "the system behave?"
        ),
        task_type="cv_writing",
    )

    assert "p_no_claims" in ids


def test_booking_and_transport_queries_recover_travel_rule() -> None:
    store = ProceduralJsonStore(RULES)

    for query in (
        "How should you answer my bus or train questions?",
        "What should I do if booking information is unclear?",
    ):
        ids = retrieve_ids(
            store,
            user_id="user03",
            query=query,
            task_type="travel_planning",
        )
        assert "p_travel_steps" in ids


def test_food_recommendation_recovers_location_rule() -> None:
    store = ProceduralJsonStore(RULES)

    ids = retrieve_ids(
        store,
        user_id="user03",
        query=(
            "Explain which memory supports using city centre "
            "for food recommendations."
        ),
        task_type="travel_planning",
        asks_explanation=True,
    )

    assert "p_location" in ids


def test_priority_and_scope_alone_do_not_admit_unrelated_rule() -> None:
    store = ProceduralJsonStore(RULES)

    ids = retrieve_ids(
        store,
        user_id="user03",
        query="What should I do if booking information is unclear?",
        task_type="travel_planning",
    )

    assert "p_unrelated_high_priority" not in ids


def test_user_isolation_is_preserved() -> None:
    store = ProceduralJsonStore(RULES)

    ids = retrieve_ids(
        store,
        user_id="user01",
        query="Use HR-friendly CV wording.",
        task_type="cv_writing",
    )

    assert "p_cv_style" not in ids


def test_retrieved_rule_exposes_field_level_audit_metadata() -> None:
    store = ProceduralJsonStore(RULES)

    output = store.retrieve(
        memory_type=MemoryType.PROCEDURAL,
        state=SimpleNamespace(user_id="user03"),
        features=features(
            "What should I do if booking information is unclear?",
            task_type="travel_planning",
        ),
        top_k=15,
    )
    item = next(
        candidate
        for candidate in output
        if candidate.memory_id == "p_travel_steps"
    )

    assert item.metadata["retrieval_version"] == "rc8.3b"
    assert item.metadata["direct_match"] is True
    assert item.metadata["condition_match"] > 0.0
    assert item.metadata["preliminary_procedural_score"] > 0.0
    assert "travel_transport" in item.metadata["derived_concepts"]
    assert item.metadata["triggers"]
