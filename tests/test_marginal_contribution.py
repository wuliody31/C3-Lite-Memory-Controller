from types import SimpleNamespace

from src.marginal_contribution import (
    MarginalContributionEvaluator,
)
from src.slot_fidelity import TemporalPathState


def result(
    *,
    coverage=1.0,
    sufficient=True,
):
    return SimpleNamespace(
        hard_slot_coverage=coverage,
        slot_sufficient=sufficient,
    )


def slot_dict(
    *,
    slot_id="slot_1",
    kind="CONTENT",
    complete=True,
    count=1,
    ids=("m1",),
):
    return {
        "slot_statuses": [
            {
                "slot_id": slot_id,
                "kind": kind,
                "complete": complete,
                "satisfied_count": count,
                "supporting_memory_ids": list(ids),
            }
        ]
    }


def path(
    *,
    applicable=False,
    endpoint=False,
    transition=False,
):
    return TemporalPathState(
        applicable=applicable,
        reason="test",
        endpoint_sets_disjoint=endpoint,
        endpoint_reconstruction_available=endpoint,
        explicit_transition_support_available=transition,
    )


def evaluator():
    return MarginalContributionEvaluator(
        sufficiency=None,
        transition_fidelity=None,
    )


def test_last_hard_slot_support_is_critical():
    item = evaluator()._compare(
        memory_id="m1",

        before=result(
            coverage=1.0,
            sufficient=True,
        ),
        before_dict=slot_dict(
            complete=True,
            count=1,
            ids=("m1",),
        ),
        before_path=path(),

        after=result(
            coverage=0.0,
            sufficient=False,
        ),
        after_dict=slot_dict(
            complete=False,
            count=0,
            ids=(),
        ),
        after_path=path(),

        hard_slot_ids={"slot_1"},
    )

    assert item.criticality_class == "critical"
    assert item.hard_slots_lost == ("slot_1",)
    assert "hard_slot_destroyed" in item.reasons


def test_redundant_support_reduction_is_contributory():
    item = evaluator()._compare(
        memory_id="m1",

        before=result(),
        before_dict=slot_dict(
            complete=True,
            count=2,
            ids=("m1", "m2"),
        ),
        before_path=path(),

        after=result(),
        after_dict=slot_dict(
            complete=True,
            count=1,
            ids=("m2",),
        ),
        after_path=path(),

        hard_slot_ids={"slot_1"},
    )

    assert item.criticality_class == "contributory"
    assert (
        item.support_count_drop_slots
        == ("slot_1",)
    )


def test_irrelevant_removal_is_redundant():
    item = evaluator()._compare(
        memory_id="noise",

        before=result(),
        before_dict=slot_dict(
            complete=True,
            count=1,
            ids=("m1",),
        ),
        before_path=path(),

        after=result(),
        after_dict=slot_dict(
            complete=True,
            count=1,
            ids=("m1",),
        ),
        after_path=path(),

        hard_slot_ids={"slot_1"},
    )

    assert item.criticality_class == "redundant"
    assert (
        "no_requirement_or_path_state_changed"
        in item.reasons
    )


def test_endpoint_path_destruction_is_critical():
    item = evaluator()._compare(
        memory_id="current_endpoint",

        before=result(),
        before_dict=slot_dict(),
        before_path=path(
            applicable=True,
            endpoint=True,
            transition=True,
        ),

        after=result(),
        after_dict=slot_dict(),
        after_path=path(
            applicable=True,
            endpoint=False,
            transition=True,
        ),

        hard_slot_ids={"slot_1"},
    )

    assert item.criticality_class == "critical"
    assert item.endpoint_path_lost is True
    assert (
        "endpoint_reconstruction_destroyed"
        in item.reasons
    )


def test_transition_loss_with_endpoint_path_preserved_is_contributory():
    item = evaluator()._compare(
        memory_id="transition",

        before=result(),
        before_dict=slot_dict(),
        before_path=path(
            applicable=True,
            endpoint=True,
            transition=True,
        ),

        after=result(),
        after_dict=slot_dict(),
        after_path=path(
            applicable=True,
            endpoint=True,
            transition=False,
        ),

        hard_slot_ids={"slot_1"},
    )

    assert item.endpoint_path_lost is False
    assert item.transition_support_lost is True
    assert item.criticality_class == "contributory"



def test_temporal_hard_slot_loss_is_substituted_by_endpoint_path():
    item = evaluator()._compare(
        memory_id="transition",

        before=result(
            coverage=1.0,
            sufficient=True,
        ),
        before_dict=slot_dict(
            slot_id="transition_slot",
            kind="TEMPORAL_TRANSITION",
            complete=True,
            count=1,
            ids=("transition",),
        ),
        before_path=path(
            applicable=True,
            endpoint=True,
            transition=True,
        ),

        after=result(
            coverage=0.0,
            sufficient=False,
        ),
        after_dict=slot_dict(
            slot_id="transition_slot",
            kind="TEMPORAL_TRANSITION",
            complete=False,
            count=0,
            ids=(),
        ),
        after_path=path(
            applicable=True,
            endpoint=True,
            transition=False,
        ),

        hard_slot_ids={
            "transition_slot"
        },
    )

    assert item.hard_slots_lost == (
        "transition_slot",
    )

    assert (
        item.effective_hard_slots_lost
        == ()
    )

    assert (
        item.path_substituted_hard_slots
        == ("transition_slot",)
    )

    assert item.endpoint_path_lost is False

    assert (
        item.criticality_class
        == "contributory"
    )
