from src.slot_fidelity import (
    TransitionSetFidelity,
    TransitionSlotFidelityEvaluator,
)


def path_state(**kwargs):
    fidelity = TransitionSetFidelity(
        available=True,
        reason="test",
        **kwargs,
    )

    return (
        TransitionSlotFidelityEvaluator
        ._temporal_path_state(fidelity)
    )


def test_disjoint_endpoints_enable_reconstruction():
    state = path_state(
        historical_support_ids=("h1",),
        current_support_ids=("c1",),
        transition_support_ids=("t1",),
        best_bridge_fidelity=0.0,
    )

    assert state.endpoint_sets_disjoint is True
    assert state.endpoint_reconstruction_available is True
    assert (
        state.recoverability_status
        == "endpoint_reconstruction_available"
    )


def test_endpoint_overlap_blocks_reconstruction():
    state = path_state(
        historical_support_ids=("same",),
        current_support_ids=("same",),
        transition_support_ids=("t1",),
        best_bridge_fidelity=0.2,
    )

    assert state.endpoint_sets_disjoint is False
    assert state.endpoint_reconstruction_available is False
    assert state.endpoint_overlap_ids == ("same",)
    assert (
        state.recoverability_status
        == "transition_only_requires_fidelity_judgement"
    )


def test_missing_current_endpoint_is_transition_only():
    state = path_state(
        historical_support_ids=("h1",),
        current_support_ids=(),
        transition_support_ids=("t1",),
        best_bridge_fidelity=0.3,
    )

    assert state.endpoint_reconstruction_available is False
    assert state.explicit_transition_support_available is True
    assert (
        state.recoverability_status
        == "transition_only_requires_fidelity_judgement"
    )


def test_no_endpoint_path_and_no_transition_is_unavailable():
    state = path_state(
        historical_support_ids=("h1",),
        current_support_ids=(),
        transition_support_ids=(),
    )

    assert state.endpoint_reconstruction_available is False
    assert state.explicit_transition_support_available is False
    assert (
        state.recoverability_status
        == "no_structural_temporal_support"
    )


def test_non_applicable_transition_set_remains_non_applicable():
    fidelity = TransitionSetFidelity(
        available=False,
        reason="no_temporal_transition_slot",
    )

    state = (
        TransitionSlotFidelityEvaluator
        ._temporal_path_state(fidelity)
    )

    assert state.applicable is False
    assert state.recoverability_status == "not_applicable"
