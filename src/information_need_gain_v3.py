from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .coverage_estimator import CoverageEstimator
from .schemas import MemoryCandidate
from .text_utils import tokenize


@dataclass(slots=True)
class InformationNeedGainResult:
    memory_id: str

    covered_before: list[int]
    candidate_covered: list[int]
    gained_indices: list[int]
    covered_after: list[int]

    gained_needs: list[str]

    coverage_before: float
    coverage_after: float
    gain: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class InformationNeedMatcherV3:
    """Strict semantic matcher for C3-v3 information-need coverage.

    IMPORTANT:
    This matcher is intentionally separate from the legacy CoverageEstimator.

    The legacy coverage model mixes:
      - loose lexical overlap,
      - temporal-role coverage,
      - procedural/structural coverage.

    That behaviour is useful for broad RC8 coverage estimation, but is too
    permissive for deciding whether one evidence item semantically subsumes
    another information need.

    C3-v3 therefore uses anchor-token coverage:
      1. remove generic temporal/query framing tokens from the need;
      2. require a majority of the remaining semantic anchors to be present;
      3. temporal compatibility is handled elsewhere by
         QueryRelativeTemporalValidity and is NOT allowed to establish
         semantic need coverage by itself.
    """

    STRUCTURAL_NEED_TOKENS = {
        # temporal framing
        "current",
        "currently",
        "present",
        "latest",
        "now",
        "historical",
        "history",
        "previous",
        "previously",
        "earlier",
        "original",
        "originally",
        "past",
        "before",
        "timeline",
        "change",
        "changed",
        "changes",
        "changing",
        "transition",
        "over",
        "time",
        # broad task framing
        "project",
        "state",
        "information",
        "need",
        "needs",
        "answer",
        "target",
        "evidence",
        "support",
        "supporting",
        # Chinese temporal framing
        "当前",
        "现在",
        "目前",
        "最新",
        "历史",
        "之前",
        "以前",
        "最初",
        "变化",
        "改变",
        "演变",
        "更新",
    }

    def __init__(
        self,
        coverage: CoverageEstimator,
        *,
        anchor_ratio: float = 0.60,
    ) -> None:
        if not 0.0 < anchor_ratio <= 1.0:
            raise ValueError(
                "anchor_ratio must be in (0, 1]."
            )

        self.stopwords = set(coverage.stopwords)
        self.anchor_ratio = float(anchor_ratio)

    def candidate_coverage(
        self,
        candidate: MemoryCandidate,
        information_needs: list[str],
    ) -> set[int]:
        candidate_tokens = set(
            tokenize(
                self._candidate_text(candidate),
                self.stopwords,
            )
        )

        covered: set[int] = set()

        for index, need in enumerate(
            information_needs
        ):
            need_tokens = tokenize(
                need,
                self.stopwords,
            )

            anchors = [
                token
                for token in need_tokens
                if token
                not in self.STRUCTURAL_NEED_TOKENS
            ]

            # A need with no semantic anchors is too abstract to use as a
            # semantic-subsumption signal. Role/temporal logic must handle it.
            if not anchors:
                continue

            unique_anchors = set(anchors)
            matched = (
                unique_anchors
                & candidate_tokens
            )

            if len(unique_anchors) == 1:
                required = 1
            else:
                required = max(
                    2,
                    math.ceil(
                        self.anchor_ratio
                        * len(unique_anchors)
                    ),
                )

            if len(matched) >= required:
                covered.add(index)

        return covered

    @staticmethod
    def _candidate_text(
        candidate: MemoryCandidate,
    ) -> str:
        values = [
            candidate.text,
            candidate.subject or "",
            candidate.predicate or "",
            candidate.object_value or "",
        ]

        return " ".join(
            value
            for value in values
            if value
        )


class InformationNeedGainEvaluator:
    """Measure strict marginal semantic information-need coverage.

    This evaluator is shadow-only in M2-C2B.2A.

    It deliberately does NOT call CoverageEstimator.candidate_coverage()
    because that legacy API mixes semantic, temporal and structural coverage.
    """

    def __init__(
        self,
        coverage: CoverageEstimator,
    ) -> None:
        self.matcher = (
            InformationNeedMatcherV3(
                coverage
            )
        )
    def covered_indices(
        self,
        *,
        selected: list[MemoryCandidate],
        information_needs: list[str],
    ) -> set[int]:
        covered: set[int] = set()

        for item in selected:
            covered |= (
                self.matcher.candidate_coverage(
                    item,
                    information_needs,
                )
            )

        return covered

    def coverage_ratio(
        self,
        *,
        selected: list[MemoryCandidate],
        information_needs: list[str],
    ) -> float:
        if not information_needs:
            return 0.0

        covered = self.covered_indices(
            selected=selected,
            information_needs=information_needs,
        )

        return round(
            len(covered)
            / len(information_needs),
            6,
        )
    def evaluate(
        self,
        *,
        candidate: MemoryCandidate,
        selected: list[MemoryCandidate],
        information_needs: list[str],
    ) -> InformationNeedGainResult:

        covered_before_set: set[int] = set()

        for item in selected:
            covered_before_set |= (
                self.matcher.candidate_coverage(
                    item,
                    information_needs,
                )
            )

        candidate_covered_set = (
            self.matcher.candidate_coverage(
                candidate,
                information_needs,
            )
        )

        gained_set = (
            candidate_covered_set
            - covered_before_set
        )

        covered_after_set = (
            covered_before_set
            | candidate_covered_set
        )

        total = len(
            information_needs
        )

        if total > 0:
            coverage_before = (
                len(covered_before_set)
                / total
            )

            coverage_after = (
                len(covered_after_set)
                / total
            )

            gain = (
                len(gained_set)
                / total
            )

        else:
            coverage_before = 0.0
            coverage_after = 0.0
            gain = 0.0

        gained_needs = [
            information_needs[index]
            for index in sorted(
                gained_set
            )
            if (
                0
                <= index
                < len(
                    information_needs
                )
            )
        ]

        return InformationNeedGainResult(
            memory_id=(
                candidate.memory_id
            ),

            covered_before=sorted(
                covered_before_set
            ),

            candidate_covered=sorted(
                candidate_covered_set
            ),

            gained_indices=sorted(
                gained_set
            ),

            covered_after=sorted(
                covered_after_set
            ),

            gained_needs=(
                gained_needs
            ),

            coverage_before=round(
                coverage_before,
                6,
            ),

            coverage_after=round(
                coverage_after,
                6,
            ),

            gain=round(
                gain,
                6,
            ),
        )