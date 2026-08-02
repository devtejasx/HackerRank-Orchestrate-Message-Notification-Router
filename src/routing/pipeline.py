"""The complete system: a message in, a routing decision out.

    Repository -> Feature extraction -> Classification -> Routing signals
      -> Decision -> Evidence -> Reason -> Confidence -> RoutingResult

:class:`RoutingPipeline` wraps the existing :class:`~src.pipeline.MessagePipeline`
rather than reimplementing it, so Phases 1-3 keep their single entry point and
this class only adds the final stage.

Phase 5 has one job left: write :meth:`RoutingResult.to_output_row` to CSV.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Self

from src import config
from src.data.models import MessageRecord
from src.data.repository import DataRepository
from src.media.understanding import MediaUnderstanding
from src.pipeline import MessageAnalysis, MessagePipeline
from src.routing.evidence import EvidenceEngine
from src.routing.models import DecisionContext, RoutingResult
from src.routing.router import Router

__all__ = ["RoutingPipeline"]

_LOGGER = config.get_logger("routing.pipeline")


class RoutingPipeline:
    """Runs every phase and returns the final routing decision.

    Args:
        repo: A loaded repository.
        analysis_pipeline: The Phase 1-3 pipeline. One is built when omitted.
        router: The Phase 4 router. One is built when omitted, sharing the
            analysis pipeline's feature extractor and classifier so history
            classification reuses their caches.

    Example:
        >>> pipeline = RoutingPipeline.load()          # doctest: +SKIP
        >>> results = pipeline.route_all()             # doctest: +SKIP
        >>> results[0].to_output_row()["action"]       # doctest: +SKIP
        'mute'
    """

    def __init__(
        self,
        repo: DataRepository,
        analysis_pipeline: MessagePipeline | None = None,
        router: Router | None = None,
        understanding: MediaUnderstanding | None = None,
    ) -> None:
        self._repo = repo
        self._analysis = analysis_pipeline or MessagePipeline(
            repo, understanding=understanding
        )
        if self._analysis.engine is None:
            raise ValueError(
                "RoutingPipeline needs routing signals; construct the analysis "
                "pipeline with personalize=True"
            )
        self._router = router or Router(
            evidence_engine=EvidenceEngine(
                repo,
                extractor=self._analysis.extractor,
                classifier=self._analysis.classifier,
            )
        )

    @classmethod
    def load(
        cls,
        dataset_dir: Path | None = None,
        understanding: MediaUnderstanding | None = None,
    ) -> Self:
        """Load the dataset and build a ready-to-use routing pipeline.

        Args:
            dataset_dir: Dataset directory. Defaults to
                :data:`src.config.DATASET_DIR`.
            understanding: OCR / speech-to-text provider. This argument is the
                whole multimodal integration surface: pass one and recovered
                text flows through classification, routing and evidence with no
                other change. See :mod:`src.media`.

        Returns:
            A pipeline over a loaded, validated and indexed repository.
        """
        return cls(DataRepository.load(dataset_dir), understanding=understanding)

    @property
    def repository(self) -> DataRepository:
        """The underlying repository."""
        return self._repo

    @property
    def analysis(self) -> MessagePipeline:
        """The Phase 1-3 pipeline."""
        return self._analysis

    @property
    def router(self) -> Router:
        """The Phase 4 router."""
        return self._router

    def route(self, message: MessageRecord) -> RoutingResult:
        """Run every phase over one message.

        Args:
            message: The incoming message.

        Returns:
            The final routing result.
        """
        return self.route_analysis(self._analysis.analyse(message))

    def route_analysis(self, analysis: MessageAnalysis) -> RoutingResult:
        """Route a message that has already been through Phases 1-3.

        Useful when the analysis is needed for something else too, so it is
        not recomputed.

        Args:
            analysis: A completed analysis carrying routing signals.

        Returns:
            The final routing result.

        Raises:
            ValueError: If the analysis has no routing signals attached.
        """
        if analysis.routing is None:
            raise ValueError(
                f"{analysis.message_id} has no routing signals; "
                "run the analysis pipeline with personalize=True"
            )
        context = DecisionContext(
            features=analysis.features,
            classification=analysis.classification,
            signals=analysis.routing,
            repo=self._repo,
        )
        return self._router.route(context)

    def route_many(self, messages: Iterable[MessageRecord]) -> tuple[RoutingResult, ...]:
        """Route many messages, preserving input order."""
        results = tuple(self.route(message) for message in messages)
        _LOGGER.info("Routed %d message(s)", len(results))
        return results

    def route_all(self) -> tuple[RoutingResult, ...]:
        """Route every incoming message in the dataset.

        Returns:
            One result per row of ``messages.csv``, in dataset order - exactly
            what the submission format requires.
        """
        return self.route_many(self._repo.get_messages())
