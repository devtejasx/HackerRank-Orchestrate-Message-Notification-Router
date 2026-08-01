"""Historical behaviour features, derived from past messages and reactions.

All counts are **recipient-scoped**: "messages from this sender" means
messages from that sender *to this user*, not that sender's global volume.
Phase 3 personalisation needs the per-user view, and the global view is
already available as ``group_message_volume_30d`` in the context block.

Per-user aggregates are memoised on the extractor instance, so processing a
whole dataset stays linear even though several messages share a recipient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from src.data.models import GroupMember, Message, MessageEvent, UserBusinessHistory
from src.data.repository import DataRepository
from src.features.feature_models import HistoricalFeatures
from src.utils.helpers import clamp, ratio

__all__ = ["EngagementWeights", "HistoricalFeatureExtractor"]


@dataclass(frozen=True, slots=True)
class EngagementWeights:
    """Weights combining interaction rates into a single engagement score.

    Positive weights reward attention, negative weights penalise rejection.
    Scores are clamped to ``[0, 1]`` after weighting, so the weights need not
    sum to one.
    """

    opened: float = 0.50
    replied: float = 0.50
    dismissed: float = 0.30
    reported: float = 0.60
    muted: float = 0.40

    #: Replies at which the reply component of relationship engagement saturates.
    reply_saturation: int = 3

    #: Group replies at which the group reply component saturates.
    group_reply_saturation: int = 5

    #: Penalty applied to group engagement when the user has muted the group.
    group_mute_penalty: float = 0.30


DEFAULT_ENGAGEMENT_WEIGHTS: Final[EngagementWeights] = EngagementWeights()

#: Split between "did they look at it" and "did they answer it" when scoring a
#: business relationship. Attention dominates because most business messages
#: are transactional and never warrant a reply.
ATTENTION_WEIGHT: Final[float] = 0.70
RESPONSIVENESS_WEIGHT: Final[float] = 0.30

#: The same split for group membership, where replying is a stronger signal of
#: involvement, so responsiveness carries more weight.
GROUP_ATTENTION_WEIGHT: Final[float] = 0.60
GROUP_RESPONSIVENESS_WEIGHT: Final[float] = 0.40


@dataclass(frozen=True, slots=True)
class _InteractionRates:
    """Aggregated reaction rates for one user across all their history."""

    total: int
    open_rate: float
    reply_rate: float
    dismiss_rate: float
    report_rate: float
    mute_rate: float
    engagement: float


class HistoricalFeatureExtractor:
    """Builds :class:`HistoricalFeatures`, memoising per-user aggregates.

    Args:
        repo: Loaded repository.
        weights: Engagement weighting. Override to retune without editing code.

    Example:
        >>> extractor = HistoricalFeatureExtractor(repo)     # doctest: +SKIP
        >>> extractor.extract(message).open_rate             # doctest: +SKIP
        0.62
    """

    def __init__(
        self,
        repo: DataRepository,
        weights: EngagementWeights = DEFAULT_ENGAGEMENT_WEIGHTS,
    ) -> None:
        self._repo = repo
        self._weights = weights
        self._rates_cache: dict[str, _InteractionRates] = {}
        self._sender_counts: dict[tuple[str, str], int] = {}
        self._group_counts: dict[tuple[str, str], int] = {}
        self._business_counts: dict[tuple[str, str], int] = {}

    def extract(self, message: Message) -> HistoricalFeatures:
        """Build the historical feature block for one message.

        Args:
            message: The message being analysed.

        Returns:
            A fully populated, immutable feature block. Rates are ``0.0`` when
            the user has no history; use
            :attr:`HistoricalFeatures.has_history` to distinguish that from
            genuine disengagement.
        """
        user_id = message.user_id
        rates = self._rates_for(user_id)
        relationship = (
            self._repo.get_user_business(user_id, message.business_id)
            if message.business_id
            else None
        )
        membership = (
            self._repo.get_group_member(message.group_id, user_id)
            if message.group_id
            else None
        )

        return HistoricalFeatures(
            sender_message_count=self._count_from_sender(user_id, message.sender_user_id),
            group_message_count=self._count_from_group(user_id, message.group_id),
            business_message_count=self._count_from_business(user_id, message.business_id),
            total_interactions=rates.total,
            open_rate=rates.open_rate,
            reply_rate=rates.reply_rate,
            dismiss_rate=rates.dismiss_rate,
            report_rate=rates.report_rate,
            mute_rate=rates.mute_rate,
            user_engagement=rates.engagement,
            business_engagement=self._business_engagement(relationship),
            group_engagement=self._group_engagement(membership),
            has_business_relationship=relationship is not None,
            allows_promotions=relationship.allows_promotions if relationship else None,
            opted_out_of_promotions=relationship.has_opted_out if relationship else False,
        )

    # ------------------------------------------------------------------ #
    # Per-user reaction rates
    # ------------------------------------------------------------------ #

    def _rates_for(self, user_id: str) -> _InteractionRates:
        """Return cached reaction rates for ``user_id``."""
        cached = self._rates_cache.get(user_id)
        if cached is not None:
            return cached

        events = self._repo.get_user_events(user_id)
        rates = self._aggregate(events)
        self._rates_cache[user_id] = rates
        return rates

    def _aggregate(self, events: tuple[MessageEvent, ...]) -> _InteractionRates:
        """Reduce a user's interaction events to rates and an engagement score."""
        total = len(events)
        if total == 0:
            return _InteractionRates(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        open_rate = ratio(sum(e.message_opened for e in events), total)
        reply_rate = ratio(sum(e.message_replied for e in events), total)
        dismiss_rate = ratio(sum(e.notification_dismissed for e in events), total)
        report_rate = ratio(sum(e.message_reported for e in events), total)
        mute_rate = ratio(sum(e.muted_after_message for e in events), total)

        weights = self._weights
        engagement = clamp(
            weights.opened * open_rate
            + weights.replied * reply_rate
            - weights.dismissed * dismiss_rate
            - weights.reported * report_rate
            - weights.muted * mute_rate
        )
        return _InteractionRates(
            total=total,
            open_rate=open_rate,
            reply_rate=reply_rate,
            dismiss_rate=dismiss_rate,
            report_rate=report_rate,
            mute_rate=mute_rate,
            engagement=engagement,
        )

    # ------------------------------------------------------------------ #
    # Relationship-scoped engagement
    # ------------------------------------------------------------------ #

    def _business_engagement(self, relationship: UserBusinessHistory | None) -> float:
        """Score how much this user engages with this business, in ``[0, 1]``.

        ``0.0`` when no relationship exists, which the classifier reads
        alongside :attr:`HistoricalFeatures.has_business_relationship`.
        """
        if relationship is None:
            return 0.0

        weights = self._weights
        attention = ratio(
            relationship.messages_opened_30d,
            relationship.messages_opened_30d + relationship.messages_dismissed_30d,
        )
        responsiveness = min(
            ratio(relationship.messages_replied_30d, weights.reply_saturation), 1.0
        )
        return clamp(
            ATTENTION_WEIGHT * attention + RESPONSIVENESS_WEIGHT * responsiveness
        )

    def _group_engagement(self, membership: GroupMember | None) -> float:
        """Score how much this user engages with this group, in ``[0, 1]``."""
        if membership is None:
            return 0.0

        weights = self._weights
        attention = ratio(
            membership.messages_read_30d,
            membership.messages_read_30d + membership.notifications_dismissed_30d,
        )
        responsiveness = min(
            ratio(membership.replies_sent_30d, weights.group_reply_saturation), 1.0
        )
        score = (
            GROUP_ATTENTION_WEIGHT * attention
            + GROUP_RESPONSIVENESS_WEIGHT * responsiveness
        )
        if membership.group_muted_by_user:
            score -= weights.group_mute_penalty
        return clamp(score)

    # ------------------------------------------------------------------ #
    # Recipient-scoped history counts
    # ------------------------------------------------------------------ #

    def _count_from_sender(self, user_id: str, sender_id: str | None) -> int:
        """Past messages this user received from ``sender_id``."""
        if sender_id is None:
            return 0
        key = (user_id, sender_id)
        cached = self._sender_counts.get(key)
        if cached is None:
            history = self._repo.get_sender_history(sender_id)
            cached = sum(1 for record in history if record.user_id == user_id)
            self._sender_counts[key] = cached
        return cached

    def _count_from_group(self, user_id: str, group_id: str | None) -> int:
        """Past messages this user received through ``group_id``."""
        if group_id is None:
            return 0
        key = (user_id, group_id)
        cached = self._group_counts.get(key)
        if cached is None:
            history = self._repo.get_group_history(group_id)
            cached = sum(1 for record in history if record.user_id == user_id)
            self._group_counts[key] = cached
        return cached

    def _count_from_business(self, user_id: str, business_id: str | None) -> int:
        """Past messages this user received from ``business_id``."""
        if business_id is None:
            return 0
        key = (user_id, business_id)
        cached = self._business_counts.get(key)
        if cached is None:
            history = self._repo.get_business_history(business_id)
            cached = sum(1 for record in history if record.user_id == user_id)
            self._business_counts[key] = cached
        return cached
