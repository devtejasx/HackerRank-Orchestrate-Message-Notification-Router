"""Scoped interaction statistics: how a user treated one counterparty's messages.

Phase 2 already computes reaction rates across a user's whole history and
Phase 3 reuses those directly. What Phase 2 does not provide, because it did
not need it, is the same rates *scoped to one sender, business or group* -
which is exactly what personalisation turns on. This module supplies that, once,
for every scope, so the sender, business and group calculators share one
implementation rather than three near-identical ones.

Every result is memoised per ``(scope, id, user)``, so analysing a whole
dataset stays linear even though many messages share a counterparty.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Final

from src.data.models import MessageEvent, MessageHistory
from src.data.repository import DataRepository
from src.personalization.normalization import NEUTRAL, trend_score
from src.utils.helpers import ratio

__all__ = ["InteractionScope", "InteractionStats", "InteractionStatsProvider"]

#: Minimum records before a trend can be computed. Below this, one message
#: flipping would swing the trend entirely, so neutral is reported instead.
MIN_RECORDS_FOR_TREND: Final[int] = 4


class InteractionScope(StrEnum):
    """Which counterparty a set of statistics is scoped to."""

    USER = "user"
    SENDER = "sender"
    BUSINESS = "business"
    GROUP = "group"


@dataclass(frozen=True, slots=True)
class InteractionStats:
    """How a user reacted to one counterparty's past messages.

    All rates are in ``[0, 1]`` and are ``0.0`` when there is no history; read
    them alongside :attr:`total` to tell "never happened" apart from "happened
    and went badly".

    Attributes:
        scope: What the statistics are scoped to.
        total: Number of past messages in scope.
        open_rate: Fraction the user opened.
        reply_rate: Fraction the user replied to.
        dismiss_rate: Fraction whose notification the user dismissed.
        report_rate: Fraction the user reported.
        mute_rate: Fraction after which the user muted the conversation.
        last_seen: Timestamp of the most recent message in scope.
        first_seen: Timestamp of the earliest message in scope.
        engagement_trend: Recent engagement against earlier engagement, with
            ``0.5`` meaning flat or not enough data.
    """

    scope: InteractionScope
    total: int
    open_rate: float
    reply_rate: float
    dismiss_rate: float
    report_rate: float
    mute_rate: float
    last_seen: datetime | None
    first_seen: datetime | None
    engagement_trend: float

    @property
    def has_history(self) -> bool:
        """Whether any past message exists in this scope."""
        return self.total > 0

    @property
    def engagement(self) -> float:
        """Attention and response combined, in ``[0, 1]``.

        An unweighted mean of open and reply rate: opening shows the message
        got through, replying shows it mattered.
        """
        return (self.open_rate + self.reply_rate) / 2.0

    @property
    def rejection(self) -> float:
        """Dismissal, muting and reporting combined, in ``[0, 1]``."""
        return max(self.dismiss_rate, self.mute_rate, self.report_rate)

    @property
    def span_days(self) -> float:
        """Days between the first and last message in scope."""
        if self.first_seen is None or self.last_seen is None:
            return 0.0
        return max(0.0, (self.last_seen - self.first_seen).total_seconds() / 86_400.0)


#: Statistics for a scope with no history at all.
def _empty(scope: InteractionScope) -> InteractionStats:
    """Return zeroed statistics for a scope with no records."""
    return InteractionStats(
        scope=scope,
        total=0,
        open_rate=0.0,
        reply_rate=0.0,
        dismiss_rate=0.0,
        report_rate=0.0,
        mute_rate=0.0,
        last_seen=None,
        first_seen=None,
        engagement_trend=NEUTRAL,
    )


class InteractionStatsProvider:
    """Builds and caches :class:`InteractionStats` for each scope.

    Args:
        repo: A loaded repository. Every lookup goes through it.
    """

    def __init__(self, repo: DataRepository) -> None:
        self._repo = repo
        self._cache: dict[tuple[InteractionScope, str, str], InteractionStats] = {}

    def for_user(self, user_id: str) -> InteractionStats:
        """Statistics across everything this user has received."""
        return self._cached(
            InteractionScope.USER,
            user_id,
            user_id,
            lambda: self._repo.get_user_history(user_id),
        )

    def for_sender(self, user_id: str, sender_id: str | None) -> InteractionStats:
        """Statistics for messages this user received from ``sender_id``."""
        if sender_id is None:
            return _empty(InteractionScope.SENDER)
        return self._cached(
            InteractionScope.SENDER,
            sender_id,
            user_id,
            lambda: self._scoped(self._repo.get_sender_history(sender_id), user_id),
        )

    def for_business(self, user_id: str, business_id: str | None) -> InteractionStats:
        """Statistics for messages this user received from ``business_id``."""
        if business_id is None:
            return _empty(InteractionScope.BUSINESS)
        return self._cached(
            InteractionScope.BUSINESS,
            business_id,
            user_id,
            lambda: self._scoped(self._repo.get_business_history(business_id), user_id),
        )

    def for_group(self, user_id: str, group_id: str | None) -> InteractionStats:
        """Statistics for messages this user received through ``group_id``."""
        if group_id is None:
            return _empty(InteractionScope.GROUP)
        return self._cached(
            InteractionScope.GROUP,
            group_id,
            user_id,
            lambda: self._scoped(self._repo.get_group_history(group_id), user_id),
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _cached(
        self,
        scope: InteractionScope,
        counterparty_id: str,
        user_id: str,
        build: Callable[[], Sequence[MessageHistory]],
    ) -> InteractionStats:
        """Return cached statistics, computing them on first request."""
        key = (scope, counterparty_id, user_id)
        cached = self._cache.get(key)
        if cached is None:
            cached = self._summarise(scope, build())
            self._cache[key] = cached
        return cached

    @staticmethod
    def _scoped(
        records: Sequence[MessageHistory], user_id: str
    ) -> tuple[MessageHistory, ...]:
        """Keep only the records this user actually received."""
        return tuple(record for record in records if record.user_id == user_id)

    def _summarise(
        self, scope: InteractionScope, records: Sequence[MessageHistory]
    ) -> InteractionStats:
        """Reduce history records and their events to rates and timestamps.

        Records are already oldest-first from the Phase 1 indexes, which the
        trend calculation relies on.
        """
        total = len(records)
        if total == 0:
            return _empty(scope)

        events = [self._repo.get_message_event(record.message_id) for record in records]
        present = [event for event in events if event is not None]

        return InteractionStats(
            scope=scope,
            total=total,
            open_rate=ratio(sum(e.message_opened for e in present), total),
            reply_rate=ratio(sum(e.message_replied for e in present), total),
            dismiss_rate=ratio(sum(e.notification_dismissed for e in present), total),
            report_rate=ratio(sum(e.message_reported for e in present), total),
            mute_rate=ratio(sum(e.muted_after_message for e in present), total),
            last_seen=records[-1].created_at,
            first_seen=records[0].created_at,
            engagement_trend=self._trend(events),
        )

    @staticmethod
    def _trend(events: Sequence[MessageEvent | None]) -> float:
        """Compare engagement in the recent half against the earlier half.

        Args:
            events: Interaction events in chronological order, possibly with
                gaps where a message has no recorded event.

        Returns:
            ``0.5`` when flat or when there is too little history to say.
        """
        if len(events) < MIN_RECORDS_FOR_TREND:
            return NEUTRAL

        midpoint = len(events) // 2
        earlier = _engagement_of(events[:midpoint])
        recent = _engagement_of(events[midpoint:])
        return trend_score(recent, earlier)


def _engagement_of(events: Sequence[MessageEvent | None]) -> float:
    """Mean of open and reply rate across a slice of events."""
    present = [event for event in events if event is not None]
    if not present:
        return 0.0
    opened = ratio(sum(e.message_opened for e in present), len(present))
    replied = ratio(sum(e.message_replied for e in present), len(present))
    return (opened + replied) / 2.0
