"""How much this group matters to this recipient.

A group's importance is mostly about the recipient's own behaviour in it -
whether they read it, reply in it, or have muted it - rather than about the
group's size or volume. Two structural facts still count: a message from an
admin usually carries announcements, and a very large, very busy group is
noisier per message than a small quiet one.

Muting is weighted the heaviest single factor because it is an explicit
instruction from the user, not an inference about them.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, time
from typing import ClassVar, Final

from src.data.models import Group, GroupMember
from src.personalization.base import SignalCalculator, SignalContext
from src.personalization.normalization import (
    NEUTRAL,
    Contribution,
    days_between,
    evidence_confidence,
    saturating,
)
from src.personalization.signal_models import SignalPolarity
from src.utils.helpers import ratio

__all__ = ["GroupPriorityCalculator"]

#: Days of membership at which a user counts as half-settled in a group.
TENURE_HALF_POINT: Final[float] = 180.0

#: Messages sent in 30 days at which participation is half-established.
PARTICIPATION_HALF_POINT: Final[float] = 5.0

#: Replies in 30 days at which involvement is half-established. Lower than the
#: participation half-point because replying is rarer and says more.
REPLY_HALF_POINT: Final[float] = 3.0

#: Group size at which a group counts as half "large", where any single
#: message is less likely to concern the recipient personally.
SIZE_HALF_POINT: Final[float] = 100.0

#: Monthly message volume at which a group counts as half "busy".
VOLUME_HALF_POINT: Final[float] = 300.0

#: Membership rows plus prior messages needed before group statistics are
#: half-trusted.
CONFIDENCE_HALF_POINT: Final[float] = 4.0


class GroupPriorityCalculator(SignalCalculator):
    """Scores the standing of a group conversation with this recipient."""

    name: ClassVar[str] = "group_priority"
    polarity: ClassVar[SignalPolarity] = SignalPolarity.BOOST

    def contributions(self, context: SignalContext) -> Sequence[Contribution]:
        """Return the evidence about this group.

        Empty when the message did not arrive through a group.
        """
        group = context.repo.get_group(context.group_id) if context.group_id else None
        if group is None:
            return ()

        membership = context.repo.get_group_member(group.group_id, context.user_id)
        return (
            *self._membership_contributions(membership, context),
            *self._structure_contributions(group, context),
        )

    def _membership_contributions(
        self, membership: GroupMember | None, context: SignalContext
    ) -> tuple[Contribution, ...]:
        """Evidence from the recipient's own behaviour in this group."""
        if membership is None:
            return (
                Contribution(
                    name="not_muted",
                    value=NEUTRAL,
                    weight=0.22,
                ),
                Contribution(
                    name="reading",
                    value=0.0,
                    weight=0.20,
                    low_reason="User is not a recorded member of this group.",
                ),
            )

        # joined_at is a date; recency arithmetic needs a datetime, so it is
        # taken at midnight on the joining day.
        joined = datetime.combine(membership.joined_at, time.min)
        tenure = days_between(joined, context.now) or 0.0
        reading = ratio(
            membership.messages_read_30d,
            membership.messages_read_30d + membership.notifications_dismissed_30d,
        )

        return (
            Contribution(
                name="not_muted",
                value=0.0 if membership.group_muted_by_user else 1.0,
                weight=0.22,
                high_reason="Group is not muted by the user.",
                low_reason="User has muted this group.",
            ),
            Contribution(
                name="reading",
                value=reading,
                weight=0.20,
                high_reason=f"User reads most messages in this group ({reading:.0%}).",
                low_reason=(
                    f"User dismisses most notifications from this group "
                    f"({membership.notifications_dismissed_30d} in 30 days)."
                ),
            ),
            Contribution(
                name="participation",
                value=saturating(membership.messages_sent_30d, PARTICIPATION_HALF_POINT),
                weight=0.16,
                high_reason=(
                    f"User participates actively in this group "
                    f"({membership.messages_sent_30d} messages in 30 days)."
                ),
                low_reason="User does not post in this group.",
            ),
            Contribution(
                name="replying",
                value=saturating(membership.replies_sent_30d, REPLY_HALF_POINT),
                weight=0.14,
                high_reason="User replies regularly in this group.",
                low_reason="User rarely replies in this group.",
            ),
            Contribution(
                name="tenure",
                value=saturating(tenure, TENURE_HALF_POINT),
                weight=0.10,
                low_reason="User joined this group recently.",
            ),
        )

    def _structure_contributions(
        self, group: Group, context: SignalContext
    ) -> tuple[Contribution, ...]:
        """Evidence from the group's shape and who is speaking."""
        noise = (
            saturating(group.member_count, SIZE_HALF_POINT)
            + saturating(group.messages_30d, VOLUME_HALF_POINT)
        ) / 2.0

        return (
            Contribution(
                name="admin_sender",
                value=1.0 if context.features.context.sender_is_admin else NEUTRAL,
                weight=0.10,
                high_reason="Message comes from a group admin.",
            ),
            Contribution(
                name="signal_to_noise",
                value=1.0 - noise,
                weight=0.08,
                low_reason=(
                    f"Large, busy group ({group.member_count} members, "
                    f"{group.messages_30d} messages in 30 days)."
                ),
            ),
        )

    def confidence(self, context: SignalContext) -> float:
        """Confidence grows with prior group messages, plus known membership."""
        if context.group_id is None:
            return 0.0
        is_member = (
            context.repo.get_group_member(context.group_id, context.user_id) is not None
        )
        sample = context.group_stats.total + (1 if is_member else 0)
        return evidence_confidence(sample, CONFIDENCE_HALF_POINT)
