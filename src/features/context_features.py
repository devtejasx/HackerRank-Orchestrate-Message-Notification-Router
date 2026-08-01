"""Conversation and sender context, resolved through the repository layer.

Every lookup goes through :class:`~src.data.repository.DataRepository`, which
is dict-backed, so extraction stays O(1) per field. No CSV is read here.

Absent context is represented as ``None`` rather than a zero default wherever
the distinction matters: an unknown business age is not the same as a
brand-new account, and an unknown domain is not the same as a mismatched one.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.data.models import Message
from src.data.repository import DataRepository
from src.features.feature_models import ContextFeatures
from src.utils.helpers import is_within_time_window, ratio

__all__ = ["extract_context_features"]


def extract_context_features(
    message: Message, repo: DataRepository
) -> ContextFeatures:
    """Build :class:`ContextFeatures` for one incoming message.

    Args:
        message: The message being analysed.
        repo: Loaded repository used for every lookup.

    Returns:
        A fully populated, immutable feature block.
    """
    recipient = repo.get_user(message.user_id)
    group = repo.get_group(message.group_id) if message.group_id else None
    business = repo.get_business(message.business_id) if message.business_id else None

    sender_membership = (
        repo.get_group_member(message.group_id, message.sender_user_id)
        if message.group_id and message.sender_user_id
        else None
    )
    recipient_membership = (
        repo.get_group_member(message.group_id, message.user_id)
        if message.group_id
        else None
    )
    daily_summary = repo.get_notification_summary_for_date(
        message.user_id, message.created_at.date()
    )
    load = _notification_load(message.user_id, repo)

    return ContextFeatures(
        conversation_type=message.conversation_type,
        is_personal=message.is_personal,
        is_group=message.is_group,
        is_business=message.is_business,
        media_type=message.media_type,
        has_media=message.has_media,
        forwarded_count=message.forwarded_count,
        sender_exists=(
            message.sender_user_id is not None
            and repo.get_user(message.sender_user_id) is not None
        ),
        group_exists=group is not None,
        business_exists=business is not None,
        business_verified=business.verified if business else False,
        business_age_days=business.account_age_days if business else None,
        business_domain_matches=(
            business.sender_domain_matches_official if business else None
        ),
        business_reports_30d=business.user_reports_30d if business else None,
        group_size=group.member_count if group else None,
        group_message_volume_30d=group.messages_30d if group else None,
        sender_is_admin=sender_membership.is_admin if sender_membership else False,
        recipient_is_admin=recipient_membership.is_admin if recipient_membership else False,
        group_muted=(
            recipient_membership.group_muted_by_user if recipient_membership else False
        ),
        in_quiet_hours=_is_in_quiet_hours(message, repo),
        notification_load=daily_summary.notifications_sent if daily_summary else None,
        notifications_dismissed_today=(
            daily_summary.notifications_dismissed if daily_summary else None
        ),
        avg_daily_notifications=load.average_sent,
        avg_daily_dismissed=load.average_dismissed,
        notification_dismiss_rate=load.dismiss_rate,
        recent_activity=recipient.messages_opened_30d if recipient else 0,
    )


@dataclass(frozen=True, slots=True)
class _NotificationLoad:
    """A recipient's notification volume across every recorded day."""

    average_sent: float
    average_dismissed: float
    dismiss_rate: float


def _notification_load(user_id: str, repo: DataRepository) -> _NotificationLoad:
    """Summarise a recipient's notification volume over the recorded window.

    The shipped ``daily_notification_summary`` stops the day before the
    earliest incoming message, so a same-day lookup always misses. Averaging
    the recorded window gives a usable fatigue signal instead of ``None``.

    Args:
        user_id: The recipient.
        repo: Loaded repository.

    Returns:
        Zeroed values when the recipient has no recorded days.
    """
    rows = repo.get_notification_summary(user_id)
    if not rows:
        return _NotificationLoad(0.0, 0.0, 0.0)

    sent = sum(row.notifications_sent for row in rows)
    dismissed = sum(row.notifications_dismissed for row in rows)
    return _NotificationLoad(
        average_sent=sent / len(rows),
        average_dismissed=dismissed / len(rows),
        dismiss_rate=ratio(dismissed, sent),
    )


def _is_in_quiet_hours(message: Message, repo: DataRepository) -> bool:
    """Return whether the message landed inside the recipient's quiet hours.

    ``False`` when the user is unknown or their window is unparseable, so a
    missing setting never reads as "do not disturb".
    """
    recipient = repo.get_user(message.user_id)
    if recipient is None:
        return False
    window = recipient.quiet_hours
    if window is None:
        return False
    return is_within_time_window(message.created_at.time(), window)
