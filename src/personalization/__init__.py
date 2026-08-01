"""Phase 3 personalisation: routing signals, not routing decisions.

Turns Phase 2 outputs plus repository context into a
:class:`~src.personalization.signal_models.RoutingSignals` record - ten
independent, normalised, explained signals that Phase 4 will combine into a
``notify`` / ``digest`` / ``mute`` decision.

Public exports are declared once every module exists; see
:mod:`src.personalization.engine` for the entry point.
"""
