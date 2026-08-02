"""Measurement against the labelled examples in ``sample_messages.csv``.

    from src.evaluation import evaluate_samples

    report = evaluate_samples(pipeline)
    report.action_accuracy      # 0.967
"""

from src.evaluation.evaluator import (
    EvaluationReport,
    SampleOutcome,
    as_message,
    evaluate_samples,
    format_misses,
)

__all__ = [
    "EvaluationReport",
    "SampleOutcome",
    "as_message",
    "evaluate_samples",
    "format_misses",
]
