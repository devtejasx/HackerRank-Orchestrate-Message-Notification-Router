"""Data layer: schema registry, models, loading, validation and indexing.

Nothing outside this package should read a CSV directly. Start here:

    from src.data import DataRepository

    repo = DataRepository.load()
    user = repo.get_user("u_001")
    history = repo.get_user_history("u_001", limit=10, newest_first=True)

:class:`~src.data.repository.DataRepository` owns the whole pipeline; the
loader, index and validation report are reachable through it when a caller
genuinely needs the lower level.
"""

from src.data.indexes import DataIndex, build_indexes
from src.data.loader import DataLoader, DatasetError, MissingDatasetFileError
from src.data.repository import DataRepository
from src.data.schema import TABLES, TableSpec, get_spec
from src.data.validation import (
    DatasetValidationError,
    Severity,
    ValidationIssue,
    ValidationReport,
    validate_dataset,
)

__all__ = [
    "TABLES",
    "DataIndex",
    "DataLoader",
    "DataRepository",
    "DatasetError",
    "DatasetValidationError",
    "MissingDatasetFileError",
    "Severity",
    "TableSpec",
    "ValidationIssue",
    "ValidationReport",
    "build_indexes",
    "get_spec",
    "validate_dataset",
]
