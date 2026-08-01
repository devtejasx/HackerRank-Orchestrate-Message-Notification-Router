"""Shared fixtures.

The real dataset is loaded once per session; tests that need to break
something work on a throwaway copy under ``tmp_path`` instead.
"""

from __future__ import annotations

import csv
import shutil
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import pytest

from src import config
from src.data.loader import DataLoader
from src.data.repository import DataRepository

#: Row of a CSV as read by :class:`csv.DictReader`.
Row = dict[str, str]


@pytest.fixture(scope="session")
def dataset_dir() -> Path:
    """Directory holding the real, shipped dataset."""
    if not config.DATASET_DIR.is_dir():
        pytest.skip(f"dataset directory not found: {config.DATASET_DIR}")
    return config.DATASET_DIR


@pytest.fixture(scope="session")
def loader(dataset_dir: Path) -> DataLoader:
    """A loader over the real dataset, warmed once."""
    instance = DataLoader(dataset_dir)
    instance.load_all()
    return instance


@pytest.fixture(scope="session")
def repo(dataset_dir: Path) -> DataRepository:
    """A fully loaded, validated and indexed repository over the real dataset."""
    return DataRepository.load(dataset_dir)


@pytest.fixture(scope="session")
def busy_user(repo: DataRepository) -> str:
    """Id of the user with the most historical messages."""
    return max(
        repo.index.history_by_user,
        key=lambda user_id: len(repo.index.history_by_user[user_id]),
    )


@pytest.fixture
def dataset_copy(dataset_dir: Path, tmp_path: Path) -> Path:
    """Return a writable copy of the dataset that tests may corrupt."""
    destination = tmp_path / "dataset"
    shutil.copytree(dataset_dir, destination)
    return destination


@pytest.fixture
def read_csv() -> Callable[[Path], tuple[list[Row], list[str]]]:
    """Return a function reading a CSV into ``(rows, fieldnames)``."""

    def _read(path: Path) -> tuple[list[Row], list[str]]:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            return list(reader), fieldnames

    return _read


@pytest.fixture
def write_csv() -> Callable[[Path, Sequence[str], Iterable[Row]], None]:
    """Return a function writing rows back to a CSV."""

    def _write(path: Path, fieldnames: Sequence[str], rows: Iterable[Row]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
            writer.writeheader()
            writer.writerows(rows)

    return _write


@pytest.fixture
def mutate_csv(
    read_csv: Callable[[Path], tuple[list[Row], list[str]]],
    write_csv: Callable[[Path, Sequence[str], Iterable[Row]], None],
) -> Callable[[Path, Callable[[list[Row], list[str]], tuple[list[Row], list[str]]]], None]:
    """Return a function that rewrites a CSV through a transform.

    The transform receives ``(rows, fieldnames)`` and returns the new pair,
    which keeps corruption tests to a couple of readable lines.
    """

    def _mutate(
        path: Path,
        transform: Callable[[list[Row], list[str]], tuple[list[Row], list[str]]],
    ) -> None:
        rows, fieldnames = read_csv(path)
        new_rows, new_fieldnames = transform(rows, fieldnames)
        write_csv(path, new_fieldnames, new_rows)

    return _mutate
