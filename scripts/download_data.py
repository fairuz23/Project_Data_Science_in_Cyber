#!/usr/bin/env python3
"""Download MalwareDataSet.csv into data-set/ (idempotent — skips if already cached).

Usage:
    python scripts/download_data.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.data_loading import DEFAULT_CSV_PATH, download_dataset  # noqa: E402


def main() -> None:
    path = download_dataset(DEFAULT_CSV_PATH)
    print(f"Dataset available at: {path.resolve()}")


if __name__ == "__main__":
    main()
