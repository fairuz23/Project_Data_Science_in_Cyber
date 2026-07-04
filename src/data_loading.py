"""
Data loading and basic sanity checks for the MalwareDataSet.csv dataset.

IMPORTANT LIMITATION (see README / report Section 4): the dataset carries
**no per-file identifier** — no SHA-256 hash, filename, or unique ID — only
a plain auto-generated RangeIndex and the 8 header feature columns. This
means any "duplicate" analysis in this project (see ``get_duplicate_stats``
in ``features.py``) can only ever be interpreted as duplicate *feature
vectors*, not confirmed duplicate *physical files*. Two distinct binaries
built by the same compiler/toolchain could legitimately share an identical
8-value header signature.
"""
from __future__ import annotations

import pathlib
import urllib.request

import pandas as pd

CSV_URL = (
    "https://raw.githubusercontent.com/emr4h/"
    "Malware-Detection-Using-Machine-Learning/main/data-set/MalwareDataSet.csv"
)
DEFAULT_CSV_PATH = pathlib.Path("data-set/MalwareDataSet.csv")


def download_dataset(dest: pathlib.Path = DEFAULT_CSV_PATH, url: str = CSV_URL) -> pathlib.Path:
    """Download MalwareDataSet.csv from the source repository if not already cached."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urllib.request.urlretrieve(url, dest)
    return dest


def load_dataset(csv_path: pathlib.Path = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """Load the raw dataset and normalize the misleading label column name.

    The source's own data-generation code assigns ``legitimate = 1`` for
    malware and ``legitimate = 0`` for safe files — i.e. the column name is
    inverted relative to its values. This function renames it to
    ``is_malware`` so the semantics are unambiguous for the rest of the
    pipeline.
    """
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"legitimate": "is_malware"})
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the original 8 PE-header feature column names (excludes the label)."""
    return [c for c in df.columns if c != "is_malware"]
