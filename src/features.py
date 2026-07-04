"""
Feature engineering: DllCharacteristics bitmask decoding and duplicate-vector
statistics.
"""
from __future__ import annotations

import pandas as pd

# Bit positions per the PE/COFF specification (IMAGE_DLLCHARACTERISTICS_*).
DLL_CHARACTERISTICS_FLAGS = {
    "dyn_base_aslr": 0x0040,          # image can be relocated (ASLR)
    "force_integrity": 0x0080,
    "nx_compat_dep": 0x0100,          # Data Execution Prevention compatible
    "no_isolation": 0x0200,
    "no_seh": 0x0400,                 # no Structured Exception Handling
    "no_bind": 0x0800,
    "wdm_driver": 0x2000,
    "terminal_server_aware": 0x8000,
}


def decode_dll_characteristics(df: pd.DataFrame, column: str = "DllCharacteristics") -> pd.DataFrame:
    """Return a copy of ``df`` with one new boolean column per DLL characteristic flag."""
    out = df.copy()
    for name, bitmask in DLL_CHARACTERISTICS_FLAGS.items():
        out[name] = (out[column] & bitmask).astype(bool).astype(int)
    return out


def get_duplicate_stats(df: pd.DataFrame, feature_cols: list[str], label_col: str = "is_malware") -> dict:
    """Compute duplicate-row statistics.

    NOTE ON TERMINOLOGY: this measures duplicate *feature vectors* (rows whose
    header values, and optionally label, are identical), not confirmed
    duplicate *physical files*. The dataset has no file hash or unique
    identifier, so two different binaries that happen to share an identical
    8-value header signature (e.g. built by the same toolchain) are
    indistinguishable, in this data, from two copies of the same file. Read
    all "duplicate" figures below with that caveat.
    """
    n_full_dupes = df.duplicated().sum()
    n_feature_dupes = df.duplicated(subset=feature_cols).sum()
    return {
        "n_rows": len(df),
        "n_full_duplicate_rows": int(n_full_dupes),
        "pct_full_duplicate_rows": n_full_dupes / len(df),
        "n_duplicate_feature_vectors": int(n_feature_dupes),
        "pct_duplicate_feature_vectors": n_feature_dupes / len(df),
        "note": (
            "Counts reflect duplicate FEATURE VECTORS, not confirmed duplicate "
            "physical files (no file hash/ID exists in this dataset)."
        ),
    }


def deduplicate(df: pd.DataFrame, feature_cols: list[str], label_col: str = "is_malware") -> pd.DataFrame:
    """Drop rows whose feature+label vector is an exact duplicate of an earlier row.

    This is the central methodological fix applied in Pipeline B: the source
    article calls train_test_split on the raw, un-deduplicated data, which
    lets duplicate feature vectors leak between train and test.
    """
    return df.drop_duplicates(subset=feature_cols + [label_col]).reset_index(drop=True)
