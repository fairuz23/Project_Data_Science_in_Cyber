import pandas as pd
import pytest

from src.features import DLL_CHARACTERISTICS_FLAGS, decode_dll_characteristics, deduplicate, get_duplicate_stats


def test_decode_dll_characteristics_sets_expected_flags():
    # 0x0140 = nx_compat_dep (0x0100) | dyn_base_aslr (0x0040)
    df = pd.DataFrame({"DllCharacteristics": [0x0140, 0x0000]})
    out = decode_dll_characteristics(df)
    assert out.loc[0, "nx_compat_dep"] == 1
    assert out.loc[0, "dyn_base_aslr"] == 1
    assert out.loc[0, "no_seh"] == 0
    assert out.loc[1, "nx_compat_dep"] == 0
    # every declared flag column should exist
    for flag_name in DLL_CHARACTERISTICS_FLAGS:
        assert flag_name in out.columns


@pytest.fixture
def dup_df() -> pd.DataFrame:
    # rows 0 and 1 share an identical feature+label vector; row 2 differs.
    return pd.DataFrame(
        {
            "f1": [1, 1, 2],
            "f2": [5, 5, 6],
            "is_malware": [0, 0, 1],
        }
    )


def test_get_duplicate_stats_counts_feature_vector_duplicates(dup_df):
    stats = get_duplicate_stats(dup_df, feature_cols=["f1", "f2"])
    assert stats["n_rows"] == 3
    assert stats["n_duplicate_feature_vectors"] == 1
    assert stats["n_full_duplicate_rows"] == 1
    # the stats dict must always carry the feature-vector-vs-physical-file caveat
    assert "feature vector" in stats["note"].lower() or "FEATURE VECTORS" in stats["note"]


def test_deduplicate_drops_exact_feature_label_duplicates(dup_df):
    deduped = deduplicate(dup_df, feature_cols=["f1", "f2"])
    assert len(deduped) == 2
    assert deduped["f1"].tolist() == [1, 2]
