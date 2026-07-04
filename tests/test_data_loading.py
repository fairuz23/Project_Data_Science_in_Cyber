import pandas as pd
import pytest

from src.data_loading import feature_columns


@pytest.fixture
def toy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "AddressOfEntryPoint": [1, 2, 3],
            "MajorLinkerVersion": [9, 9, 10],
            "is_malware": [0, 1, 0],
        }
    )


def test_feature_columns_excludes_label(toy_df):
    cols = feature_columns(toy_df)
    assert "is_malware" not in cols
    assert set(cols) == {"AddressOfEntryPoint", "MajorLinkerVersion"}
