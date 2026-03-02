from __future__ import annotations

import numpy as np
import pandas as pd

from tab_audit.checks.cleanliness import run_cleanliness_checks


def test_cleanliness_invalid_values_counted():
    df = pd.DataFrame(
        {
            "a": [1.0, np.inf, -np.inf, 4.0],
            "b": [" ", "ok", "", None],
            "c": ["1", "x", "2", "y"],
        }
    )

    result = run_cleanliness_checks(df)
    assert result["invalid_numeric_count"] == 2
    assert result["whitespace_only_count"] >= 2
    assert result["mixed_type_columns_count"] >= 1
