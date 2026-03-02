from __future__ import annotations

import pandas as pd


def maybe_sample_rows(df: pd.DataFrame, max_rows: int | None, random_seed: int) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=random_seed)
