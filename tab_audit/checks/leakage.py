from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def _normalize_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def run_leakage_checks(df: pd.DataFrame, target: str | None, task: str | None, random_seed: int) -> dict:
    if not target or target not in df.columns or task not in {"classification", "regression"}:
        return {"warnings": ["leakage checks skipped (no supervised target)"], "signals": []}

    warnings: list[str] = []
    signals: list[dict] = []
    y = df[target]

    target_norm = _normalize_series(y)
    for col in df.columns:
        if col == target:
            continue
        x_norm = _normalize_series(df[col])
        if x_norm.equals(target_norm):
            signals.append({"type": "identical_to_target", "column": col, "severity": "high"})

    if task == "regression":
        y_num = pd.to_numeric(y, errors="coerce")
        for col in df.columns:
            if col == target:
                continue
            x_num = pd.to_numeric(df[col], errors="coerce")
            if x_num.notna().sum() < 20:
                continue
            corr = x_num.corr(y_num)
            if pd.notna(corr) and abs(float(corr)) > 0.999:
                signals.append({"type": "near_perfect_corr", "column": col, "corr": float(corr), "severity": "high"})

    if task == "classification":
        for col in df.columns:
            if col == target:
                continue
            group_nuniq = df.groupby(target, dropna=False)[col].nunique(dropna=False)
            if (group_nuniq <= 1).all() and df[col].nunique(dropna=False) >= y.nunique(dropna=False):
                signals.append({"type": "class_constant_separator", "column": col, "severity": "medium"})

    near_unique_cols = [c for c in df.columns if c != target and df[c].nunique(dropna=True) / max(1, len(df)) > 0.95]
    for col in near_unique_cols[:20]:
        X_col = df[[col]].astype(str)
        X_train, X_test, y_train, y_test = train_test_split(X_col, y, test_size=0.3, random_state=random_seed)

        if task == "classification":
            le = LabelEncoder()
            Xt = le.fit_transform(X_train.iloc[:, 0])[:, None]
            known_values = set(X_train.iloc[:, 0])
            Xv = le.transform(X_test.iloc[:, 0])[:, None] if set(X_test.iloc[:, 0]).issubset(known_values) else None
            if Xv is None:
                continue
            clf = KNeighborsClassifier(n_neighbors=1)
            clf.fit(Xt, y_train)
            score = accuracy_score(y_test, clf.predict(Xv))
            dt = DecisionTreeClassifier(max_depth=2, random_state=random_seed)
            dt.fit(Xt, y_train)
            dt_score = accuracy_score(y_test, dt.predict(Xv))
            best = max(score, dt_score)
        else:
            x_train = pd.to_numeric(X_train.iloc[:, 0], errors="coerce").fillna(0).to_numpy()[:, None]
            x_test = pd.to_numeric(X_test.iloc[:, 0], errors="coerce").fillna(0).to_numpy()[:, None]
            knn = KNeighborsRegressor(n_neighbors=1)
            knn.fit(x_train, y_train)
            score = r2_score(y_test, knn.predict(x_test))
            dt = DecisionTreeRegressor(max_depth=2, random_state=random_seed)
            dt.fit(x_train, y_train)
            dt_score = r2_score(y_test, dt.predict(x_test))
            best = max(score, dt_score)

        if best > 0.95:
            signals.append({"type": "id_leakage_suspected", "column": col, "score": float(best), "severity": "high"})

    if signals:
        warnings.append("potential leakage signals detected")
    return {"warnings": warnings, "signals": signals}
