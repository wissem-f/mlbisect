"""Offline synthetic classification data and case evaluations."""

from collections.abc import Mapping

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

CASE_IDS = (
    "L01_data_skew",
    "L02_preproc_bump",
    "L03_interaction",
    "L04_small_effect",
    "L05_decoy_axes",
    "L06_no_regression",
)


def evaluate_case(
    case_id: str, axis_values: Mapping[str, str], seed: int
) -> float:
    """Train and evaluate the scikit-learn model for one lab case."""
    if case_id not in CASE_IDS:
        raise ValueError(f"unknown lab case: {case_id}")

    rng = np.random.default_rng(seed)
    train_x, train_y = _make_split(case_id, axis_values, rng, 320)
    test_x, test_y = _make_split(case_id, axis_values, rng, 320)

    if case_id in {"L02_preproc_bump", "L03_interaction"} and axis_values.get(
        "deps", "v1"
    ) == "v1":
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)

    model = KNeighborsClassifier(n_neighbors=7)
    model.fit(train_x, train_y)
    metric = float(model.score(test_x, test_y))

    if case_id == "L04_small_effect":
        metric += float(rng.normal(0.0, 0.015))
    return float(np.clip(metric, 0.0, 1.0))


def _make_split(
    case_id: str,
    axis_values: Mapping[str, str],
    rng: np.random.Generator,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels = rng.integers(0, 2, size=size)
    signed_labels = labels * 2 - 1

    signal_scale = 1.4
    signal_noise = 0.7
    if case_id == "L01_data_skew" and axis_values.get("data") == "v2":
        signal_scale = 0.45
        signal_noise = 1.05
    elif case_id == "L04_small_effect" and axis_values.get("data") == "v2":
        signal_scale = 1.34
        signal_noise = 0.73

    feature_1 = signed_labels * signal_scale + rng.normal(0, signal_noise, size)
    feature_2 = rng.normal(0, 0.1, size)
    feature_3 = rng.normal(0, 0.1, size)

    if case_id in {"L02_preproc_bump", "L03_interaction"}:
        if case_id == "L02_preproc_bump":
            feature_2 = rng.normal(0, 250.0, size)
        elif axis_values.get("data") == "v2":
            # A new, noisy feature is harmless when all features are scaled.
            feature_2 = rng.normal(0, 250.0, size)

    if case_id == "L05_decoy_axes" and axis_values.get("axis_3") == "v2":
        feature_3 = rng.normal(0, 50.0, size)

    return np.column_stack((feature_1, feature_2, feature_3)), labels
