import os
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DIR = os.path.join(PROJECT_ROOT, "sample_data")


def test_landmark_sample_schema():
    path = os.path.join(SAMPLE_DIR, "landmark_features_sample.csv")
    df = pd.read_csv(path)

    assert "target" in df.columns
    assert "img_path" in df.columns
    assert len(df) > 0

    feature_cols = [c for c in df.columns if c not in ["target", "img_path"]]
    assert len(feature_cols) > 0

    # For 8-class stratified split, sample should have at least 2 examples per class
    class_counts = df["target"].value_counts()
    assert class_counts.min() >= 2


def test_xgb_prediction_sample_schema():
    path = os.path.join(SAMPLE_DIR, "xgb_predictions_sample.csv")
    df = pd.read_csv(path)

    required_cols = [
        "true_label",
        "true_emotion",
        "predicted_label",
        "predicted_emotion",
    ]

    for col in required_cols:
        assert col in df.columns

    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    assert len(prob_cols) == 8

    row_sums = df[prob_cols].sum(axis=1)
    assert all(abs(row_sums - 1.0) < 1e-6)


def test_bregnext_prediction_sample_schema():
    path = os.path.join(SAMPLE_DIR, "bregnext_predictions_sample.csv")
    df = pd.read_csv(path)

    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    assert len(prob_cols) == 8

    row_sums = df[prob_cols].sum(axis=1)
    assert all(abs(row_sums - 1.0) < 1e-6)