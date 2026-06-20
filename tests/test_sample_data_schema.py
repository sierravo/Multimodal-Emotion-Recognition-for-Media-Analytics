import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EMOTIONS = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]


def test_landmark_sample_schema():
    df = pd.read_csv(ROOT / "sample_data" / "landmark_features_sample.csv")
    assert "target" in df.columns
    assert "img_path" in df.columns
    assert len([c for c in df.columns if c.startswith("f")]) == 55
    assert set(df["target"].unique()) == set(range(8))


def test_prediction_sample_schema():
    for fname in ["xgb_predictions_sample.csv", "bregnext_predictions_sample.csv"]:
        df = pd.read_csv(ROOT / "sample_data" / fname)
        for emotion in EMOTIONS:
            assert f"prob_{emotion}" in df.columns
