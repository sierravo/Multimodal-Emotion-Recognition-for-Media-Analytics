import os
import pandas as pd
import joblib

AFFECTNET_LABELS = {
    0: "Neutral",
    1: "Happy",
    2: "Sad",
    3: "Surprise",
    4: "Fear",
    5: "Disgust",
    6: "Anger",
    7: "Contempt",
}


def _sanitize_name(name: str) -> str:
    return name.replace(" ", "_")


def get_repo_root() -> str:
    """
    Return the project root assuming this file lives in:
    <repo>/scripts/landmark_preprocessing/utils.py
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def get_output_dir(custom_dir: str = None) -> str:
    """
    Default output directory: <repo>/outputs
    """
    if custom_dir:
        return ensure_dir(custom_dir)
    return ensure_dir(os.path.join(get_repo_root(), "outputs"))


def get_model_dir(custom_dir: str = None) -> str:
    """
    Default model directory: <repo>/models
    """
    if custom_dir:
        return ensure_dir(custom_dir)
    return ensure_dir(os.path.join(get_repo_root(), "models"))


def save_predictions(y_test, y_pred, y_proba, dataset_name, model_name, output_dir=None):
    """
    Save prediction probabilities along with true and predicted labels as CSV.
    """
    output_dir = get_output_dir(output_dir)

    df_preds = pd.DataFrame(
        y_proba,
        columns=[f"prob_{AFFECTNET_LABELS[i]}" for i in range(len(AFFECTNET_LABELS))]
    )

    y_test_series = pd.Series(y_test).reset_index(drop=True)
    y_pred_series = pd.Series(y_pred)

    df_preds.insert(0, "true_label", y_test_series)
    df_preds.insert(1, "true_emotion", y_test_series.map(AFFECTNET_LABELS))
    df_preds.insert(2, "predicted_label", y_pred_series)
    df_preds.insert(3, "predicted_emotion", y_pred_series.map(AFFECTNET_LABELS))

    filename = (
        f"{_sanitize_name(dataset_name)}_"
        f"{_sanitize_name(model_name)}_"
        f"affectnet_predictions.csv"
    )
    filepath = os.path.join(output_dir, filename)

    df_preds.to_csv(filepath, index=False)
    print(f"Saved predictions to {filepath}")


def save_model(model, model_name, dataset_name, output_dir=None):
    """
    Save the trained model to disk.
    """
    output_dir = get_model_dir(output_dir)

    filename = (
        f"best_model_{_sanitize_name(model_name)}_"
        f"{_sanitize_name(dataset_name)}.joblib"
    )
    filepath = os.path.join(output_dir, filename)

    joblib.dump(model, filepath)
    print(f"Best model saved to: {filepath}")