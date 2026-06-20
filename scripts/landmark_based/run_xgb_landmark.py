import pandas as pd
import joblib
import os
import argparse

# AffectNet label mapping
affectnet_labels = {
    0: 'Neutral',
    1: 'Happy',
    2: 'Sad',
    3: 'Surprise',
    4: 'Fear',
    5: 'Disgust',
    6: 'Anger',
    7: 'Contempt'
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run XGBoost model on new data")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model (.joblib)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to new dataset CSV")
    parser.add_argument("--output_path", type=str, default="outputs/xgb_predictions.csv", help="Path to save predictions")
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = args.model_path
    data_path = args.data_path
    output_path = args.output_path

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ==== LOAD MODEL ====
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

    # ==== LOAD NEW DATA ====
    print(f"Loading new dataset from: {data_path}")
    df_new = pd.read_csv(data_path)

    # Drop non-feature columns if present
    drop_cols = [c for c in ['target', 'label', 'emotion', 'class', 'img_path'] if c in df_new.columns]
    X_new = df_new.drop(columns=drop_cols, errors='ignore')

    # ==== PREDICT ====
    print("Predicting labels and probabilities...")
    y_pred = model.predict(X_new)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_new)
    else:
        raise ValueError("Model does not support probability predictions")

    # ==== OUTPUT ====
    df_output = pd.DataFrame(
        y_proba,
        columns=[f"prob_{affectnet_labels[i]}" for i in range(len(affectnet_labels))]
    )

    df_output.insert(0, "predicted_label", y_pred)
    df_output.insert(1, "predicted_emotion", [affectnet_labels[label] for label in y_pred])

    df_output.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()