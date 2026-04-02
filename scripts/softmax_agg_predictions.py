import numpy as np
import pandas as pd
import argparse


# Emotion labels
EMOTION_LABELS = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt']

def combine_predictions(preds_dl, preds_xgb):
    """
    Combines predictions from two models using softmax on the sum of logits.

    Parameters:
    - preds_dl (np.ndarray): Predicted log-probabilities from BReG-NeXt (n_samples, 8)
    - preds_xgb (np.ndarray): Predicted log-probabilities from XGBoost (n_samples, 8)

    Returns:
    - final_probs (np.ndarray): Combined softmax probabilities (n_samples, 8)
    - final_labels (List[str]): Predicted emotion category for each image
    """
    preds_dl = np.asarray(preds_dl)
    preds_xgb = np.asarray(preds_xgb)

    assert preds_dl.shape == preds_xgb.shape, "Shape mismatch between model predictions"
    assert preds_dl.shape[1] == 8, "Expecting 8 emotion categories"

    combined_logits = preds_dl + preds_xgb

    # Compute softmax
    combined_logits = combined_logits - np.max(combined_logits, axis=1, keepdims=True)
    # avoid overflow
    exp_logits = np.exp(combined_logits)
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Final prediction labels (forced-choice)
    predicted_indices = np.argmax(softmax_probs, axis=1)
    final_labels = [emotion_labels[idx] for idx in predicted_indices]

    return softmax_probs, final_labels

def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine XGBoost and deep learning emotion predictions."
    )
    parser.add_argument("--xgb_input", required=True, help="Path to XGBoost prediction CSV")
    parser.add_argument("--dl_input", required=True, help="Path to deep learning prediction CSV")
    parser.add_argument("--output_csv", required=True, help="Path to save combined predictions")
    return parser.parse_args()

def main():
    args = parse_args()

    preds_xgb = pd.read_csv(args.xgb_input)
    preds_dl = pd.read_csv(args.dl_input)

    # Adjust this slice if your XGB output format changes
    preds_xgb_values = preds_xgb.iloc[:, 2:].to_numpy()
    preds_dl_values = preds_dl.to_numpy()

    softmax_probs, final_labels = combine_predictions(preds_dl_values, preds_xgb_values)

    output_df = pd.DataFrame(
        softmax_probs,
        columns=[f"prob_{label}" for label in EMOTION_LABELS]
    )
    output_df.insert(0, "predicted_label", final_labels)

    output_df.to_csv(args.output_csv, index=False)
    print(f"Saved combined predictions to {args.output_csv}")


if __name__ == "__main__":
    main()


