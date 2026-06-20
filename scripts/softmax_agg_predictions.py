import argparse

import numpy as np
import pandas as pd

EMOTION_LABELS = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
PROB_COLS = [f"prob_{label}" for label in EMOTION_LABELS]


def _select_prob_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return probability columns in canonical emotion order."""
    lower_map = {c.lower(): c for c in df.columns}
    selected = []
    for col in PROB_COLS:
        key = col.lower()
        if key not in lower_map:
            raise ValueError(f"Missing probability column '{col}'. Available columns: {list(df.columns)}")
        selected.append(lower_map[key])
    return df[selected]


def combine_predictions(preds_dl, preds_xgb):
    """Combine two prediction matrices by adding scores and applying row-wise softmax."""
    preds_dl = np.asarray(preds_dl, dtype=float)
    preds_xgb = np.asarray(preds_xgb, dtype=float)

    if preds_dl.shape != preds_xgb.shape:
        raise ValueError(f"Shape mismatch: dl={preds_dl.shape}, xgb={preds_xgb.shape}")
    if preds_dl.ndim != 2 or preds_dl.shape[1] != len(EMOTION_LABELS):
        raise ValueError(f"Expected matrices with {len(EMOTION_LABELS)} emotion columns")

    combined_logits = preds_dl + preds_xgb
    combined_logits = combined_logits - np.max(combined_logits, axis=1, keepdims=True)
    exp_logits = np.exp(combined_logits)
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    predicted_indices = np.argmax(softmax_probs, axis=1)
    final_labels = [EMOTION_LABELS[idx] for idx in predicted_indices]
    return softmax_probs, final_labels


def parse_args():
    parser = argparse.ArgumentParser(description="Combine XGBoost and deep-learning emotion predictions.")
    parser.add_argument("--xgb_input", required=True, help="Path to XGBoost prediction CSV")
    parser.add_argument("--dl_input", required=True, help="Path to deep-learning prediction CSV")
    parser.add_argument("--output_csv", required=True, help="Path to save combined predictions")
    return parser.parse_args()


def main():
    args = parse_args()
    preds_xgb = pd.read_csv(args.xgb_input)
    preds_dl = pd.read_csv(args.dl_input)

    preds_xgb_values = _select_prob_columns(preds_xgb).to_numpy()
    preds_dl_values = _select_prob_columns(preds_dl).to_numpy()

    softmax_probs, final_labels = combine_predictions(preds_dl_values, preds_xgb_values)
    output_df = pd.DataFrame(softmax_probs, columns=PROB_COLS)
    output_df.insert(0, "predicted_label", final_labels)

    output_df.to_csv(args.output_csv, index=False)
    print(f"Saved combined predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
