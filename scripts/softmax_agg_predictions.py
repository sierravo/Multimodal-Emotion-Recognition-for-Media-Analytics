import numpy as np


XGB_OUTPUT_PATH = "..." # path to XGB prediction results
DL_OUTPUT_PATH = "..." # path to BReg-NeXt prediction results

preds_xgb = pd.read_csv(XGB_OUTPUT_PATH)
preds_dl = pd.read_csv(DL_OUTPUT_PATH)

preds_xgb = preds_xgb.iloc[:, 2:]

# Emotion labels
emotion_labels = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt']

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
    assert preds_dl.shape == preds_xgb.shape, "Shape mismatch between model predictions"
    assert preds_dl.shape[1] == 8, "Expecting 8 emotion categories"

    combined_logits = preds_dl + preds_xgb

    # Compute softmax
    exp_logits = np.exp(combined_logits)
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Final prediction labels (forced-choice)
    predicted_indices = np.argmax(softmax_probs, axis=1)
    final_labels = [emotion_labels[idx] for idx in predicted_indices]

    return softmax_probs, final_labels
