import numpy as np

from scripts.softmax_agg_predictions import combine_predictions, EMOTION_LABELS


def test_combine_predictions_shape():
    preds_dl = np.array([
        [0.70, 0.05, 0.05, 0.04, 0.04, 0.03, 0.05, 0.04],
        [0.05, 0.72, 0.04, 0.05, 0.03, 0.03, 0.04, 0.04],
    ])

    preds_xgb = np.array([
        [0.65, 0.08, 0.06, 0.04, 0.04, 0.03, 0.06, 0.04],
        [0.06, 0.69, 0.05, 0.05, 0.04, 0.03, 0.04, 0.04],
    ])

    probs, labels = combine_predictions(preds_dl, preds_xgb)

    assert probs.shape == (2, 8)
    assert len(labels) == 2
    assert all(label in EMOTION_LABELS for label in labels)


def test_combine_predictions_probabilities_sum_to_one():
    preds_dl = np.ones((3, 8))
    preds_xgb = np.ones((3, 8))

    probs, labels = combine_predictions(preds_dl, preds_xgb)

    row_sums = probs.sum(axis=1)

    assert np.allclose(row_sums, 1.0)