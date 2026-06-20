import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from softmax_agg_predictions import combine_predictions


def test_combine_predictions_shape_and_normalization():
    a = np.ones((3, 8))
    b = np.zeros((3, 8))
    probs, labels = combine_predictions(a, b)
    assert probs.shape == (3, 8)
    assert len(labels) == 3
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(3))
