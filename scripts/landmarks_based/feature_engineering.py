import numpy as np
import pandas as pd
from landmark_utils import get_angle, get_distance, parse_landmarks, LANDMARK_COUNT

# Angle and distance index tuples for feature extraction (same as your original)
ANGLES = [
    (0, 2, 4), (1, 3, 5), (2, 4, 6), (4, 6, 8), (5, 7, 9),
    (5, 8, 11), (7, 9, 11), (8, 10, 12), (11, 13, 15),
    (17, 18, 21), (17, 18, 19), (17, 19, 21),
    (26, 25, 24), (26, 25, 22), (26, 24, 22),
    (36, 37, 39), (37, 38, 39), (37, 36, 41),
    (44, 45, 46), (45, 44, 43), (44, 43, 42),
    (49, 50, 52), (59, 58, 57), (61, 62, 63), (67, 66, 65),
    (49, 48, 59), (52, 54, 56), (60, 58, 56),
    (33, 27, 30)
]

DISTANCES = [
    (17, 21), (17, 18), (18, 21), (19, 21),
    (22, 24), (22, 25), (22, 26), (25, 26),
    (21, 22),
    (36, 39), (37, 41), (38, 40),
    (42, 45), (43, 47), (44, 46),
    (30, 33), (31, 33), (35, 33),
    (48, 50), (54, 50), (48, 58), (54, 58), (50, 58), (48, 54),
    (27, 39), (27, 42)
]

def compute_inter_distances(landmarks):
    """
    Compute Euclidean distances between every unique pair of landmarks.
    
    Args:
        landmarks (list): List of (x, y) tuples.
        
    Returns:
        list: Distances.
    """
    return [np.linalg.norm(np.array(landmarks[i]) - np.array(landmarks[j]))
            for i in range(len(landmarks)) for j in range(i + 1, len(landmarks))]

def compute_center_distances(landmarks):
    """
    Compute distances of all landmarks from the center landmark (index 33).
    
    Args:
        landmarks (list): List of (x, y) tuples.
        
    Returns:
        list: Distances.
    """
    center = np.array(landmarks[33])
    return [np.linalg.norm(np.array(pt) -
