import os
import math
import cv2
import numpy as np
from PIL import Image
from landmark_utils import parse_landmarks, normalize_landmarks, shift_landmarks, rotate_landmarks, scale_landmarks

TEMP_PATH = "shifted_temp.png"

def change_image(image_path, landmark_row):
    """
    Process an image and its landmarks: normalize, shift, rotate, scale landmarks,
    transform image accordingly and return processed PIL image and new landmark string.
    
    Args:
        image_path (str): Path to the input image.
        landmark_row (pandas.Series): Row containing 'facial_landmarks' string.
    
    Returns:
        PIL.Image: Transformed image.
        str: New landmarks as a semicolon-separated string.
    """
    landmarks = parse_landmarks(landmark_row['facial_landmarks'])
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found or unreadable: {image_path}")

    h, w = img.shape[:2]
    norm_landmarks = normalize_landmarks(landmarks, w, h)
    dx, dy = 100 - norm_landmarks[33][0], 100 - norm_landmarks[33][1]
    shifted_landmarks = shift_landmarks(norm_landmarks, dx, dy)

    img_resized = cv2.resize(img, (200, 200))
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    img_shifted = cv2.warpAffine(img_resized, M, (200, 200))
    cv2.imwrite(TEMP_PATH, img_shifted)

    dx_r = shifted_landmarks[27][0] - shifted_landmarks[33][0]
    dy_r = shifted_landmarks[27][1] - shifted_landmarks[33][1]
    angle = math.degrees(math.atan2(dx_r, -dy_r))
    rotated_landmarks = rotate_landmarks(shifted_landmarks, angle)
    img_rotated = Image.open(TEMP_PATH).rotate(angle)

    brow_nose_dist = rotated_landmarks[33][1] - rotated_landmarks[27][1]
    scale_y = 45 / brow_nose_dist if abs(brow_nose_dist) > 1e-3 else 1.0
    scaled_landmarks_y = scale_landmarks(rotated_landmarks, 1.0, scale_y)

    eye_width = scaled_landmarks_y[35][0] - scaled_landmarks_y[31][0]
    scale_x = 30 / eye_width if abs(eye_width) > 1e-3 else 1.0
    final_landmarks = scale_landmarks(scaled_landmarks_y, scale_x, 1.0)

    new_w, new_h = int(200 * scale_x), int(200 * scale_y)
    img_final = img_rotated.resize((new_w, new_h))

    dx_final, dy_final = new_w / 2 - 100, new_h / 2 - 100
    final_landmarks_shifted = shift_landmarks(final_landmarks, dx_final, dy_final)

    landmark_str = ";".join(f"{x:.2f};{y:.2f}" for x, y in final_landmarks_shifted)
    return img_final, landmark_str
