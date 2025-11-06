import math
import numpy as np

LANDMARK_COUNT = 68

def parse_landmarks(landmark_str):
    """
    Parse a semicolon-separated string of landmark coordinates into a list of (x, y) tuples.
    
    Args:
        landmark_str (str): Landmark coordinates as "x0;y0;x1;y1;...".
        
    Returns:
        List[Tuple[float, float]]: List of (x, y) coordinates.
    """
    coords = list(map(float, landmark_str.strip().split(";")))
    if len(coords) != LANDMARK_COUNT * 2:
        raise ValueError(f"Expected {LANDMARK_COUNT*2} coordinates, got {len(coords)}")
    return [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]

def normalize_landmarks(landmarks, width, height):
    """
    Normalize landmarks relative to image center and scale to a 200x200 coordinate system.
    
    Args:
        landmarks (list): List of (x, y) tuples.
        width (int): Image width.
        height (int): Image height.
        
    Returns:
        list: Normalized landmarks.
    """
    return [((x - width/2) * 200/width + 100, (y - height/2) * 200/height + 100) for x, y in landmarks]

def shift_landmarks(landmarks, dx, dy):
    """
    Shift landmarks by dx, dy.
    
    Args:
        landmarks (list): List of (x, y) tuples.
        dx (float): Shift in x.
        dy (float): Shift in y.
        
    Returns:
        list: Shifted landmarks.
    """
    return [(x + dx, y + dy) for x, y in landmarks]

def rotate_landmarks(landmarks, angle_deg, center=(100, 100)):
    """
    Rotate landmarks around a center by angle degrees.
    
    Args:
        landmarks (list): List of (x, y) tuples.
        angle_deg (float): Rotation angle in degrees.
        center (tuple): Center of rotation.
        
    Returns:
        list: Rotated landmarks.
    """
    angle_rad = math.radians(angle_deg)
    cx, cy = center
    return [(cx + math.cos(angle_rad)*(x-cx) - math.sin(angle_rad)*(y-cy),
             cy + math.sin(angle_rad)*(x-cx) + math.cos(angle_rad)*(y-cy)) for x, y in landmarks]

def scale_landmarks(landmarks, factor_x, factor_y, center=(100, 100)):
    """
    Scale landmarks around a center by factor_x and factor_y.
    
    Args:
        landmarks (list): List of (x, y) tuples.
        factor_x (float): Scaling factor in x.
        factor_y (float): Scaling factor in y.
        center (tuple): Center of scaling.
        
    Returns:
        list: Scaled landmarks.
    """
    cx, cy = center
    return [((x - cx) * factor_x + cx, (y - cy) * factor_y + cy) for x, y in landmarks]

def get_angle(a, b, c):
    """
    Calculate the angle (in degrees) at point b formed by points a, b, c.
    
    Args:
        a, b, c (tuple): Points (x, y).
        
    Returns:
        float: Angle in degrees.
    """
    ang = 180 - math.degrees(math.atan2(abs(b[1] - a[1]), abs(a[0] - b[0])) +
                             math.atan2(abs(c[1] - b[1]), abs(c[0] - b[0])))
    return ang + 360 if ang < 0 else ang

def get_distance(a, b):
    """
    Euclidean distance between two points.
    
    Args:
        a, b (tuple): Points (x, y).
        
    Returns:
        float: Distance.
    """
    return np.linalg.norm(np.array(a) - np.array(b))
