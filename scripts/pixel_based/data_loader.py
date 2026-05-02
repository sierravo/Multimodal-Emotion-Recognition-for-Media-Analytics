import os
import numpy as np
import cv2
import random
from mtcnn import MTCNN
from skimage import io, transform

class DataLoader:
    """
    Handles loading images from disk, detecting faces using MTCNN,
    cropping face regions with padding, resizing, and optionally saving images.
    """

    def __init__(self, images_path, save_examples_dir='examples'):
        """
        Args:
            images_path (str): Path to directory containing images (*.jpg).
            save_examples_dir (str): Directory to save example images.
        """
        if not os.path.isdir(images_path):
            raise ValueError(f"Invalid image directory: {images_path}")

        self.images_path = images_path
        self.save_examples_dir = save_examples_dir

            
        self.images = [
            os.path.join(images_path, f)
            for f in os.listdir(images_path)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        if not self.images:
            raise ValueError(f"No valid images found in: {images_path}")

        self.detector = MTCNN()
        self.save_dir = save_examples_dir
        os.makedirs(self.save_dir, exist_ok=True)


    def load_image(self, img_path):
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")

        if len(img.shape) != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected 3-channel image, got shape {img.shape} for {img_path}")

        return img


    def get_new_image(self, idx=None, img_path=None, max_attempts=20):
        """
        Load an image either randomly from images_path or from a provided path.
        Save a copy of the loaded image with the index in save_dir.

        Args:
            n (int): index for naming saved example.
            img_path (str): optional explicit image path to load.

        Returns:
            np.ndarray: RGB image array.
        """
        if not self.images:
            raise ValueError("No images available in DataLoader")

        attempts = 0
        tried_paths = set()

        while attempts < max_attempts:
            attempts += 1

            # choose input path
            if img_path is not None and attempts == 1:
                candidate = img_path
            elif idx is not None and attempts == 1:
                candidate = self.images[idx]
            else:
                candidate = random.choice(self.images)

            if candidate in tried_paths:
                continue
            tried_paths.add(candidate)

            try:
                img = self.load_image(candidate)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # if you do face detection here, validate result
                boxes, _ = self.detector.detect(rgb_img)
                if boxes is None or len(boxes) == 0:
                    continue

                return rgb_img

            except Exception as e:
                print(f"[!] Skipping image {candidate}: {e}")
                continue

        raise RuntimeError(
            f"Could not find a usable image with detectable face after {max_attempts} attempts"
        )

    def find_faces(self, n, img):
        """
        Detect faces, crop with padding, resize to 64x64, save cropped faces.

        Args:
            n (int): index for naming saved faces.
            img (np.ndarray): RGB image.

        Returns:
            np.ndarray: Array of resized face images (N, 64, 64, 3).
        """
        if img is None:
            return []

        face_locations = self.detector.detect_faces(img)

        if face_locations is None:
            return []

        img_height, img_width, _ = img.shape
        image_faces = []

        for i, fl in enumerate(face_locations):
            x, y, width, height = fl['box']
            top = max(y - int(0.15 * height), 0)
            right = min(x + width + int(0.15 * width), img_width)
            bottom = min(y + height + int(0.15 * height), img_height)
            left = max(x - int(0.15 * width), 0)

            face_crop = img[top:bottom, left:right, :]
            face_img = transform.resize(face_crop, (64, 64))
            io.imsave(os.path.join(self.save_dir, f'fig{n}_face{i+1}.png'), face_img)
            image_faces.append(face_img)

        return np.array(image_faces)
