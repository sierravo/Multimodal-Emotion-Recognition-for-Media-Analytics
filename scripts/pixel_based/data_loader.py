import os
import numpy as np
import cv2
from mtcnn import MTCNN
from skimage import io, transform
from glob import glob

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
        self.images = glob(images_path + '*.jpg')
        self.detector = MTCNN()
        self.save_dir = save_examples_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def get_new_image(self, n, img_path=None):
        """
        Load an image either randomly from images_path or from a provided path.
        Save a copy of the loaded image with the index in save_dir.

        Args:
            n (int): index for naming saved example.
            img_path (str): optional explicit image path to load.

        Returns:
            np.ndarray: RGB image array.
        """
        if img_path:
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            face_locations = self.detector.detect_faces(img)
            print(f"{img_path}: {len(face_locations)} faces detected")
            io.imsave(os.path.join(self.save_dir, f'fig{n}_full.png'), img)
            return img

        while True:
            rand_img_path = np.random.choice(self.images)
            img = cv2.cvtColor(cv2.imread(rand_img_path), cv2.COLOR_BGR2RGB)
            face_locations = self.detector.detect_faces(img)
            if len(face_locations) > 0:
                print(f"{rand_img_path}: {len(face_locations)} faces detected")
                io.imsave(os.path.join(self.save_dir, f'fig{n}_full.png'), img)
                return img

    def find_faces(self, n, img):
        """
        Detect faces, crop with padding, resize to 64x64, save cropped faces.

        Args:
            n (int): index for naming saved faces.
            img (np.ndarray): RGB image.

        Returns:
            np.ndarray: Array of resized face images (N, 64, 64, 3).
        """
        face_locations = self.detector.detect_faces(img)
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
