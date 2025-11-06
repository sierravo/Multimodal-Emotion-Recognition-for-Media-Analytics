import numpy as np
import torch
from skimage import transform

class ANRescale:
    """
    Rescale image to the given output size for AffectNet models.
    """
    def __init__(self, output_size=(224, 224)):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        return transform.resize(image, self.output_size)

class ANToTensor:
    """
    Convert numpy image (H, W, C) to PyTorch tensor (C, H, W).
    """
    def __call__(self, image):
        return torch.from_numpy(image.transpose((2, 0, 1))).float()

class SFNormalize:
    """
    Normalize SongFan images by mean and std.
    """
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image):
        return (image / 255. - self.mean) / self.std

class SFRescale:
    """
    Resize images to two sizes for SongFan input.
    """
    def __init__(self, output_size_1=(600,800), output_size_2=(300,400)):
        self.output_size_1 = output_size_1
        self.output_size_2 = output_size_2

    def __call__(self, image):
        img1 = transform.resize(image, self.output_size_1)
        img2 = transform.resize(image, self.output_size_2)
        return img1, img2

class SFToTensor:
    """
    Convert tuple of numpy images to tuple of PyTorch tensors.
    """
    def __call__(self, sample):
        img1, img2 = sample
        img1 = torch.from_numpy(img1.transpose((2, 0, 1))).float()
        img2 = torch.from_numpy(img2.transpose((2, 0, 1))).float()
        return img1, img2
