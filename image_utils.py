"""
Image Feature Extraction Utilities
Course: Introduction to Artificial Intelligence (AI-101L)
Author: Mohsin Raza (SAP ID: 70177136)
Email: 70177136@student.uol.edu.pk
Institution: The University of Lahore (UOL)
"""

import numpy as np
from PIL import Image


def image_to_features(path, size=(64, 64)):
    """Load, resize, grayscale and flatten an image into a 1-D feature vector.

    Parameters
    ----------
    path : str
        Path to an image file.
    size : tuple[int, int], optional
        Target (width, height) resolution. Default is (64, 64).
        Every image is forced to this uniform size so that every returned 
        feature vector has identical length (size[0] * size[1]), meeting
        the input requirements of classical linear ML models.

    Returns
    -------
    numpy.ndarray
        A 1-D array of shape (size[0] * size[1],) with dtype uint8 (pixel intensities 0-255).
    """
    # 1. Open image and ensure 3-channel RGB format to standardize varying input formats (RGBA, CMYK, etc.)
    img = Image.open(path).convert("RGB")
    
    # 2. Resize to consistent spatial dimensions
    img = img.resize(size)
    
    # 3. Convert to Grayscale ('L' luminance channel = 0.299R + 0.587G + 0.114B)
    img = img.convert("L")
    
    # 4. Flatten the 2D matrix (H x W) into a 1D vector of length (H * W)
    feature_vector = np.array(img).flatten()
    
    return feature_vector


if __name__ == "__main__":
    print("image_utils module loaded successfully.")
