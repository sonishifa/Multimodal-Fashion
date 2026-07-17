"""
Loads images and prepares batches for indexing.
"""

import os
from PIL import Image


def load_image_paths(image_dir):
    """
    Returns sorted list of image paths.
    """

    image_paths = []

    for file in sorted(os.listdir(image_dir)):
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        ):
            image_paths.append(
                os.path.join(image_dir, file)
            )

    return image_paths


def load_images(paths):
    """
    Loads PIL images.
    """

    images = []

    for p in paths:
        images.append(
            Image.open(p).convert("RGB")
        )

    return images


def batch_iterator(items, batch_size):
    """
    Yield batches.
    """

    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]