import numpy as np
from PIL import Image
from scipy.spatial.distance import cosine
from skimage import io, color, feature


def generate_digital_pattern(image_path):
    """Generate a HOG-based digital fingerprint from a handwritten image."""
    image = io.imread(image_path)

    if not isinstance(image, np.ndarray):
        image = np.array(image.convert('RGB'))

    # Handle RGBA images
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    # Convert to grayscale manually
    if image.ndim == 3:
        grayscale_array = (
            0.2125 * image[:, :, 0] +
            0.7154 * image[:, :, 1] +
            0.0721 * image[:, :, 2]
        )
    else:
        grayscale_array = image.astype(float)

    grayscale_image = Image.fromarray(grayscale_array.astype(np.uint8))

    features = feature.hog(grayscale_image, pixels_per_cell=(16, 16))
    return features


def compare_patterns(pattern1, pattern2):
    """Compare two HOG patterns using cosine similarity."""
    similarity = 1 - cosine(pattern1, pattern2)
    return similarity
