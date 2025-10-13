import numpy as np


def salt(image: np.ndarray, p: float) -> np.ndarray:
    noise = np.random.rand(*image.shape)
    return np.select(
        (noise < p, noise > (1-p)),
        (0, 1),
        default=image,
    )


def normal(image: np.ndarray, scale: float) -> np.ndarray:
    return image * np.random.normal(1, scale, image.shape)
