from functools import reduce
from itertools import product
import numpy as np

from utils import pad

def absdif(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.abs(left-right)

def linear(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    w, h, *_ = image.shape
    assert len(kernel.shape) == 2
    k_w, k_h = kernel.shape
    padded = pad(image, k_w//2, k_h//2)
    return reduce(np.add, (
        padded[x:x+w, y:y+h, ...] * mul
        for mul, (x, y) in zip(
            kernel.ravel(),
            product(range(k_w), range(k_h))
        )
    ))


def average(image: np.ndarray, size: int) -> np.ndarray:
    kernel = np.full((size, size), size**-2)
    return linear(image, kernel)


def median(image: np.ndarray, window: int) -> np.ndarray:
    w, h, *_ = image.shape
    assert window & 1
    padded = pad(image, window//2)
    return np.median(
        [ padded[x:x+w, y:y+h, ...]
          for x, y in product(range(window), repeat=2) ],
        axis=0
    )


def G1(loc: float, scale: float, x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * ((x-loc)/scale)**2) / np.sqrt(2*np.pi) / scale


def blur(image: np.ndarray, scale: float) -> np.ndarray:
    x = np.arange(np.floor(-3*scale), np.ceil(3*scale)+1)
    kernel = G1(0, scale, x)[..., np.newaxis]
    return linear(linear(image, kernel), kernel.T)

def soft(image: np.ndarray, window: int, sigma: float) -> np.ndarray:
    assert window & 1, "Window can't be even"
    w, h, *_ = image.shape

    padded = pad(image, window//2)
    cursum = np.zeros_like(image)
    curcount = np.zeros(image.shape, int)
    for x, y in product(range(window), repeat=2):
        sloice = padded[x:x+w, y:y+h, ...]
        good = (sloice >= image-sigma) & (sloice <= image + sigma)
        curcount += good
        cursum[good] += sloice[good]
    return cursum / curcount

def unsharp(image: np.ndarray, radius: float, amount: float) -> np.ndarray:
    blurred = blur(image, radius)
    return image + (image - blurred) * amount