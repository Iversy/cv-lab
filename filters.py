from functools import reduce
from itertools import product
import numpy as np

from utils import pad, intensity_image, sobel, convolution

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
    kernel /= sum(kernel)
    print(f"Kernel size {len(kernel)}")
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

def harris(image: np.ndarray, a=0.04, threshold=0.01)-> np.ndarray:
    if image.ndim == 3:
        image = intensity_image(image)
    Ix, Iy = sobel(image)
    
    Ix2, Ixy = sobel(Ix)
    Iy2, _ = sobel(Iy)
    
    Sx2 = blur(Ix2, scale=1)
    Sy2 = blur(Iy2, scale=1)
    Sxy = blur(Ixy, scale=1)
    
    det = Sx2 * Sy2 - Sxy ** 2
    trace = Sx2 + Sy2
    res = det - a * trace ** 2
    
    corners = np.zeros_like(res)
    corners[res > threshold * np.max(res)] = 1

    return corners

def gesse(image: np.ndarray, threshold=0.01)-> np.ndarray:
    if image.ndim == 3:
        image = intensity_image(image)
    Ix, Iy = sobel(image)
    
    Ix2, Ixy = sobel(Ix)
    Iy2, _ = sobel(Iy)
    
    Sx2 = blur(Ix2, scale=1)
    Sy2 = blur(Iy2, scale=1)
    Sxy = blur(Ixy, scale=1)
    
    a = 1
    b = -2 * Sx2 * Sy2
    c = Sx2 * Sy2 - 2 * Sxy
    sD = np.sqrt(b ** 2 - 4 * a * c) 
    l1 = (-b + sD) / 2 
    l2 = (-b - sD) / 2 
    mask1 = l1 > threshold 
    mask2 = l2 > threshold
    mask = mask1 & mask2
    
    corners = np.zeros_like(l1)
    corners[mask] = 1

    return corners

def DoG(image: np.ndarray, sigma=1, alpha=1.6) -> np.ndarray:
    shakal1 = blur(image, scale=sigma)
    shakal10 = blur(image,scale=sigma*alpha)
    return shakal10 - shakal1
    
def LoG(image: np.ndarray, sigma=1, alpha=1.6) -> np.ndarray:
    return DoG(image,sigma,alpha) / ((alpha - 1) * sigma**2)