from functools import reduce
from itertools import product
from typing import Optional

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

Image = np.ndarray


def int2float(image: np.ndarray) -> np.ndarray:
    type_ = image.dtype
    if not np.issubdtype(type_, np.unsignedinteger):
        raise TypeError(f"{type_} is not an unsigned integer")
    return image.astype(float) / (1 << type_.itemsize*8)


def float2int(image: np.ndarray) -> np.ndarray:
    return (image*256).astype(np.uint8)


def panels() -> tuple[Axes, Axes]:
    fig = plt.figure(figsize=(12, 6))
    return fig.subplots(1, 2)


def imshow(image: np.ndarray, axis=None, cmap=None):
    assert np.issubdtype(type_ := image.dtype, np.floating), \
        f"I'm not gona show your fatty {type_} image"
    if not axis:
        plt.figure(dpi=200)
    axis = axis or plt
    axis.imshow(
        np.clip(image, 0, 1),
        cmap=cmap or 'gray',
    )
    axis.axis('off')


def plot(func, axes: Axes, label: str, *args, **kwargs):
    x = np.linspace(0, 1, 1000)
    data = dict(x=x, y=func(x, *args, **kwargs))
    sns.lineplot(data, x='x', y='y', ax=axes, label=label)


def pad(image: np.ndarray, vertical: int, horizontal: Optional[int] = None):
    padding = [(vertical,)*2, (horizontal or vertical,)*2]
    for _ in range(len(image.shape)-2):
        padding.append((0, 0))
    return np.pad(image, padding, mode='edge')


def linear(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    w, h, *_ = image.shape
    assert kernel.shape[0] == kernel.shape[1]
    k, *_ = kernel.shape
    padded = pad(image, k//2)
    return reduce(np.add, (
        padded[x:x+w, y:y+h, ...] * mul
        for mul, (x, y) in zip(
            kernel.ravel(),
            product(range(k), repeat=2)
        )
    ))


def convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    k, *_ = kernel.shape
    h, w, *_ = image.shape
    height = h - k + 1
    width = w - k + 1
    output = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            output[y, x] = np.sum(image[y:y+k, x:x+k] * kernel)
    return output


def demo(func, image: np.ndarray, name: str, formula: str = ''):
    def inner(*args, **kwargs):
        left, right = panels()
        plt.suptitle(name)
        imshow(func(image, *args, **kwargs), left)
        plot(func, right, label=formula, *args, **kwargs)
    return inner


def to_matrix(string: str) -> np.ndarray:
    try:
        matrix = string.split("\n")

    except:
        raise Exception


def is_matrix():
    pass


def intensity_image(image: np.ndarray) -> np.ndarray:
    return np.mean(image, axis=2)


def sobel(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    operator = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])

    return linear(image, operator), linear(image, operator.T)
