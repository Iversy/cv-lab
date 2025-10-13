from functools import reduce
from itertools import product
from typing import Optional
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes


def imshow(image: np.ndarray, axis=None):
    axis = axis or plt
    if len(image.shape) <= 2:
        image = np.tile(image[..., np.newaxis], 3)
    axis.imshow(np.clip(image, 0, 1))
    axis.axis('off')


def plot(func, axes: Axes, label: str, *args, **kwargs):
    x = np.linspace(0, 1, 1000)
    data = dict(x=x, y=func(x, *args, **kwargs))
    sns.lineplot(data, x='x', y='y', ax=axes, label=label)


def pad(image: np.ndarray, vertical: int, horizontal: Optional[int]=None):
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


def panels() -> tuple[Axes, Axes]:
    fig = plt.figure(figsize=(12, 6))
    return fig.subplots(1, 2)


def demo(func, image: np.ndarray, name: str, formula: str = ''):
    def inner(*args, **kwargs):
        left, right = panels()
        plt.suptitle(name)
        imshow(func(image, *args, **kwargs), left)
        plot(func, right, label=formula, *args, **kwargs)
    return inner
