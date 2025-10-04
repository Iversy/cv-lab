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
