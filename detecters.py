from itertools import product
from typing import Iterator

import numpy as np

from filters import DoG, intensity_image, linear
from utils import sobel as sobel_


def doger(image: np.ndarray, sigma=1, alpha=1.6) -> np.ndarray:
    return DoG(intensity_image(image), sigma, alpha)


def neigh4(
    i: int, j: int, width: int, height: int
) -> Iterator[tuple[int, int]]:
    if i > 0:
        yield i-1, j
    if i+1 < width:
        yield i+1, j
    if j > 0:
        yield i, j-1
    if j+1 < height:
        yield i, j+1


def walk(sectors: np.ndarray, start: tuple[int, int], paint: int):
    stack = [start]
    while stack:
        cur = stack.pop()
        sectors[cur] = paint
        for neig in neigh4(*cur, *sectors.shape[:2]):
            if sectors[neig]:
                continue
            stack.append(neig)


def edges2sectors(edges: np.ndarray) -> np.ndarray:
    sectors = np.where(edges, 1, 0)
    n_sectors = 1
    for cur in product(*map(range, edges.shape[:2])):
        assert len(cur) == 2
        if sectors[cur]:
            continue
        n_sectors += 1
        walk(sectors, cur, n_sectors)

    sectors -= 1
    return sectors


def sobel(image: np.ndarray) -> np.ndarray:
    parts = sobel_(image)
    return np.abs(parts[0]) + np.abs(parts[1])
