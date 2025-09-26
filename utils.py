from matplotlib import pyplot as plt
import numpy as np

def imshow(image: np.ndarray):
    if len(image.shape) <= 2:
        image = np.tile(image[..., np.newaxis], 3)
    plt.imshow(image)
    plt.axis('off')
