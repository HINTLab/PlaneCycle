import numpy as np


def rotation(array, angle):
    X = np.rot90(array, angle[0], axes=(0, 1))  # rotate in X-axis
    Y = np.rot90(X, angle[1], axes=(0, 2))  # rotate in Y'-axis
    Z = np.rot90(Y, angle[2], axes=(1, 2))  # rotate in Z"-axis
    return Z


def crop(array, zyx, dhw):
    y, x = zyx
    h, w = dhw
    cropped = array[:, :, y - h // 2:y + h // 2, x - w // 2:x + w // 2]
    return cropped


def random_center(shape, move):
    offset = np.random.randint(-move, move + 1, size=2)
    zy = np.array(shape) // 2 + offset
    return zy
