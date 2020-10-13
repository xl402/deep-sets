import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def get_nearest_neighbours(dim, size):
    xs = np.random.randn(size, dim)
    distances = euclidean_distances(xs, xs)
    np.fill_diagonal(distances, np.inf)
    nearest_idx = np.argmin(distances, axis=1)
    nearest_x = xs[nearest_idx]
    ys = np.linalg.norm(nearest_x, ord=1, axis=1)
    return xs, ys
