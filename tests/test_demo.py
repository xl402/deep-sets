import itertools

from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import pytest

from demo.data import NearestNeighbourGenerator


DATA_DIMS = [1, 10, 100]
DATA_SIZE = [1, 1000]

@pytest.mark.parametrize('data_dim, data_size',
                         itertools.product(DATA_DIMS, DATA_SIZE))
def _test_nearest_neighbour_generator(data_dim, data_size):
    xs, ys = data.get_nearest_neighbours(data_dim, data_size)
    x_norms = np.linalg.norm(xs, ord=1, axis=1)
    nearest_neighbours = xs[[np.argwhere(x_norms == y)[0, 0] for y in ys]]

    for idx, x in enumerate(xs):
        distance_to_nearest = np.linalg.norm(x - nearest_neighbours[idx])
        distance_to_others = np.linalg.norm(xs - x, axis=1)
        distance_to_others[idx] = np.inf
        assert np.all(distance_to_others >= distance_to_nearest - 1e8)



def test_nearest_neighbour_generator():
    params = {'dim': 2, 'max_len': 10, 'data_size': 10, 'batch_size': 2}
    nn_generator = NearestNeighbourGenerator(**params)
    for x, y in nn_generator:
        print(y.shape)
