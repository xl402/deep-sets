import itertools
from hashlib import sha1

from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import pytest

from demo.data import NearestNeighbourGenerator


DATA_DIMS = [1, 10, 100]
MAX_SEQ_LEN = [2, 100]
BATCH_SIZE = [1, 5, 10]


def test_nearest_neighbour_generator_has_repr():
    params = {'dim': 5, 'max_len': 10, 'data_size': 10, 'batch_size': 2}
    nn_generator = NearestNeighbourGenerator(**params)
    for key, value in params.items():
        assert '{}={}'.format(key, str(value)) in repr(nn_generator)


def test_nearest_neighbour_generator_raises_on_bad_data_dimension():
    params = {'dim': -1, 'max_len': 10, 'data_size': 10, 'batch_size': 2}
    with pytest.raises(AssertionError) as e:
        nn_generator = NearestNeighbourGenerator(**params)
    assert 'data dimension must be at least 1' in str(e)


def test_nearest_neighbour_generator_raises_when_batch_size_greater_than_data_size():
    params = {'dim': 1, 'max_len': 10, 'data_size': 10, 'batch_size': 20}
    with pytest.raises(AssertionError) as e:
        nn_generator = NearestNeighbourGenerator(**params)
    assert 'data size must be larger than batch' in str(e)


def test_nearest_neighbour_generator_raises_when_max_len_is_less_than_two():
    params = {'dim': 10, 'max_len': 1, 'data_size': 10, 'batch_size': 5}
    with pytest.raises(AssertionError) as e:
        nn_generator = NearestNeighbourGenerator(**params)
    assert 'sequence length is 2 at minimum' in str(e)


@pytest.mark.parametrize('dim, max_len, batch_size',
                         itertools.product(DATA_DIMS, MAX_SEQ_LEN, BATCH_SIZE))
def test_nearest_neighbour_generator_yields_correct_data_shape(dim, max_len, batch_size):
    params = {'dim': dim, 'max_len': max_len, 'data_size': 100, 'batch_size': batch_size}
    nn_generator = NearestNeighbourGenerator(**params)
    for x, y in nn_generator:
        batch_x, seq_x, dim_x = x.shape
        batch_y, seq_y = y.shape
        assert batch_x == batch_size
        assert seq_x == seq_y
        assert seq_x <= max_len
        assert dim_x == dim


@pytest.mark.parametrize('dim, max_len, batch_size',
                         itertools.product(DATA_DIMS, MAX_SEQ_LEN, BATCH_SIZE))
def test_nearest_neighbour_generator_yields_correct_ground_truth(dim, max_len, batch_size):
    params = {'dim': dim, 'max_len': max_len, 'data_size': 100, 'batch_size': batch_size}
    nn_generator = NearestNeighbourGenerator(**params)
    for xs, ys in nn_generator:
        for i in range(batch_size):
            _test_nearest_neighbour_ground_truth(xs[i], ys[i])


def _test_nearest_neighbour_ground_truth(xs, ys):
    x_norms = np.linalg.norm(xs, ord=1, axis=1)
    nearest_neighbours = xs[[np.argwhere(x_norms == y)[0, 0] for y in ys]]
    for idx, x in enumerate(xs):
        distance_to_nearest = np.linalg.norm(x - nearest_neighbours[idx])
        distance_to_others = np.linalg.norm(xs - x, axis=1)
        distance_to_others[idx] = np.inf
        assert np.all(distance_to_others >= distance_to_nearest - 1e8)


def test_nearest_neighbour_generator_yields_reproducible_data():
    params = {'dim': 2, 'max_len': 10, 'data_size': 10, 'batch_size': 2, 'seed': 0}
    results = dict()
    for i in range(2):
        nn_generator = NearestNeighbourGenerator(**params)
        for x, y in nn_generator:
            x_hash = sha1(x).hexdigest()
            if x_hash not in results:
                results[x_hash] = y
            else:
                print('hi')
                assert np.all(results[x_hash] == y)
    assert len(results) == nn_generator.__len__() 
