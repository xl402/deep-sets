from sklearn.metrics.pairwise import euclidean_distances
from tensorflow.keras.utils import Sequence
import numpy as np


class NearestNeighbourGenerator(Sequence):
    def __init__(self, dim: int, max_len: int, data_size: int, batch_size=1, seed=0):
        assert data_size > batch_size, 'data size must be larger than batch'
        assert max_len >= 2, 'sequence length is 2 at minimum'
        assert dim >= 1, 'data dimension must be at least 1'
        self.dim = dim
        self.max_len = max_len
        self.data_size = data_size
        self.batch_size = batch_size
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def __repr__(self):
        params = 'dim={}, max_len={}, data_size={}, batch_size={}, seed={}'
        params = params.format(self.dim, self.max_len,
                               self.data_size, self.batch_size, self.seed)
        out = 'NearestNeighbourGenerator({})'
        return out.format(params)

    def __len__(self):
        return self.data_size // self.batch_size

    def __getitem__(self, idx):
        seq_len = self.rng.randint(2, self.max_len + 1)
        x_data, y_data = [], []
        for i in range(self.batch_size):
            xs, ys = self._get_nearest_neighbours(seq_len, self.dim)
            x_data.append(xs)
            y_data.append(ys)
        return np.array(x_data), np.array(y_data)

    def _get_nearest_neighbours(self, sequence_len, dim):
        xs = self.rng.randn(sequence_len, dim)
        distances = euclidean_distances(xs, xs)
        np.fill_diagonal(distances, np.inf)
        nearest_idx = np.argmin(distances, axis=1)
        nearest_x = xs[nearest_idx]
        ys = np.linalg.norm(nearest_x, ord=1, axis=1)
        ys = np.expand_dims(ys, axis=-1)
        return xs, ys
