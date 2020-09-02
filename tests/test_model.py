import numpy as np

from model import get_random_normal

def test_get_random_normal():
    result = get_random_normal(10000)
    print(np.mean(result))
    assert np.allclose(np.mean(result), 0., atol=0.5)
